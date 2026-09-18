"""Small length-prefixed JSON protocol used across the Python 3.9/3.12 boundary."""

from __future__ import annotations

import json
import socket
import struct
from typing import Any, Dict


MAX_MESSAGE_BYTES = 16 * 1024 * 1024


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("peer closed the inference connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_message(sock: socket.socket, payload: Dict[str, Any]) -> None:
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(encoded) > MAX_MESSAGE_BYTES:
        raise ValueError(f"message is too large: {len(encoded)} bytes")
    sock.sendall(struct.pack("!I", len(encoded)) + encoded)


def receive_message(sock: socket.socket) -> Dict[str, Any]:
    (size,) = struct.unpack("!I", _recv_exact(sock, 4))
    if size <= 0 or size > MAX_MESSAGE_BYTES:
        raise ValueError(f"invalid message size: {size}")
    decoded = json.loads(_recv_exact(sock, size).decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("protocol payload must be a JSON object")
    return decoded
