#!/usr/bin/env python3
"""Stop stale SCIP measurement state and verify one Hokuyo identity."""

import argparse
import socket
import time


def read_response(sock: socket.socket, command: str, timeout: float = 5.0) -> list[str]:
    deadline = time.monotonic() + timeout
    lines: list[str] = []
    line = bytearray()
    found_echo = False

    while time.monotonic() < deadline:
        sock.settimeout(max(0.05, deadline - time.monotonic()))
        byte = sock.recv(1)
        if not byte:
            raise RuntimeError(f"connection closed while waiting for {command}")
        if byte in (b"\n", b"\r"):
            text = line.decode("ascii", errors="replace")
            line.clear()
            if not found_echo:
                if text == command:
                    found_echo = True
                    lines.append(text)
                continue
            if text:
                lines.append(text)
            elif len(lines) >= 2:
                return lines
            continue
        line.extend(byte)

    raise RuntimeError(f"timeout waiting for complete {command} response")


def exchange(ip_address: str, expected_serial: str) -> None:
    with socket.create_connection((ip_address, 10940), timeout=3.0) as sock:
        sock.sendall(b"QT\n")
        qt_lines = read_response(sock, "QT")
        if not qt_lines[1].startswith("00"):
            raise RuntimeError(f"QT failed: {qt_lines[1]}")

        sock.sendall(b"VV\n")
        vv_lines = read_response(sock, "VV")
        if not vv_lines[1].startswith("00"):
            raise RuntimeError(f"VV failed: {vv_lines[1]}")
        if not any("PROD:UST-20LX" in line for line in vv_lines):
            raise RuntimeError("endpoint is not a UST-20LX")
        if not any(f"SERI:{expected_serial}" in line for line in vv_lines):
            raise RuntimeError(f"expected serial {expected_serial} was not returned")

        sock.shutdown(socket.SHUT_RDWR)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ip_address")
    parser.add_argument("expected_serial")
    args = parser.parse_args()
    exchange(args.ip_address, args.expected_serial)
    print(f"Reset and verified Hokuyo {args.expected_serial} at {args.ip_address}.")


if __name__ == "__main__":
    main()
