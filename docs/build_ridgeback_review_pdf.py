#!/usr/bin/env python3
"""Build the dated Ridgeback deployment review from verified project facts."""

from pathlib import Path
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/pdf/OmTrackVLA_Ridgeback_technical_review_2026-09-29.pdf"
WIDTH, HEIGHT = 960, 540
NAVY = colors.HexColor("#122033")
BLUE = colors.HexColor("#2459B8")
TEAL = colors.HexColor("#117A74")
AMBER = colors.HexColor("#AA6517")
PALE = colors.HexColor("#F3F6FA")
MID = colors.HexColor("#D7DFE8")
TEXT = colors.HexColor("#283443")
MUTED = colors.HexColor("#586778")
WHITE = colors.white


def setup_fonts():
    base = Path("/usr/share/fonts/truetype/dejavu")
    pdfmetrics.registerFont(TTFont("DejaVu", str(base / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(base / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold")


def paragraph(c, text, x, top, width, size=12, color=TEXT, bold=False,
              leading=None, align=TA_LEFT):
    style = ParagraphStyle(
        "custom", fontName="DejaVu-Bold" if bold else "DejaVu",
        fontSize=size, leading=leading or size * 1.32, textColor=color,
        alignment=align, spaceAfter=0, spaceBefore=0,
    )
    item = Paragraph(text, style)
    _, height = item.wrap(width, HEIGHT)
    item.drawOn(c, x, top - height)
    return top - height


def page(c, number, section, title, subtitle=None):
    c.setFillColor(NAVY)
    c.rect(0, HEIGHT - 13, WIDTH, 13, fill=1, stroke=0)
    paragraph(c, section.upper(), 46, HEIGHT - 36, 850, 9, BLUE, True)
    paragraph(c, title, 46, HEIGHT - 60, 870, 24, NAVY, True, 29)
    if subtitle:
        paragraph(c, subtitle, 46, HEIGHT - 96, 865, 11, MUTED)
    c.setStrokeColor(MID)
    c.line(46, 32, WIDTH - 46, 32)
    paragraph(c, "OmTrackVLA / Ridgeback r100-0160  |  29 Sep 2026", 46, 25, 800, 8, MUTED)
    paragraph(c, str(number), WIDTH - 70, 25, 24, 8, MUTED, align=TA_CENTER)


def panel(c, x, top, width, height, title, body, border=BLUE, fill=PALE,
          title_size=13, body_size=10.5):
    y = top - height
    c.setFillColor(fill)
    c.setStrokeColor(MID)
    c.roundRect(x, y, width, height, 9, fill=1, stroke=1)
    c.setFillColor(border)
    c.roundRect(x, y, 5, height, 2, fill=1, stroke=0)
    paragraph(c, title, x + 17, top - 13, width - 30, title_size, NAVY, True)
    paragraph(c, body, x + 17, top - 40, width - 30, body_size, TEXT)


def line(c, x1, y1, x2, y2, color=BLUE, arrow=False, width=1.8):
    c.setStrokeColor(color)
    c.setLineWidth(width)
    c.line(x1, y1, x2, y2)
    if arrow:
        import math
        angle = math.atan2(y2 - y1, x2 - x1)
        length = 7
        for delta in (-0.52, 0.52):
            c.line(x2, y2, x2 - length * math.cos(angle + delta),
                   y2 - length * math.sin(angle + delta))


def row(c, top, label, value, yheight=38, label_width=210, x=48, width=865,
        alternate=False, label_size=10.2, value_size=10.2):
    if alternate:
        c.setFillColor(PALE)
        c.rect(x, top-yheight, width, yheight, fill=1, stroke=0)
    paragraph(c, label, x + 10, top - 9, label_width - 15, label_size, NAVY, True)
    paragraph(c, value, x + label_width + 8, top - 9, width-label_width-20,
              value_size, TEXT)
    c.setStrokeColor(MID)
    c.line(x, top-yheight, x+width, top-yheight)


def source(c, text):
    paragraph(c, text, 48, 48, 860, 8, MUTED)


def title_page(c):
    c.setFillColor(NAVY)
    c.rect(0, 0, WIDTH, HEIGHT, fill=1, stroke=0)
    c.setFillColor(BLUE)
    c.rect(0, HEIGHT - 13, WIDTH, 13, fill=1, stroke=0)
    paragraph(c, "TECHNICAL REVIEW  /  29 SEPTEMBER 2026", 56, 455, 850, 12,
              colors.HexColor("#8FB7FF"), True)
    paragraph(c, "OmTrackVLA on Clearpath Ridgeback", 56, 401, 840, 32,
              WHITE, True, 38)
    paragraph(c, "Current two-GPU split, target identity, safety, and measured evidence",
              56, 347, 790, 17, colors.HexColor("#CED9E7"))
    panel(c, 56, 261, 260, 124, "RIDGEBACK RTX 5060",
          "YOLO11n + Grounding DINO. Ridgeback CPU tracks identity and owns the 20 Hz motion gate.",
          BLUE, colors.HexColor("#EAF1FF"), 12.5, 10.5)
    panel(c, 349, 261, 260, 124, "NVIDIA THOR",
          "DINOv3 + SigLIP + Qwen3 + OmTrackVLA on Thor CUDA. Its CPU handles the service and cache.",
          TEAL, colors.HexColor("#E8F6F3"), 12.5, 10.5)
    panel(c, 642, 261, 260, 124, "CURRENT STATUS",
          "Hybrid mode is implemented and dry-run only. Live locked-target cadence and physical motion remain unverified.",
          AMBER, colors.HexColor("#FFF4E7"), 12.5, 10.5)
    paragraph(c, "Project-specific integration; upstream OmTrackVLA remains the waypoint model.",
              56, 57, 840, 10, colors.HexColor("#B9C9DC"))
    c.showPage()


def architecture_page(c):
    page(c, 2, "System architecture", "The hybrid split uses both GPUs")
    panel(c, 48, 425, 405, 270, "RIDGEBACK COMPUTER  |  192.168.131.1",
          "<b>RTX GPU:</b> YOLO11n person boxes; Grounding DINO when searching or recovering.<br/><br/>"
          "<b>CPU:</b> BoT-SORT association, optical flow, HSV clothing gallery, ROS sensors, depth position, 75/25 command fusion, 20 Hz safety gate, /cmd_vel.",
          BLUE, PALE, 14, 12)
    panel(c, 506, 425, 405, 270, "NVIDIA THOR  |  192.168.131.51",
          "<b>Thor GPU:</b> DINOv3, SigLIP, Qwen3, OmTrackVLA waypoint model and visual history.<br/><br/>"
          "<b>Thor CPU:</b> TCP/JSON service, JPEG decode, request validation and planner cache.<br/><br/>"
          "<b>Interface:</b> project TCP protocol on port 18765; Thor is not a ROS inference node.",
          TEAL, PALE, 14, 12)
    line(c, 452, 276, 504, 276, BLUE, True, 2.5)
    line(c, 504, 244, 452, 244, TEAL, True, 2.5)
    paragraph(c, "Ridgeback to Thor: same JPEG and target only while LOCKED. Otherwise, target metadata without a JPEG.",
              63, 133, 835, 11, TEXT)
    paragraph(c, "Thor to Ridgeback: waypoints and a matching target result, checked before command fusion.",
              63, 102, 835, 11, TEXT)
    source(c, "Implementation: real_robot/inference_server.py, ridgeback_ros2_node.py, hybrid_protocol.py")
    c.showPage()


def sequence_page(c):
    page(c, 3, "One frame and identity", "Perception finishes before the Thor planner request")
    steps = [
        ("1  SELECT", "Ridgeback selects a compressed RGB frame for the target loop (up to 3 Hz)."),
        ("2  FIND", "RTX YOLO detects all people. Grounding DINO checks the prompt attribute when due."),
        ("3  VERIFY", "CPU BoT-SORT and clothing features keep the same person; only LOCKED is valid."),
        ("4  PLAN", "For LOCKED, Thor receives the same JPEG and target. Its planner updates at up to 2 Hz; cached paths have a 0.75 s age limit."),
        ("5  CONTROL", "Ridgeback checks reply identity and freshness, localizes depth, fuses direction, and runs the 20 Hz gate."),
    ]
    y = 421
    for index, (label, desc) in enumerate(steps):
        panel(c, 48, y, 865, 61, label, desc, BLUE if index < 3 else TEAL,
              WHITE if index % 2 else PALE, 11.5, 10.4)
        y -= 68
    paragraph(c, "Prompt example: \"Follow the person who is holding blue basket.\" Grounding checks the basket; the HSV signature describes the person's visible clothing. Recovery checks both prompt and appearance.",
              61, 73, 833, 10, MUTED)
    c.showPage()


def safety_page(c):
    page(c, 4, "Fusion and safety", "Ridgeback makes the final motion decision",
         "The locked leader and the model path meet in the local controller.")
    panel(c, 48, 421, 406, 190, "LEADER DIRECTION",
          "Ridgeback projects the locked person's box into a depth-measured position. The controller aims toward that position while respecting a 0.9 m follow distance.",
          BLUE, PALE, 14, 12)
    panel(c, 507, 421, 405, 190, "MODEL DIRECTION",
          "Thor's OmTrackVLA predicts eight local waypoints. Waypoint index 1 supplies the raw direction; the model does not report who it is following.",
          TEAL, PALE, 14, 12)
    line(c, 251, 230, 251, 216, BLUE, True, 2.5)
    line(c, 709, 230, 709, 216, TEAL, True, 2.5)
    panel(c, 48, 214, 865, 112, "DIRECTION FUSION ON RIDGEBACK CPU",
          "When the model direction agrees with the leader, the controller blends about <b>75% leader direction and 25% aligned model direction</b>. An opposing or sideways model proposal is replaced by the leader direction. This ratio describes command fusion, not the share of GPU computation.",
          AMBER, colors.HexColor("#FFF4E7"), 14, 11.5)
    paragraph(c, "Every 50 ms: dry-run, deadman, E-stop, camera/model/scan freshness, LOCKED target, depth, LiDAR clearance, speed and acceleration checks. Failure commands zero or stops publication.",
              61, 88, 835, 11, TEXT)
    c.showPage()


def evidence_page(c):
    page(c, 5, "Measured evidence", "What the September 28-29 checks establish")
    top = 423
    rows = [
        ("Full-Thor live dry-run", "187 frames; 93 LOCKED. The planner updated 31 times over 30.241 s of lock (~1.03 Hz under a 2 Hz cap). E-stop stayed active."),
        ("Hybrid offline replay", "90 current-camera blue-basket frames; 49 LOCKED. All five compared target/trajectory decision fields matched the earlier full RTX replay."),
        ("Hybrid transfer", "41 invalid-target replay frames sent target metadata to Thor without a JPEG. Ridgeback perception used about 2,390 MiB RTX memory."),
        ("Hybrid live searching", "46 no-person frames at 3.054 Hz. Bridge pipeline median/p95: 25.564/27.673 ms. 20 Hz control interval maximum: 51.455 ms."),
        ("Service failures", "Stopping either service cleared the prediction in separate dry-runs. Deadman was unheld, so physical stopping was not measured."),
        ("Tests", "66 project tests passed after the hybrid implementation. Hybrid locked-target live planner cadence and motion have not passed validation."),
    ]
    for i, (label, value) in enumerate(rows):
        row(c, top, label, value, 57, 204, alternate=i % 2 == 0,
            value_size=10.0)
        top -= 57
    paragraph(c, "These runs used different scenes and timings. They do not show a hybrid speedup over full RTX or full Thor.",
              58, 74, 844, 10.5, AMBER, True)
    source(c, "Measurements and limits: docs/COMPUTE_PLACEMENT_PLAN.md and docs/PROJECT_CONTEXT.md")
    c.showPage()


def next_page(c):
    page(c, 6, "Operator status and next checks", "Hybrid remains a dry-run launch")
    row(c, 425, "Thor terminal", "ssh robot@192.168.131.51; cd /home/robot/dev/omtrackvla; ./real_robot/start_planner_thor.sh", 56, 192, alternate=True, value_size=10.1)
    row(c, 369, "Ridgeback terminal", "cd /home/robot/Desktop/omtrackvla; set ROBOT_NAMESPACE=r100_0160, CAMERA_TOPIC=camera/color/image_raw/compressed, CAMERA_COMPRESSED=true; run ./real_robot/start_ridgeback_hybrid.sh", 79, 192, value_size=9.8)
    row(c, 290, "RViz", "On Ridgeback: ./real_robot/start_ridgeback_rviz.sh in a separate terminal.", 48, 192, alternate=True)
    paragraph(c, "Validation before hybrid motor output", 49, 225, 840, 16, NAVY, True)
    checks = [
        "Capture a live blue-basket LOCKED dry-run and measure Thor planner updates, target IDs, and stale frames.",
        "Compare full RTX, full Thor, and hybrid with the same scene; record p50/p95/max stage times, GPU load/memory, CPU/network load, and 20 Hz deadlines.",
        "Test disconnect/reconnect with the deadman held under a controlled physical safety procedure before any arming change.",
    ]
    y = 192
    for item in checks:
        c.setFillColor(TEAL)
        c.circle(57, y - 6, 3, fill=1, stroke=0)
        y = paragraph(c, item, 72, y, 820, 10.5, TEXT) - 16
    paragraph(c, "The local full RTX supervised launcher remains the only validated armable path. The hybrid launcher rejects OMTRACKVLA_ARM_OUTPUT=1.",
              49, 65, 855, 10, AMBER, True)
    c.showPage()


def main():
    setup_fonts()
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = canvas.Canvas(str(path), pagesize=(WIDTH, HEIGHT), pageCompression=1)
    doc.setTitle("OmTrackVLA Ridgeback technical review - 29 September 2026")
    doc.setAuthor("OmTrackVLA Ridgeback project")
    for builder in (title_page, architecture_page, sequence_page, safety_page,
                    evidence_page, next_page):
        builder(doc)
    doc.save()
    print(path)


if __name__ == "__main__":
    main()
