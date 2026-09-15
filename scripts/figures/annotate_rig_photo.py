"""Label the rig photo in the image itself, so the caption does not carry a legend.

The photo is used whole rather than cropped. It is printed at roughly 1.9 inches
wide, so text set on it has to be large in pixels to survive the reduction: at
1100 px wide, a 6 pt label needs about 50 px of cap height. That is why the
labels are single words.
"""

from __future__ import annotations

from pathlib import Path

import cv2

IMAGES = Path("/Users/ankit_nash/Desktop/ankit_summer_2026/dsrc/paper/images")

WHITE = (255, 255, 255)
INK = (18, 18, 18)
FONT = cv2.FONT_HERSHEY_DUPLEX
SCALE = 1.9
THICK = 3
PAD = 12

# (text, the point it names, the label's anchor, which side the leader leaves from)
CALLOUTS = [
    ("phone", (522, 169), (690, 96), "left"),
    ("Jetson", (720, 667), (830, 560), "left"),
    ("inverter", (508, 619), (96, 556), "right"),
    ("laptop", (396, 990), (660, 940), "left"),
]


def draw(img, text, target, anchor, side):
    (tw, th), _ = cv2.getTextSize(text, FONT, SCALE, THICK)
    x, y = anchor
    x0, y0, x1, y1 = x, y, x + tw + 2 * PAD, y + th + 2 * PAD
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), INK, -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    cv2.rectangle(img, (x0, y0), (x1, y1), WHITE, 3, cv2.LINE_AA)
    cv2.putText(img, text, (x0 + PAD, y1 - PAD - 2), FONT, SCALE, WHITE, THICK, cv2.LINE_AA)
    start = (x0, (y0 + y1) // 2) if side == "left" else (x1, (y0 + y1) // 2)
    cv2.line(img, start, target, INK, 7, cv2.LINE_AA)
    cv2.line(img, start, target, WHITE, 3, cv2.LINE_AA)
    cv2.circle(img, target, 11, WHITE, -1, cv2.LINE_AA)


def main() -> None:
    img = cv2.imread(str(IMAGES / "deployed_prototype_upright.jpg")).copy()
    for text, target, anchor, side in CALLOUTS:
        draw(img, text, target, anchor, side)
    out = IMAGES / "rig_dashboard.jpg"
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 93])
    print("wrote", out, img.shape, "aspect", round(img.shape[1] / img.shape[0], 2))


if __name__ == "__main__":
    main()
