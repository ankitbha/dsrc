"""Label the rig photo in the image itself, so the caption does not carry a legend.

The photo is used whole rather than cropped. In Fig. 1 it is printed 1.48 in tall, about
1.11 in wide, so text set on it has to be large in pixels to survive the reduction: at
1100 px wide, a 6.5 pt label is about 90 px. That is why the labels are single words.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

IMAGES = Path(__file__).resolve().parents[2] / "paper" / "images"
FONT_FILE = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

PRINT_WIDTH_IN = 1.11
LABEL_PT = 6.5
WHITE = (255, 255, 255)
INK = (18, 18, 18)
PAD = 18
BORDER = 5

# (text, the point it names, the label box's top-left corner)
CALLOUTS = [
    ("phone", (522, 169), (600, 50)),
    ("Jetson", (720, 667), (640, 420)),
    ("inverter", (508, 619), (15, 430)),
    ("laptop", (396, 990), (630, 1040)),
]


def font_px(image_width: int) -> int:
    return round(LABEL_PT / 72 * image_width / PRINT_WIDTH_IN)


def draw(img: Image.Image, font, text: str, target, corner) -> tuple[int, int, int, int]:
    pen = ImageDraw.Draw(img)
    left, top, right, bottom = pen.textbbox((0, 0), text, font=font)
    x0, y0 = corner
    x1, y1 = x0 + (right - left) + 2 * PAD, y0 + (bottom - top) + 2 * PAD
    if x0 < 0 or y0 < 0 or x1 > img.width or y1 > img.height:
        raise SystemExit(f"the {text} label runs off the photo: {(x0, y0, x1, y1)}")
    if x0 <= target[0] <= x1 and y0 <= target[1] <= y1:
        raise SystemExit(f"the {text} label covers the point it names")
    # The leader leaves from the point of the box nearest the target.
    start = (min(max(target[0], x0), x1), min(max(target[1], y0), y1))
    pen.line([start, target], fill=INK, width=12)
    pen.line([start, target], fill=WHITE, width=5)
    pen.ellipse([target[0] - 16, target[1] - 16, target[0] + 16, target[1] + 16], fill=WHITE)
    shade = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(shade).rectangle([x0, y0, x1, y1], fill=INK + (217,))
    img.paste(Image.alpha_composite(img.convert("RGBA"), shade).convert("RGB"))
    pen = ImageDraw.Draw(img)
    pen.rectangle([x0, y0, x1, y1], outline=WHITE, width=BORDER)
    pen.text((x0 + PAD - left, y0 + PAD - top), text, font=font, fill=WHITE)
    return x0, y0, x1, y1


def main() -> None:
    img = Image.open(IMAGES / "deployed_prototype_upright.jpg").convert("RGB")
    font = ImageFont.truetype(FONT_FILE, font_px(img.width))
    boxes = [draw(img, font, *callout) for callout in CALLOUTS]
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                raise SystemExit(f"two labels overlap: {a} and {b}")
    out = IMAGES / "rig_dashboard.jpg"
    img.save(out, quality=93)
    print("wrote", out, img.size, "font", font_px(img.width), "px")


if __name__ == "__main__":
    main()
