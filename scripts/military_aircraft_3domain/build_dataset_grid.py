#!/usr/bin/env python3
"""Build a 5-class by 3-domain sample grid from MilitaryAircraft3D."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


REPO = Path(__file__).resolve().parents[2]
DATASET = (
    REPO / "docs" / "figures" / "military_aircraft_dataset_samples" /
    "root" / "autodl-tmp" / "datasets" / "MilitaryAircraft3D")
OUTPUT = REPO / "docs" / "figures" / "military_aircraft_5x3_grid.png"

DOMAINS = [
    ("aerial", "航空影像域"),
    ("natural", "自然图像域"),
    ("recon", "侦察图像域"),
]
CLASSES = ["B-52", "C-130", "C-17", "F-15", "F-16"]

TILE_W, TILE_H = 420, 260
LABEL_W, HEADER_H = 160, 82
GAP = 14
MARGIN = 24


def font(size: int):
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def centered(draw, box, text, text_font, fill=(20, 20, 20)):
    left, top, right, bottom = box
    bounds = draw.textbbox((0, 0), text, font=text_font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.text(
        ((left + right - width) / 2, (top + bottom - height) / 2 - 2),
        text, font=text_font, fill=fill)


def main():
    width = MARGIN * 2 + LABEL_W + GAP + 3 * TILE_W + 2 * GAP
    height = MARGIN * 2 + HEADER_H + GAP + 5 * TILE_H + 4 * GAP
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    header_font = font(34)
    class_font = font(34)
    corner_font = font(28)

    centered(
        draw,
        (MARGIN, MARGIN, MARGIN + LABEL_W, MARGIN + HEADER_H),
        "类别/域", corner_font)

    grid_left = MARGIN + LABEL_W + GAP
    grid_top = MARGIN + HEADER_H + GAP
    for column, (_, domain_label) in enumerate(DOMAINS):
        x = grid_left + column * (TILE_W + GAP)
        centered(
            draw, (x, MARGIN, x + TILE_W, MARGIN + HEADER_H),
            domain_label, header_font)

    for row, class_name in enumerate(CLASSES):
        y = grid_top + row * (TILE_H + GAP)
        centered(
            draw, (MARGIN, y, MARGIN + LABEL_W, y + TILE_H),
            class_name, class_font)
        for column, (domain_name, _) in enumerate(DOMAINS):
            source = DATASET / domain_name / class_name / "0000.jpg"
            if not source.exists():
                raise FileNotFoundError(source)
            with Image.open(source) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                tile = ImageOps.fit(
                    image, (TILE_W, TILE_H),
                    method=Image.Resampling.LANCZOS,
                    centering=(0.5, 0.5))
            x = grid_left + column * (TILE_W + GAP)
            canvas.paste(tile, (x, y))
            draw.rectangle(
                (x, y, x + TILE_W - 1, y + TILE_H - 1),
                outline=(120, 120, 120), width=2)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT, dpi=(300, 300), optimize=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
