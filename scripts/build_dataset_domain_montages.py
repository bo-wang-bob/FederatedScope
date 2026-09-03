from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(r"D:\Projects\FederatedScope")
OUT = ROOT / "docs" / "dataset_domain_examples"
OUT.mkdir(parents=True, exist_ok=True)


def font(size: int, bold: bool = False):
    candidates = [
        Path(r"C:\Windows\Fonts\timesbd.ttf" if bold else r"C:\Windows\Fonts\times.ttf"),
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


LABEL_FONT = font(30, bold=True)
SMALL_FONT = font(22)


def fit_image(path: Path, size: tuple[int, int], pad: int = 16) -> Image.Image:
    image = Image.open(path).convert("RGB")
    inner = (size[0] - pad * 2, size[1] - pad * 2)
    resampling = (
        Image.Resampling.NEAREST
        if max(image.size) <= 64
        else Image.Resampling.LANCZOS
    )
    image = ImageOps.contain(image, inner, method=resampling)
    panel = Image.new("RGB", size, "white")
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    panel.paste(image, (x, y))
    return panel


def image_montage(items, output: Path, panel_size=(360, 300)):
    label_height = 56
    gap = 18
    margin = 20
    width = margin * 2 + len(items) * panel_size[0] + (len(items) - 1) * gap
    height = margin * 2 + panel_size[1] + label_height
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    for index, (label, path) in enumerate(items):
        x = margin + index * (panel_size[0] + gap)
        y = margin
        panel = fit_image(path, panel_size)
        panel = ImageOps.expand(panel, border=2, fill="black")
        canvas.paste(panel, (x, y))
        box = draw.textbbox((0, 0), label, font=LABEL_FONT)
        text_width = box[2] - box[0]
        draw.text(
            (x + (panel_size[0] - text_width) / 2, y + panel_size[1] + 13),
            label,
            font=LABEL_FONT,
            fill="black",
        )

    canvas.save(output, dpi=(300, 300))


def text_montage(items, output: Path):
    panel_width = 390
    panel_height = 270
    gap = 18
    margin = 20
    width = margin * 2 + len(items) * panel_width + (len(items) - 1) * gap
    height = margin * 2 + panel_height
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    for index, (label, review) in enumerate(items):
        x = margin + index * (panel_width + gap)
        y = margin
        draw.rectangle(
            (x, y, x + panel_width, y + panel_height),
            outline="black",
            width=2,
        )
        draw.text((x + 20, y + 18), label, font=LABEL_FONT, fill="black")
        draw.line((x + 20, y + 62, x + panel_width - 20, y + 62), fill="black", width=1)
        lines = wrap(review, width=36)
        line_y = y + 82
        for line in lines[:6]:
            draw.text((x + 20, line_y), line, font=SMALL_FONT, fill="black")
            line_y += 30

    canvas.save(output, dpi=(300, 300))


officehome = ROOT / "OfficeHomeDataset_10072016"
image_montage(
    [
        ("Art", officehome / "Art" / "Alarm_Clock" / "00001.jpg"),
        ("Clipart", officehome / "Clipart" / "Alarm_Clock" / "00001.jpg"),
        ("Product", officehome / "Product" / "Alarm_Clock" / "00001.jpg"),
        ("Real World", officehome / "Real World" / "Alarm_Clock" / "00001.jpg"),
    ],
    OUT / "officehome_1x4_domains.png",
)


digits = ROOT / "data" / "digit_three_domain" / "images"
image_montage(
    [
        (
            "EMNIST Digits",
            digits / "emnist_digits" / "train" / "4" / "00002772.png",
        ),
        ("USPS", digits / "usps" / "train" / "4" / "00000002.png"),
        ("SVHN", digits / "svhn" / "train" / "4" / "00000229.png"),
    ],
    OUT / "mddigits_1x3_domains.png",
)


text_montage(
    [
        (
            "Books",
            "I borrowed this book on CD from the library and ordered the book so I would have it to refer to.",
        ),
        (
            "DVD",
            "If you don't own this DVD, you need to add it to your collection. It is an excellent animated film.",
        ),
        (
            "Electronics",
            "I use it in an outdoor trail camera for wild game, and it holds several pictures in all kinds of weather.",
        ),
        (
            "Kitchen",
            "Braun makes the best coffee makers, and this feature makes it even better. The coffee tastes smoother with this filter.",
        ),
    ],
    OUT / "mdsent_1x4_domains.png",
)


print(OUT / "officehome_1x4_domains.png")
print(OUT / "mddigits_1x3_domains.png")
print(OUT / "mdsent_1x4_domains.png")
