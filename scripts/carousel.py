"""Сборка карточек для инстаграм-карусели из готовых изображений.

Карточка — работа плюс короткая реплика. Набор описывается JSON-спекой, чтобы
тексты правились без лазанья в код, а пересборка была повторяемой.

    python carousel.py build спека.json -o папка/
    python carousel.py preview папка/ -o превью.png

Спека:
{
  "size": 1080,
  "root": "/путь/к/картинкам",
  "cards": [
    {"image": "acrylic/9.jpg", "text": "Акрил не прощает"},
    {"image": "room.jpg", "text": "...", "focus_x": 0.25},
    {"image": "oil/23.jpg", "text": "Две скорости.\\nОдин горизонт.", "cta": "shvoren.com"}
  ]
}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageStat

SERIF = os.environ.get("VS_CARD_SERIF", "/System/Library/Fonts/Supplemental/Georgia.ttf")
SANS = os.environ.get("VS_CARD_SANS", "/System/Library/Fonts/Supplemental/Arial.ttf")

# Меньше половины кадра под текст: у Meta есть правило про долю площади,
# занятой надписью, и плотный блок читается как баннер, а не как работа.
SCRIM_FRACTION = 0.46


def fit_square(im: Image.Image, size: int, focus: tuple[float, float] = (0.5, 0.5)) -> Image.Image:
    """Кадр в квадрат без искажения пропорций.

    `focus` — куда смотреть при обрезке, долями от 0 до 1. По центру годится
    для самой живописи, но на интерьерных снимках картина часто висит сбоку,
    и центральный кадр выбрасывает из карточки ровно то, ради чего она снята.
    """
    w, h = im.size
    side = min(w, h)
    fx, fy = focus
    left = int((w - side) * min(max(fx, 0.0), 1.0))
    top = int((h - side) * min(max(fy, 0.0), 1.0))
    im = im.crop((left, top, left + side, top + side))
    return im.resize((size, size), Image.LANCZOS)


def scrim(img: Image.Image, size: int) -> Image.Image:
    """Затемнение снизу, подобранное под яркость самой работы.

    Фиксированная плотность, которой хватает на тёмном холсте, полностью
    проваливается на светлом — белый текст тонет. Поэтому считаем яркость
    той области, куда ляжет надпись, и подбираем плотность под неё.
    """
    h = int(size * SCRIM_FRACTION)
    brightness = ImageStat.Stat(img.convert("L").crop((0, size - h, size, size))).mean[0]
    peak = int(min(238, max(170, 120 + brightness * 0.62)))

    grad = Image.new("L", (1, h))
    for y in range(h):
        grad.putpixel((0, y), int(peak * (y / h) ** 1.6))
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    layer.paste(Image.new("RGBA", (size, h), (16, 18, 22, 255)),
                (0, size - h), grad.resize((size, h)))
    return Image.alpha_composite(img.convert("RGBA"), layer)


def draw_text(img: Image.Image, text: str, cta: str | None, size: int) -> Image.Image:
    d = ImageDraw.Draw(img)
    lines = text.split("\n")
    font_size = int(size * 0.069) if max(len(l) for l in lines) < 20 else int(size * 0.057)
    font = ImageFont.truetype(SERIF, font_size)
    lh = int(font_size * 1.32)
    y = size - lh * len(lines) - int(size * (0.139 if cta else 0.089))
    for line in lines:
        w = d.textlength(line, font=font)
        x = (size - w) / 2
        # тень под буквой — страховка на светлых участках живописи
        d.text((x + 2, y + 2), line, font=font, fill=(10, 12, 16, 120))
        d.text((x, y), line, font=font, fill=(250, 249, 246, 255))
        y += lh
    if cta:
        f2 = ImageFont.truetype(SANS, int(size * 0.028))
        w = d.textlength(cta, font=f2)
        d.text(((size - w) / 2, size - int(size * 0.0963)), cta, font=f2,
               fill=(232, 230, 225, 240))
        half, base = round(size * 0.054), size - int(size * 0.115)
        d.line([(size / 2 - half, base), (size / 2 + half, base)],
               fill=(232, 230, 225, 150), width=1)
    return img


def cmd_build(args) -> None:
    spec = json.loads(Path(args.spec).read_text())
    size = int(spec.get("size", 1080))
    root = Path(spec.get("root", "."))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def resolve(name: str) -> Path:
        # Абсолютный путь берём как есть: карточки одного набора нередко
        # собираются из разных источников — работы, мокапы, композиты.
        p = Path(name)
        return p if p.is_absolute() else root / p

    missing = [c["image"] for c in spec["cards"] if not resolve(c["image"]).is_file()]
    if missing:
        raise SystemExit("не найдены изображения: " + ", ".join(missing))

    for i, card in enumerate(spec["cards"], 1):
        focus = (float(card.get("focus_x", 0.5)), float(card.get("focus_y", 0.5)))
        im = fit_square(Image.open(resolve(card["image"])).convert("RGB"), size, focus)
        im = draw_text(scrim(im, size), card["text"], card.get("cta"), size)
        path = out / f"card_{i}.jpg"
        im.convert("RGB").save(path, quality=94)
        print(f"  {i}. {card['image']:<18} «{card['text'].replace(chr(10), ' / ')}»")
    print(f"\n  {len(spec['cards'])} карточек {size}x{size} в {out}")


def cmd_preview(args) -> None:
    cards = sorted(Path(args.folder).glob("card_*.jpg"),
                   key=lambda p: int(p.stem.split("_")[1]))
    if not cards:
        raise SystemExit("в папке нет card_*.jpg")
    ims = [Image.open(p).convert("RGB") for p in cards]
    for im in ims:
        im.thumbnail((380, 380))
    w, h = ims[0].size
    cols = min(3, len(ims))
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (w * cols, h * rows), (22, 22, 22))
    for k, im in enumerate(ims):
        sheet.paste(im, ((k % cols) * w, (k // cols) * h))
    sheet.save(args.out)
    print(f"  превью {len(ims)} карточек: {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="собрать карточки по спеке")
    b.add_argument("spec"); b.add_argument("-o", "--out", required=True)
    b.set_defaults(fn=cmd_build)
    p = sub.add_parser("preview", help="контактный лист готовых карточек")
    p.add_argument("folder"); p.add_argument("-o", "--out", required=True)
    p.set_defaults(fn=cmd_preview)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
