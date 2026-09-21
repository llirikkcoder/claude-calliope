"""Конвертер ComfyUI-графа из UI-формата в API-формат с ролевыми тегами Calliope.

Calliope читает роли из заголовков нод API-формата: (Input:prompt), (Output:video) и т.п.
Скрипт конвертирует сохранённый из интерфейса граф, подсказывает, куда какие теги
напрашиваются, и проставляет те, что заданы явно.

    python tag_workflow.py convert граф.json -o граф_API.json
    python tag_workflow.py inspect граф_API.json
    python tag_workflow.py tag граф_API.json --set 10=Input:prompt --set 8=Output:video

Порядок входных значений для виджетов берётся из /object_info живого ComfyUI,
поэтому он должен быть запущен.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request

# Роли, которые понимает Calliope (calliope/comfyui/roles.py).
INPUT_ROLES = (
    "prompt", "negative", "width", "height", "character", "location",
    "image", "video", "audio", "seed", "duration",
)
OUTPUT_ROLES = ("image", "video")

TAG_RE = re.compile(r"\((?P<kind>Input|Output)(?::(?P<role>[a-zA-Z0-9_-]+))?\)", re.IGNORECASE)

# Классы, чьё патчируемое поле Calliope знает сама — тег можно вешать прямо на них.
# Всё остальное патчится по эвристике из class_type и легко промахивается мимо
# настоящего имени виджета, поэтому точкой входа должна быть Primitive-нода.
SAFE_TAG_CLASSES = {
    "LoadImage", "ImageLoader", "ETN_LoadImageBase64", "LoadAudio", "VHS_LoadAudio",
    "LoadVideo", "VHS_LoadVideo", "VHS_LoadVideoPath", "CLIPTextEncode",
    "SaveVideo", "VHS_VideoCombine", "VideoOutput", "SaveImage", "PreviewImage",
}
PRIMITIVE_PREFIX = "Primitive"

# Виджет seed тянет за собой скрытый control_after_generate — в widgets_values
# он лежит лишним элементом сразу после значения.
_CONTROL_VALUES = {"fixed", "increment", "decrement", "randomize"}


def fetch_object_info(base_url: str, class_type: str) -> dict:
    url = f"{base_url.rstrip('/')}/object_info/{urllib.parse.quote(class_type)}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.load(resp).get(class_type) or {}


def ui_to_api(graph: dict, base_url: str) -> dict:
    """Собрать API-формат из UI-графа."""
    if "nodes" not in graph:
        raise SystemExit("это уже не UI-формат: нет ключа 'nodes'")

    # link_id -> (node_id источника, индекс выхода)
    links: dict[int, tuple[str, int]] = {}
    for link in graph.get("links") or []:
        if isinstance(link, list) and len(link) >= 5:
            links[link[0]] = (str(link[1]), link[2])

    api: dict[str, dict] = {}
    for node in graph["nodes"]:
        cls = node.get("type")
        if cls in ("Note", "MarkdownNote", "Reroute"):
            continue
        info = fetch_object_info(base_url, cls)
        spec = {**(info.get("input", {}).get("required") or {}),
                **(info.get("input", {}).get("optional") or {})}

        widgets = list(node.get("widgets_values") or [])
        wi = 0
        inputs: dict = {}
        for slot in node.get("inputs") or []:
            name = slot.get("name")
            # Подвходы dynamic-combo (вида "format.codec") в схеме ноды не значатся,
            # но виджет под них в UI есть — их надо и потребить, и записать, иначе
            # ComfyUI откажется исполнять ноду.
            dynamic_sub = name not in spec and "." in (name or "")
            if name not in spec and not dynamic_sub:
                # Кнопки вроде `upload` виджет всё же занимают: не потребить его
                # значит сбить порядок всех последующих.
                if slot.get("link") is None and wi < len(widgets):
                    wi += 1
                continue
            link_id = slot.get("link")
            if link_id is not None:
                src = links.get(link_id)
                if src:
                    inputs[name] = [src[0], src[1]]
                continue
            if wi < len(widgets):
                kind = spec.get(name, [None])[0] if not dynamic_sub else None
                # IMAGEUPLOAD и подобные — элементы интерфейса, а не входы ноды.
                if not (isinstance(kind, str) and kind.endswith("UPLOAD")):
                    inputs[name] = widgets[wi]
                wi += 1
                # проглотить control_after_generate, если он увязался следом
                if (wi < len(widgets) and isinstance(widgets[wi], str)
                        and widgets[wi] in _CONTROL_VALUES
                        and f"{name}_control_after_generate" not in spec):
                    wi += 1

        entry: dict = {"class_type": cls, "inputs": inputs}
        title = node.get("title")
        if title:
            entry["_meta"] = {"title": title}
        api[str(node["id"])] = entry
    return api


def suggest(api: dict) -> list[str]:
    """Подсказать, где Calliope не увидит слотов и куда просятся теги."""
    notes: list[str] = []
    tagged = {nid: TAG_RE.search((n.get("_meta") or {}).get("title") or "")
              for nid, n in api.items()}
    if not any(tagged.values()):
        notes.append("Ни одной размеченной ноды — Calliope не найдёт ни входов, ни выходов.")

    for nid, node in sorted(api.items(), key=lambda kv: int(kv[0])):
        cls = node.get("class_type", "")
        m = tagged.get(nid)
        if not m:
            continue
        safe = cls in SAFE_TAG_CLASSES or cls.startswith(PRIMITIVE_PREFIX)
        if not safe:
            notes.append(
                f"нода {nid} ({cls}): тег стоит на кастомной ноде. Calliope выводит имя "
                f"патчируемого поля из class_type и, скорее всего, промахнётся мимо "
                f"настоящего виджета — значение потеряется молча. Заведи Primitive-ноду "
                f"со ссылкой на нужный вход и повесь тег на неё."
            )
    if not any(m and m.group("kind").lower() == "output" for m in tagged.values()):
        notes.append("Нет ноды с тегом (Output:…) — Calliope не заберёт результат.")
    return notes


def describe(api: dict) -> None:
    print(f"  нод: {len(api)}")
    for nid, node in sorted(api.items(), key=lambda kv: int(kv[0])):
        title = (node.get("_meta") or {}).get("title") or ""
        m = TAG_RE.search(title)
        mark = f"  <-- {m.group(0)}" if m else ""
        print(f"   {nid:>4}  {node.get('class_type','?'):<34}{title[:34]:<36}{mark}")
    notes = suggest(api)
    if notes:
        print("\n  на что посмотреть:")
        for n in notes:
            print(f"   • {n}")


def apply_tags(api: dict, assignments: list[str]) -> dict:
    for item in assignments:
        if "=" not in item:
            raise SystemExit(f"ожидался вид node_id=Input:role, получено: {item}")
        nid, tag = item.split("=", 1)
        nid = nid.strip()
        if nid not in api:
            raise SystemExit(f"в графе нет ноды {nid}")
        if not TAG_RE.fullmatch(f"({tag.strip()})"):
            raise SystemExit(f"тег '{tag}' не распознан; ожидается Input:<роль> или Output:<роль>")
        meta = api[nid].setdefault("_meta", {})
        base = TAG_RE.sub("", meta.get("title") or "").strip()
        meta["title"] = f"({tag.strip()}) {base}".strip()
    return api


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("convert", help="UI-формат -> API-формат")
    c.add_argument("graph")
    c.add_argument("-o", "--out", required=True)
    c.add_argument("--comfy-url", default="http://127.0.0.1:8188")

    i = sub.add_parser("inspect", help="показать ноды и найденные теги")
    i.add_argument("graph")

    t = sub.add_parser("tag", help="проставить теги в API-графе")
    t.add_argument("graph")
    t.add_argument("--set", action="append", default=[], metavar="ID=Input:role")
    t.add_argument("-o", "--out")

    args = ap.parse_args()
    graph = json.load(open(args.graph))

    if args.cmd == "convert":
        api = ui_to_api(graph, args.comfy_url)
        json.dump(api, open(args.out, "w"), indent=2, ensure_ascii=False)
        print(f"  записан {args.out}")
        describe(api)
        print("\n  дальше: проверь граф через `comfy validate --workflow`, "
              "затем импортируй в Calliope (Settings -> Workflows).")
    elif args.cmd == "inspect":
        describe(graph)
    else:
        api = apply_tags(graph, args.set)
        out = args.out or args.graph
        json.dump(api, open(out, "w"), indent=2, ensure_ascii=False)
        print(f"  записан {out}")
        describe(api)


if __name__ == "__main__":
    sys.exit(main())
