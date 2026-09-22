"""Мост между шот-листом claude-vs-video и локальным продакшном Calliope.

Берёт готовый шот-лист (таблица из скилла claude-vs-video), переносит его в проект
Calliope и отправляет на локальный рендер только те шоты, что помечены
«сгенерировать». Реальные фото и видео остаются как есть — по правилу скилла
реальный кадр честнее и дешевле сгенерированного.

    python vs_bridge.py parse шотлист.md
    python vs_bridge.py ingest шотлист.md --workflow 1 --project "Ролик про Flow"
    python vs_bridge.py assemble <project_id> -o готовый.mp4

Шот с источником «сгенерировать» получает соседние реальные кадры как первый и
последний: именно так работает keyframe-граф LTX — он строит переход между двумя
изображениями. Если соседей нет, шот уходит с одним промптом.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

CALLIOPE = os.environ.get("CALLIOPE_URL", "http://127.0.0.1:8247")
COMFYUI = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")

# Форматы площадок из рекламной стратегии VS (ADS_Креативы.md).
PRESETS = {
    "reels":    (1080, 1920),   # Reels / Stories, 9:16
    "story":    (1080, 1920),
    "feed":     (1080, 1080),   # Meta feed, 1:1
    "square":   (1080, 1080),
    "portrait": (1080, 1350),   # 4:5
    "site":     (1920, 1080),   # горизонталь для сайта
    "linkedin": (1920, 1080),
}

# Титры рисуются Pillow и кладутся поверх через overlay: в сборках ffmpeg
# без libfreetype фильтра drawtext попросту нет.
CAPTION_FONT = os.environ.get("VS_CAPTION_FONT", "/System/Library/Fonts/Supplemental/Arial.ttf")

REAL_RE = re.compile(r"^\s*(фото|видео|photo|video)\s*:\s*(.+?)\s*$", re.IGNORECASE)
GEN_RE = re.compile(r"^\s*(сгенерировать|generate)\s*:\s*(.+?)\s*$", re.IGNORECASE)


# ─────────────────────────── разбор шот-листа ───────────────────────────

def _clean(cell: str) -> str:
    return cell.strip().strip("`").strip()


def parse_shotlist(text: str) -> list[dict]:
    """Вытащить шоты из markdown-таблицы формата claude-vs-video."""
    shots: list[dict] = []
    for line in text.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c for c in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        first = _clean(cells[0])
        # шапка и разделитель таблицы
        if not first.isdigit():
            continue
        try:
            seconds = float(_clean(cells[1]).replace(",", "."))
        except ValueError:
            seconds = 3.0
        source_raw = _clean(cells[3])
        shot = {
            "n": int(first),
            "seconds": seconds,
            "frame": _clean(cells[2]),
            "source_raw": source_raw,
            "line": _clean(cells[4]) if len(cells) > 4 else "",
            "sound": _clean(cells[5]) if len(cells) > 5 else "",
            "kind": "unknown",
            "path": None,
            "prompt": None,
        }
        m = REAL_RE.match(source_raw)
        if m:
            shot["kind"] = "video" if m.group(1).lower() in ("видео", "video") else "photo"
            shot["path"] = _clean(m.group(2))
        elif GEN_RE.match(source_raw):
            shot["kind"] = "generate"
            shot["prompt"] = _clean(GEN_RE.match(source_raw).group(2))
        shots.append(shot)
    return sorted(shots, key=lambda s: s["n"])


def resolve_paths(shots: list[dict], roots: list[str]) -> list[str]:
    """Превратить пути шот-листа в существующие файлы. Возвращает список проблем."""
    problems: list[str] = []
    for shot in shots:
        if shot["kind"] not in ("photo", "video"):
            continue
        raw = shot["path"]
        candidates = [Path(raw)] + [Path(r) / raw.lstrip("/") for r in roots]
        found = next((c for c in candidates if c.is_file()), None)
        if found:
            shot["path"] = str(found.resolve())
        else:
            problems.append(f"шот {shot['n']}: файл не найден — {raw}")
    return problems


# ─────────────────────────── работа с сервисами ───────────────────────────

def _req(url: str, body=None, method="GET", timeout=180):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode()
    return json.loads(raw) if raw else {}


def upload_to_comfy(path: str, subfolder: str = "vs") -> str:
    """Положить файл в input ComfyUI и вернуть имя, которое поймёт LoadImage."""
    data = Path(path).read_bytes()
    boundary = uuid.uuid4().hex
    ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
    parts = []
    for field, value in (("overwrite", "true"), ("subfolder", subfolder), ("type", "input")):
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"\r\n\r\n{value}\r\n".encode()
        )
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; "
        f"filename=\"{Path(path).name}\"\r\nContent-Type: {ctype}\r\n\r\n".encode()
        + data + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(
        f"{COMFYUI}/upload/image", data=b"".join(parts), method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.load(resp)
    sub = result.get("subfolder") or subfolder
    name = result.get("name", Path(path).name)
    return f"{sub}/{name}" if sub else name


def _still_from_video(src: str, out: str) -> str:
    """Кадр из середины видео — чтобы переход начинался с того, что реально было."""
    dur = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", src],
        capture_output=True, text=True).stdout.strip()
    at = (float(dur) / 2) if dur else 0.0
    subprocess.run(["ffmpeg", "-v", "error", "-ss", str(at), "-i", src,
                    "-frames:v", "1", "-y", out], check=True)
    return out


def neighbour_frames(shots: list[dict], index: int, tmpdir: str) -> tuple[str | None, str | None]:
    """Ближайший реальный кадр слева и справа от генерируемого шота."""
    def still_of(shot: dict) -> str | None:
        if shot["kind"] == "photo":
            return shot["path"]
        if shot["kind"] == "video":
            out = os.path.join(tmpdir, f"still_{shot['n']}.png")
            return _still_from_video(shot["path"], out) if not os.path.exists(out) else out
        return None

    before = next((still_of(s) for s in reversed(shots[:index]) if s["kind"] in ("photo", "video")), None)
    after = next((still_of(s) for s in shots[index + 1:] if s["kind"] in ("photo", "video")), None)
    return before, after


# ─────────────────────────── команды ───────────────────────────

def cmd_parse(args) -> None:
    shots = parse_shotlist(Path(args.shotlist).read_text())
    if not shots:
        raise SystemExit("в файле не нашлось таблицы шотов")
    problems = resolve_paths(shots, args.root or [])
    total = sum(s["seconds"] for s in shots)
    for s in shots:
        src = {"photo": "фото", "video": "видео", "generate": "СГЕНЕРИРОВАТЬ"}.get(s["kind"], "?")
        tail = s["path"] or (s["prompt"] or "")[:60]
        print(f"  {s['n']:>2}. {s['seconds']:>4}с  {src:<14} {tail}")
        line = s["line"].strip("«»\"")
        if line and line != "—":
            print(f"        реплика: «{line}»")
    gen = [s for s in shots if s["kind"] == "generate"]
    print(f"\n  итого {total:g}с, шотов {len(shots)}, на генерацию {len(gen)}")
    for p in problems:
        print(f"  ВНИМАНИЕ {p}")


def cmd_ingest(args) -> None:
    shots = parse_shotlist(Path(args.shotlist).read_text())
    problems = resolve_paths(shots, args.root or [])
    if problems and not args.force:
        for p in problems:
            print(f"  {p}")
        raise SystemExit("часть файлов не найдена — поправь пути или запусти с --force")

    total = sum(s["seconds"] for s in shots)
    project = _req(f"{CALLIOPE}/api/projects",
                   {"title": args.project, "idea": args.idea or args.project,
                    "target_duration": f"{total:g} секунд"}, method="POST")
    pid = project["id"]
    print(f"  проект {pid}: {args.project} ({total:g}с, {len(shots)} шотов)")

    tmpdir = tempfile.mkdtemp(prefix="vs-bridge-")
    manifest = {"project_id": pid, "shotlist": str(Path(args.shotlist).resolve()),
                "workflow_id": args.workflow, "shots": []}
    queued = 0

    for idx, shot in enumerate(shots):
        scene = _req(f"{CALLIOPE}/api/projects/{pid}/scenes",
                     {"order_index": shot["n"], "heading": f"Шот {shot['n']}",
                      "action": shot["frame"], "duration_sec": int(round(shot["seconds"])),
                      "workflow_id": args.workflow if shot["kind"] == "generate" else None},
                     method="POST")
        entry = {"n": shot["n"], "seconds": shot["seconds"], "kind": shot["kind"],
                 "frame": shot["frame"], "line": shot["line"], "sound": shot["sound"],
                 "path": shot["path"], "scene_id": scene["id"], "job_id": None,
                 "output": None}

        if shot["kind"] != "generate":
            manifest["shots"].append(entry)
            continue

        clip = _req(f"{CALLIOPE}/api/projects/{pid}/scenes/{scene['id']}/clips",
                    {"order_index": 1, "description": shot["prompt"],
                     "duration_sec": int(round(shot["seconds"])),
                     "workflow_id": args.workflow}, method="POST")

        # Переход строится между соседними реальными кадрами.
        before, after = neighbour_frames(shots, idx, tmpdir)
        input_values: dict[str, str] = {}
        if before and after:
            input_values[str(args.first_node)] = upload_to_comfy(before)
            input_values[str(args.last_node)] = upload_to_comfy(after)
            entry["keyframes"] = [before, after]
        elif before or after:
            only = before or after
            input_values[str(args.first_node)] = upload_to_comfy(only)
            input_values[str(args.last_node)] = upload_to_comfy(only)
            entry["keyframes"] = [only]
            print(f"   шот {shot['n']}: реальный сосед только с одной стороны — "
                  f"переход выйдет вялым, стоит задать второй кадр вручную")
        else:
            print(f"   шот {shot['n']}: реальных соседей нет, идёт только по промпту")

        res = _req(f"{CALLIOPE}/api/jobs/projects/{pid}/generate-videos",
                   {"clip_ids": [clip["id"]], "workflow_id": args.workflow,
                    "input_values": input_values or None}, method="POST")
        for job in res.get("jobs") or []:
            entry["job_id"] = job["id"]
            queued += 1
            print(f"   шот {shot['n']}: задание {job['id']} в очереди")
        manifest["shots"].append(entry)

    out = Path(args.manifest or f"vs-project-{pid}.json")
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\n  в очередь поставлено заданий: {queued}")
    print(f"  манифест: {out}")
    print(f"  следить: python vs_bridge.py wait {out}")


def cmd_wait(args) -> None:
    manifest = json.loads(Path(args.manifest).read_text())
    pending = [s for s in manifest["shots"] if s.get("job_id")]
    started = time.time()
    while True:
        left = []
        for shot in pending:
            job = _req(f"{CALLIOPE}/api/jobs/{shot['job_id']}", timeout=60)
            status = job.get("status")
            if status in ("done", "completed"):
                outs = job.get("output_paths") or []
                shot["output"] = outs[0] if outs else None
            elif status in ("failed", "error"):
                shot["output"] = None
                print(f"  шот {shot['n']}: задание провалено — {job.get('error')}")
            else:
                left.append(shot)
        Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        done = len(pending) - len(left)
        print(f"  [{time.time()-started:5.0f}с] готово {done}/{len(pending)}")
        if not left:
            return
        pending = left
        time.sleep(args.interval)


def render_caption(text: str, width: int, height: int, out: str) -> str | None:
    """Нарисовать титр в PNG с альфой — ffmpeg положит его поверх кадра."""
    text = (text or "").strip().strip("«»\"")
    if not text or text == "—":
        return None
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None

    size = max(28, int(height * 0.045))
    try:
        font = ImageFont.truetype(CAPTION_FONT, size)
    except OSError:
        font = ImageFont.load_default()

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Перенос по ширине кадра с запасом на поля.
    limit = int(width * 0.82)
    words, lines, current = text.split(), [], ""
    for word in words:
        probe = f"{current} {word}".strip()
        if draw.textlength(probe, font=font) <= limit or not current:
            current = probe
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)

    line_h = int(size * 1.35)
    total = line_h * len(lines)
    y = height - total - int(height * 0.08)
    for line in lines:
        w = draw.textlength(line, font=font)
        x = (width - w) / 2
        # Подложка-обводка: титр обязан читаться и на светлой живописи.
        for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            draw.text((x + dx, y + dy), line, font=font, fill=(0, 0, 0, 190))
        draw.text((x, y), line, font=font, fill=(255, 255, 255, 255))
        y += line_h
    img.save(out)
    return out


def cmd_assemble(args) -> None:
    """Склеить шоты по порядку: фото — как статичный кадр, видео — обрезкой."""
    manifest = json.loads(Path(args.manifest).read_text())
    width, height = PRESETS.get(args.preset, (args.width, args.height)) if args.preset \
        else (args.width, args.height)
    tmpdir = tempfile.mkdtemp(prefix="vs-assemble-")
    pieces: list[str] = []
    missing: list[int] = []

    for shot in sorted(manifest["shots"], key=lambda s: s["n"]):
        dur = float(shot["seconds"])
        out = os.path.join(tmpdir, f"{shot['n']:03d}.mp4")
        if args.fill == "blur":
            # Горизонтальная живопись в вертикали 9:16 оставляет половину кадра
            # чёрной. Размытая копия того же кадра держит внимание и палитру.
            scale = (f"split[bg][fg];"
                     f"[bg]scale={width}:{height}:force_original_aspect_ratio=increase,"
                     f"crop={width}:{height},gblur=sigma=28[bgb];"
                     f"[fg]scale={width}:{height}:force_original_aspect_ratio=decrease[fgs];"
                     f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,setsar=1")
        else:
            scale = (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                     f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1")
        if shot["kind"] == "generate":
            src = shot.get("output")
            if not src or not Path(src).is_file():
                missing.append(shot["n"]); continue
            cmd = ["ffmpeg", "-v", "error", "-i", src]
        elif shot["kind"] == "photo":
            cmd = ["ffmpeg", "-v", "error", "-loop", "1", "-i", shot["path"]]
        else:
            cmd = ["ffmpeg", "-v", "error", "-i", shot["path"]]

        caption = None
        if not args.no_captions:
            caption = render_caption(shot.get("line", ""), width, height,
                                     os.path.join(tmpdir, f"cap_{shot['n']}.png"))
        if caption:
            cmd += ["-i", caption,
                    "-filter_complex", f"[0:v]{scale},fps={args.fps}[bg];[bg][1:v]overlay=0:0"]
        else:
            cmd += ["-vf", f"{scale},fps={args.fps}"]

        subprocess.run(cmd + ["-t", str(dur), "-an", "-c:v", "libx264",
                               "-pix_fmt", "yuv420p", "-y", out], check=True)
        pieces.append(out)

    if missing:
        print(f"  пропущены несгенерированные шоты: {missing}")
    if not pieces:
        raise SystemExit("нечего склеивать")

    listing = os.path.join(tmpdir, "list.txt")
    Path(listing).write_text("".join(f"file '{p}'\n" for p in pieces))
    subprocess.run(["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0",
                    "-i", listing, "-c", "copy", "-y", args.out], check=True)
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", args.out],
                         capture_output=True, text=True).stdout.strip()
    print(f"  собрано: {args.out} ({float(dur):.1f}с, {width}x{height}, шотов {len(pieces)})")
    if float(dur) > 3.5 and not args.no_captions:
        first = next((s for s in sorted(manifest["shots"], key=lambda x: x["n"])), None)
        if first and not (first.get("line") or "").strip("«»\" —"):
            print("  первые 3 секунды без титра — по рекламной структуре это место хука")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("parse", help="разобрать шот-лист и проверить пути")
    p.add_argument("shotlist"); p.add_argument("--root", action="append")
    p.set_defaults(fn=cmd_parse)

    i = sub.add_parser("ingest", help="перенести шот-лист в Calliope и поставить генерацию")
    i.add_argument("shotlist")
    i.add_argument("--workflow", type=int, required=True)
    i.add_argument("--project", required=True)
    i.add_argument("--idea")
    i.add_argument("--root", action="append", help="куда подставлять относительные пути")
    i.add_argument("--manifest")
    i.add_argument("--first-node", type=int, default=4, help="нода (Input:image) первого кадра")
    i.add_argument("--last-node", type=int, default=5, help="нода (Input:image) последнего кадра")
    i.add_argument("--force", action="store_true")
    i.set_defaults(fn=cmd_ingest)

    w = sub.add_parser("wait", help="дождаться рендера всех шотов")
    w.add_argument("manifest"); w.add_argument("--interval", type=float, default=30.0)
    w.set_defaults(fn=cmd_wait)

    a = sub.add_parser("assemble", help="склеить готовый ролик")
    a.add_argument("manifest"); a.add_argument("-o", "--out", required=True)
    a.add_argument("--preset", choices=sorted(PRESETS), help="формат площадки (перекрывает --width/--height)")
    a.add_argument("--width", type=int, default=1080); a.add_argument("--height", type=int, default=1920)
    a.add_argument("--fps", type=int, default=24)
    a.add_argument("--no-captions", action="store_true", help="не накладывать титры из шот-листа")
    a.add_argument("--fill", choices=("blur", "black"), default="blur",
                   help="чем заполнять поля, когда кадр не совпал с форматом")
    a.set_defaults(fn=cmd_assemble)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
