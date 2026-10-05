"""Рендер шотов на fal.ai вместо локального ComfyUI.

Тот же keyframe-переход, что делает наш MLX-граф, только на чужих GPU:
LTX-2.3 принимает начальный и конечный кадр и строит переход между ними.
Подписки нет — платишь за секунды.

    python fal_backend.py quote шотлист.md --root <корень>
    python fal_backend.py render шотлист.md --root <корень> --manifest m.json --yes

Без `--yes` ничего не запускается: сначала показывается цена.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL = "fal-ai/ltx-2.3/image-to-video/fast"
# Заявку принимает полный путь модели, а вот статус и результат спрашиваются
# по имени приложения — первые два сегмента. На полном пути сервис отвечает 405.
APP = "/".join(MODEL.split("/")[:2])
QUEUE = "https://queue.fal.run"

# Длительности, которые принимает модель. Короче шести секунд не умеет —
# четырёхсекундный шот всё равно считается и оплачивается как шестисекундный,
# а лишнее мы обрезаем при сборке.
ALLOWED_DURATIONS = (6, 8, 10, 12, 14, 16, 18, 20)
PRICE_PER_SECOND = {"1080p": 0.06, "1440p": 0.12, "2160p": 0.24}


def fal_key() -> str:
    key = os.environ.get("FAL_KEY")
    if key:
        return key
    # Ключ обычно живёт в настройках Claude Code, а не в окружении шелла.
    settings = Path.home() / ".claude" / "settings.json"
    if settings.is_file():
        key = (json.loads(settings.read_text()).get("env") or {}).get("FAL_KEY")
        if key:
            return key
    raise SystemExit("не найден FAL_KEY — ни в окружении, ни в ~/.claude/settings.json")


def billable_duration(seconds: float) -> int:
    for d in ALLOWED_DURATIONS:
        if seconds <= d:
            return d
    return ALLOWED_DURATIONS[-1]


def data_uri(path: str, max_side: int = 1920) -> str:
    """Кадр как data-URI: так не нужен отдельный аплоад в хранилище.

    Исходники у нас до 4000 px, а на выходе 1080p — большие кадры только
    раздували бы запрос.
    """
    from PIL import Image

    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _post(url: str, body: dict, key: str) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": f"Key {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.load(resp)


def _get(url: str, key: str) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": f"Key {key}"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)


def submit(shot: dict, key: str, resolution: str, fps: int) -> str:
    payload = {
        "prompt": shot["prompt"],
        "image_url": data_uri(shot["first"]),
        "duration": billable_duration(shot["seconds"]),
        "resolution": resolution,
        "fps": fps,
        "generate_audio": False,
    }
    if shot.get("last"):
        payload["end_image_url"] = data_uri(shot["last"])
    res = _post(f"{QUEUE}/{MODEL}", payload, key)
    return res["request_id"]


def wait(request_id: str, key: str, poll: float = 10.0, timeout: float = 1800) -> dict:
    started = time.time()
    status_url = f"{QUEUE}/{APP}/requests/{request_id}/status"
    while True:
        st = _get(status_url, key)
        state = st.get("status")
        if state == "COMPLETED":
            return _get(f"{QUEUE}/{APP}/requests/{request_id}", key)
        if state in ("FAILED", "CANCELLED"):
            raise RuntimeError(f"fal вернул {state}: {json.dumps(st, ensure_ascii=False)[:400]}")
        if time.time() - started > timeout:
            raise RuntimeError(f"ожидание дольше {timeout:.0f} с, запрос {request_id}")
        time.sleep(poll)


def collect_shots(shotlist: str, roots: list[str]) -> tuple[list[dict], list[dict]]:
    """Разобрать шот-лист и собрать задания на генерацию с ключевыми кадрами."""
    sys.path.insert(0, str(Path(__file__).parent))
    from vs_bridge import parse_shotlist, resolve_paths, neighbour_frames
    import tempfile

    shots = parse_shotlist(Path(shotlist).read_text())
    problems = resolve_paths(shots, roots)
    for p in problems:
        print(f"  ВНИМАНИЕ {p}")

    tmp = tempfile.mkdtemp(prefix="fal-frames-")
    jobs = []
    for idx, shot in enumerate(shots):
        if shot["kind"] != "generate":
            continue
        before, after = neighbour_frames(shots, idx, tmp)
        if not before:
            print(f"  шот {shot['n']}: нет реального кадра слева — "
                  f"модели нужен хотя бы начальный, шот пропущен")
            continue
        jobs.append({"n": shot["n"], "seconds": shot["seconds"], "prompt": shot["prompt"],
                     "first": before, "last": after})
    return shots, jobs


def quote(jobs: list[dict], resolution: str) -> float:
    rate = PRICE_PER_SECOND[resolution]
    total = 0.0
    for j in jobs:
        billed = billable_duration(j["seconds"])
        cost = billed * rate
        total += cost
        tail = "" if billed == j["seconds"] else f"  (оплата за {billed} с — короче модель не умеет)"
        print(f"   шот {j['n']}: {j['seconds']:g} с → ${cost:.2f}{tail}")
    print(f"\n  итого ${total:.2f} при {resolution}, переходов {len(jobs)}")
    return total


def cmd_quote(args) -> None:
    _, jobs = collect_shots(args.shotlist, args.root or [])
    if not jobs:
        raise SystemExit("в шот-листе нет шотов на генерацию")
    quote(jobs, args.resolution)
    print("  запустить: та же команда с `render` и `--yes`")


def cmd_render(args) -> None:
    shots, jobs = collect_shots(args.shotlist, args.root or [])
    if not jobs:
        raise SystemExit("в шот-листе нет шотов на генерацию")
    total = quote(jobs, args.resolution)
    if not args.yes:
        raise SystemExit("ничего не запущено: добавь --yes, если согласен на эту сумму")

    key = fal_key()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    results = {}
    for j in jobs:
        print(f"  шот {j['n']}: отправляю…", flush=True)
        rid = submit(j, key, args.resolution, args.fps)
        started = time.time()
        res = wait(rid, key)
        url = (res.get("video") or {}).get("url")
        if not url:
            print(f"  шот {j['n']}: ответ без видео — {json.dumps(res, ensure_ascii=False)[:200]}")
            continue
        path = out / f"shot_{j['n']:02d}.mp4"
        urllib.request.urlretrieve(url, path)
        results[j["n"]] = str(path)
        print(f"  шот {j['n']}: готов за {time.time()-started:.0f} с → {path}")

    # Манифест того же вида, что у локального пути: assemble не различает,
    # где именно считался шот.
    manifest = {"backend": "fal", "model": MODEL, "resolution": args.resolution,
                "shotlist": str(Path(args.shotlist).resolve()), "shots": []}
    for shot in shots:
        manifest["shots"].append({
            "n": shot["n"], "seconds": shot["seconds"], "kind": shot["kind"],
            "frame": shot["frame"], "line": shot["line"], "sound": shot["sound"],
            "path": shot["path"], "output": results.get(shot["n"]),
        })
    Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\n  манифест: {args.manifest}")
    print(f"  собрать: python vs_bridge.py assemble {args.manifest} -o ролик.mp4 --preset reels")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("quote", cmd_quote), ("render", cmd_render)):
        p = sub.add_parser(name, help="посчитать цену" if name == "quote" else "посчитать и отрендерить")
        p.add_argument("shotlist")
        p.add_argument("--root", action="append")
        p.add_argument("--resolution", default="1080p", choices=sorted(PRICE_PER_SECOND))
        p.add_argument("--fps", type=int, default=24, choices=(24, 25, 48, 50))
        if name == "render":
            p.add_argument("--out", default="fal-shots")
            p.add_argument("--manifest", required=True)
            p.add_argument("--yes", action="store_true", help="согласие на озвученную сумму")
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
