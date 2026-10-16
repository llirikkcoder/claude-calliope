"""Сборка озвучки в дорожку и субтитры, синхронные по построению.

`voice_clone.py batch` озвучивает каждую реплику отдельным файлом и пишет их
длительности в манифест. Значит тайминги субтитров известны точно — распознавать
речь не нужно, достаточно сложить длительности.

    python voiceover.py build озвучка/voice-manifest.json -o ролик
    python voiceover.py build … --shotlist шотлист.md --root <корень>

На выходе: дорожка .m4a, субтитры .srt и .vtt с одинаковыми таймингами.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def ts_srt(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def ts_vtt(seconds: float) -> str:
    return ts_srt(seconds).replace(",", ".")


def wrap(text: str, limit: int = 42) -> str:
    """Перенос длинной реплики на две строки — больше в кадре не читается."""
    if len(text) <= limit:
        return text
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(f"{cur} {w}".strip()) <= limit or not cur:
            cur = f"{cur} {w}".strip()
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return "\n".join(lines[:2]) if len(lines) <= 2 else "\n".join([lines[0], " ".join(lines[1:])])


def shot_offsets(shotlist: str, roots: list[str]) -> dict[int, tuple[float, float]]:
    """Начало и длительность каждого шота по шот-листу."""
    sys.path.insert(0, str(Path(__file__).parent))
    from vs_bridge import parse_shotlist

    shots = parse_shotlist(Path(shotlist).read_text())
    offsets, t = {}, 0.0
    for s in shots:
        offsets[s["n"]] = (t, float(s["seconds"]))
        t += float(s["seconds"])
    return offsets


def build_track(items: list[dict], out: str, gap: float) -> float:
    """Склеить реплики в одну дорожку, вставляя тишину по таймингам."""
    tmp = tempfile.mkdtemp(prefix="voiceover-")
    parts, cursor = [], 0.0
    for it in items:
        start = it["start"]
        if start > cursor + 0.01:
            sil = f"{tmp}/sil_{len(parts)}.m4a"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                            "anullsrc=channel_layout=mono:sample_rate=44100",
                            "-t", f"{start - cursor:.3f}", "-c:a", "aac", "-y", sil], check=True)
            parts.append(sil)
            cursor = start
        conv = f"{tmp}/part_{len(parts)}.m4a"
        subprocess.run(["ffmpeg", "-v", "error", "-i", it["file"],
                        "-ac", "1", "-ar", "44100", "-c:a", "aac", "-y", conv], check=True)
        parts.append(conv)
        cursor += it["seconds"]
        if gap:
            cursor += gap

    listing = f"{tmp}/list.txt"
    Path(listing).write_text("".join(f"file '{p}'\n" for p in parts))
    subprocess.run(["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0",
                    "-i", listing, "-c", "copy", "-y", out], check=True)
    return cursor


def cmd_build(args) -> None:
    manifest = json.loads(Path(args.manifest).read_text())
    items = [dict(i) for i in manifest.get("items") or []]
    if not items:
        raise SystemExit("в манифесте нет озвученных реплик")

    if args.shotlist:
        # Реплики ложатся на тайминги шотов: id вида "ru-3" или "3" относим
        # к шоту с таким номером, чтобы озвучка совпала с картинкой.
        offsets = shot_offsets(args.shotlist, args.root or [])
        placed = []
        for it in items:
            m = re.search(r"(\d+)", str(it.get("id") or ""))
            n = int(m.group(1)) if m else None
            if n not in offsets:
                print(f"  реплика {it.get('id')}: нет шота с таким номером, пропущена")
                continue
            start, dur = offsets[n]
            if it["seconds"] > dur + 0.05:
                print(f"  реплика {it.get('id')}: {it['seconds']:.1f} с не влезает "
                      f"в шот на {dur:g} с — субтитр оборвётся вместе с кадром")
            placed.append({**it, "start": start})
        items = sorted(placed, key=lambda x: x["start"])
    else:
        t = 0.0
        for it in items:
            it["start"] = t
            t += it["seconds"] + args.gap

    out_base = Path(args.out)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    track = str(out_base.with_suffix(".m4a"))
    total = build_track(items, track, args.gap if not args.shotlist else 0.0)

    srt, vtt = [], ["WEBVTT", ""]
    for i, it in enumerate(items, 1):
        start, end = it["start"], it["start"] + it["seconds"]
        text = wrap(it["text"])
        srt += [str(i), f"{ts_srt(start)} --> {ts_srt(end)}", text, ""]
        vtt += [f"{ts_vtt(start)} --> {ts_vtt(end)}", text, ""]
    Path(out_base.with_suffix(".srt")).write_text("\n".join(srt), encoding="utf-8")
    Path(out_base.with_suffix(".vtt")).write_text("\n".join(vtt), encoding="utf-8")

    print(f"  дорожка:  {track} ({total:.1f} с)")
    print(f"  субтитры: {out_base.with_suffix('.srt')} и .vtt — {len(items)} реплик")
    print("  тайминги взяты из длительностей файлов, распознавание не участвовало")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="дорожка + субтитры из манифеста озвучки")
    b.add_argument("manifest")
    b.add_argument("-o", "--out", required=True, help="путь без расширения")
    b.add_argument("--gap", type=float, default=0.4, help="пауза между репликами, с")
    b.add_argument("--shotlist", help="разложить реплики по таймингам шотов")
    b.add_argument("--root", action="append")
    b.set_defaults(fn=cmd_build)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
