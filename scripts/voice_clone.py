"""Клонирование голоса по образцу и озвучка текста этим голосом.

Работает через MiniMax на fal.ai: образец (mp3/wav/ogg/m4a) превращается в
голос, которым потом можно читать любой текст. Подписки нет, ключ берётся
из `FAL_KEY`.

    python voice_clone.py clone образец.ogg --name viktoria
    python voice_clone.py say "Текст реплики" --voice viktoria -o out.mp3
    python voice_clone.py say --file сценарий.txt --voice viktoria -o out.mp3
    python voice_clone.py list

КЛОНИРОВАТЬ МОЖНО ТОЛЬКО СВОЙ ГОЛОС ИЛИ ГОЛОС ТОГО, КТО ДАЛ НА ЭТО СОГЛАСИЕ.
Синтез чужой речи без спроса — это подделка личности, а не инструмент.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fal_backend import fal_key, _get, _post  # noqa: E402

CLONE_MODEL = "fal-ai/minimax/voice-clone"
TTS_MODEL = "fal-ai/minimax/speech-02-hd"
QUEUE = "https://queue.fal.run"
STORAGE = "https://rest.alpha.fal.ai/storage/upload/initiate?storage_type=fal-cdn-v3"

# Голос, которым не пользовались неделю, сервис удаляет. Реестр помнит дату,
# чтобы можно было предупредить заранее, а не узнать по отказу.
VOICE_TTL_DAYS = 7
REGISTRY = Path(os.environ.get("VS_VOICE_REGISTRY",
                               Path.home() / ".config" / "claude-calliope" / "voices.json"))

MIN_SAMPLE_SECONDS = 10

# Языки, которые принимает language_boost. "auto" отдаёт определение сервису.
LANGUAGES = (
    "auto", "English", "Russian", "Ukrainian", "German", "French", "Spanish",
    "Portuguese", "Italian", "Dutch", "Polish", "Czech", "Slovak", "Slovenian",
    "Croatian", "Hungarian", "Romanian", "Bulgarian", "Greek", "Turkish",
    "Danish", "Swedish", "Norwegian", "Nynorsk", "Finnish", "Catalan",
    "Afrikaans", "Arabic", "Hebrew", "Hindi", "Chinese", "Japanese", "Korean",
    "Vietnamese", "Thai", "Indonesian", "Malay",
)


def detect_language(text: str) -> str:
    """Русский или английский по самому тексту.

    Нужно, чтобы не задавать язык руками на каждую реплику: у двуязычного
    проекта половина строк на одном языке, половина на другом. Всё, что не
    кириллица и не латиница, отдаём сервису через "auto".
    """
    cyr = sum(1 for ch in text if "\u0400" <= ch <= "\u04ff")
    lat = sum(1 for ch in text if ("a" <= ch.lower() <= "z"))
    if cyr == 0 and lat == 0:
        return "auto"
    return "Russian" if cyr >= lat else "English"


def load_registry() -> dict:
    if REGISTRY.is_file():
        return json.loads(REGISTRY.read_text())
    return {}


def save_registry(data: dict) -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def probe_duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True).stdout.strip()
    return float(out) if out else 0.0


def to_mp3(path: str) -> str:
    """Образец в mp3: сервис принимает его надёжнее, чем opus или m4a."""
    if path.lower().endswith(".mp3"):
        return path
    out = os.path.join(tempfile.mkdtemp(prefix="voice-"), "sample.mp3")
    subprocess.run(["ffmpeg", "-v", "error", "-i", path,
                    "-ac", "1", "-ar", "44100", "-b:a", "128k", "-y", out], check=True)
    return out


def upload(path: str, key: str) -> str:
    """Положить файл в хранилище fal и вернуть ссылку на него."""
    init = _post(STORAGE, {"content_type": "audio/mpeg", "file_name": Path(path).name}, key)
    put = urllib.request.Request(init["upload_url"], data=Path(path).read_bytes(),
                                 method="PUT", headers={"Content-Type": "audio/mpeg"})
    urllib.request.urlopen(put, timeout=300)
    return init["file_url"]


def run_queue(model: str, payload: dict, key: str, timeout: float = 900) -> dict:
    """Отправить заявку и дождаться результата."""
    app = "/".join(model.split("/")[:2])
    rid = _post(f"{QUEUE}/{model}", payload, key)["request_id"]
    started = time.time()
    while True:
        st = _get(f"{QUEUE}/{app}/requests/{rid}/status", key)
        state = st.get("status")
        if state == "COMPLETED":
            return _get(f"{QUEUE}/{app}/requests/{rid}", key)
        if state in ("FAILED", "CANCELLED"):
            raise RuntimeError(f"{state}: {json.dumps(st, ensure_ascii=False)[:300]}")
        if time.time() - started > timeout:
            raise RuntimeError(f"ожидание дольше {timeout:.0f} с")
        time.sleep(5)


def synthesize(text: str, voice_id: str, out: str, key: str, *, language: str = "auto",
               speed: float = 1.0, volume: float = 1.0, pitch: int = 0,
               emotion: str | None = None, sample_rate: int = 32000) -> str:
    payload = {
        "text": text,
        "voice_setting": {"voice_id": voice_id, "speed": speed, "vol": volume, "pitch": pitch},
        "audio_setting": {"sample_rate": sample_rate, "bitrate": 128000,
                          "format": "mp3", "channel": 1},
        "output_format": "url",
    }
    if emotion:
        payload["voice_setting"]["emotion"] = emotion
    if language and language != "auto":
        payload["language_boost"] = language
    if language == "English":
        # Иначе «80x100» и «2026» читаются посимвольно.
        payload["voice_setting"]["english_normalization"] = True

    res = run_queue(TTS_MODEL, payload, key)
    url = (res.get("audio") or {}).get("url")
    if not url:
        raise RuntimeError(f"ответ без звука: {json.dumps(res, ensure_ascii=False)[:300]}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, out)
    return out


def cmd_clone(args) -> None:
    sample = args.sample
    if not Path(sample).is_file():
        raise SystemExit(f"нет файла: {sample}")
    dur = probe_duration(sample)
    if dur < MIN_SAMPLE_SECONDS:
        raise SystemExit(f"образец {dur:.1f} с — модели нужно хотя бы {MIN_SAMPLE_SECONDS}")
    print(f"  образец: {dur:.1f} с")

    if not args.yes:
        print("\n  Клонировать можно только свой голос или голос того, кто согласился.")
        print("  Подтверди, что согласие есть: добавь --yes")
        raise SystemExit(1)

    key = fal_key()
    mp3 = to_mp3(sample)
    print("  загружаю образец…", flush=True)
    url = upload(mp3, key)

    payload = {"audio_url": url, "noise_reduction": args.noise_reduction,
               "need_volume_normalization": True, "model": args.model}
    if args.preview_text:
        payload["text"] = args.preview_text
    print("  клонирую…", flush=True)
    res = run_queue(CLONE_MODEL, payload, key)

    voice_id = res.get("custom_voice_id")
    if not voice_id:
        raise SystemExit(f"сервис не вернул голос: {json.dumps(res, ensure_ascii=False)[:300]}")

    reg = load_registry()
    reg[args.name] = {"voice_id": voice_id, "model": args.model,
                      "sample": str(Path(sample).resolve()),
                      "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      "last_used": None}
    save_registry(reg)
    print(f"\n  голос «{args.name}» создан: {voice_id}")
    print(f"  реестр: {REGISTRY}")

    preview = (res.get("audio") or {}).get("url")
    if preview and args.out:
        urllib.request.urlretrieve(preview, args.out)
        print(f"  проба голоса: {args.out}")
    print(f"\n  Голос удалится, если не использовать его {VOICE_TTL_DAYS} дней.")


def cmd_register(args) -> None:
    """Записать в реестр голос, созданный на сайте.

    Клонировать удобнее бывает через веб-интерфейс, а пользоваться — из
    терминала. Эта команда связывает одно с другим: id со страницы результата
    получает короткое имя, и дальше `say`/`batch` работают как обычно.
    """
    reg = load_registry()
    reg[args.name] = {"voice_id": args.voice_id, "model": args.model,
                      "sample": args.sample or "создан на сайте fal.ai",
                      "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      "last_used": None}
    save_registry(reg)
    print(f"  голос «{args.name}» записан: {args.voice_id}")
    print(f"  реестр: {REGISTRY}")
    print(f"  Голос удалится, если не использовать его {VOICE_TTL_DAYS} дней.")


def cmd_say(args) -> None:
    text = Path(args.file).read_text().strip() if args.file else (args.text or "")
    if not text:
        raise SystemExit("нечего озвучивать: дай текст или --file")
    if len(text) > 5000:
        raise SystemExit(f"текст {len(text)} символов — модель берёт не больше 5000")

    reg = load_registry()
    if args.voice not in reg:
        known = ", ".join(reg) or "ни одного"
        raise SystemExit(f"голос «{args.voice}» не известен. Есть: {known}")
    entry = reg[args.voice]

    key = fal_key()
    language = args.language or detect_language(text)
    started = time.time()
    synthesize(text, entry["voice_id"], args.out, key, language=language,
               speed=args.speed, volume=args.volume, pitch=args.pitch,
               emotion=args.emotion, sample_rate=args.sample_rate)

    entry["last_used"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save_registry(reg)
    dur = probe_duration(args.out)
    print(f"  {language}: {len(text)} символов → {args.out} "
          f"({dur:.1f} с звука, {time.time()-started:.0f} с работы)")


def cmd_batch(args) -> None:
    """Озвучить набор реплик — каждую своим файлом и своим языком.

    Формат: либо JSON-список записей {"id", "text", опционально "language",
    "emotion", "speed"}, либо текстовый файл по реплике на строку.
    """
    src = Path(args.script)
    if src.suffix.lower() == ".json":
        items = json.loads(src.read_text())
        if isinstance(items, dict):
            items = [{"id": k, "text": v} for k, v in items.items()]
    else:
        items = [{"id": f"{i:02d}", "text": line.strip()}
                 for i, line in enumerate(src.read_text().splitlines(), 1)
                 if line.strip() and not line.strip().startswith("#")]
    if not items:
        raise SystemExit("в файле нет реплик")

    plan = []
    for it in items:
        text = (it.get("text") or "").strip()
        if not text:
            continue
        plan.append({**it, "text": text,
                     "language": it.get("language") or args.language or detect_language(text)})

    chars = sum(len(p["text"]) for p in plan)
    print(f"  реплик: {len(plan)}, символов: {chars}")
    for p in plan:
        print(f"   {p['id']:<10} {p['language']:<8} {p['text'][:52]}")
    # Голос нужен только для запуска: разбор и раскладку по языкам полезно
    # посмотреть и до того, как голос вообще создан.
    reg = load_registry()
    if args.voice not in reg:
        print(f"\n  голос «{args.voice}» не известен. Есть: {', '.join(reg) or 'ни одного'}")
        print("  создать: voice_clone.py clone образец.mp3 --name " + args.voice + " --yes")
        return
    entry = reg[args.voice]

    if not args.yes:
        print("\n  ничего не озвучено: добавь --yes")
        return

    key = fal_key()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    done = []
    for p in plan:
        path = out_dir / f"{p['id']}.mp3"
        try:
            synthesize(p["text"], entry["voice_id"], str(path), key,
                       language=p["language"], speed=float(p.get("speed", args.speed)),
                       volume=args.volume, pitch=args.pitch,
                       emotion=p.get("emotion") or args.emotion,
                       sample_rate=args.sample_rate)
        except Exception as exc:
            print(f"   {p['id']}: не вышло — {exc}")
            continue
        done.append({**p, "file": str(path), "seconds": probe_duration(str(path))})
        print(f"   {p['id']:<10} {done[-1]['seconds']:.1f} с → {path}")

    entry["last_used"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save_registry(reg)
    manifest = out_dir / "voice-manifest.json"
    manifest.write_text(json.dumps({"voice": args.voice, "items": done},
                                   ensure_ascii=False, indent=2))
    total = sum(d["seconds"] for d in done)
    print(f"\n  готово {len(done)} из {len(plan)}, суммарно {total:.1f} с звука")
    print(f"  манифест: {manifest}")


def cmd_list(_args) -> None:
    reg = load_registry()
    if not reg:
        print(f"  голосов пока нет ({REGISTRY})")
        return
    now = datetime.now(timezone.utc)
    for name, e in reg.items():
        last = e.get("last_used") or e.get("created")
        age = now - datetime.fromisoformat(last)
        left = VOICE_TTL_DAYS - age.days
        mark = f"сгорит через {left} дн." if left > 2 else (
            f"ВОТ-ВОТ СГОРИТ ({left} дн.) — озвучь что-нибудь" if left >= 0 else "возможно, уже удалён")
        print(f"  {name:<16} {e['voice_id']:<28} {mark}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("clone", help="создать голос по образцу")
    c.add_argument("sample"); c.add_argument("--name", required=True)
    c.add_argument("--model", default="speech-02-hd",
                   choices=("speech-02-hd", "speech-02-turbo", "speech-01-hd", "speech-01-turbo"))
    c.add_argument("--noise-reduction", action="store_true", help="подавить шум в образце")
    c.add_argument("--preview-text", help="фраза для пробы голоса")
    c.add_argument("--out", help="куда сохранить пробу")
    c.add_argument("--yes", action="store_true", help="подтверждение согласия владельца голоса")
    c.set_defaults(fn=cmd_clone)

    r = sub.add_parser("register", help="добавить голос, созданный на сайте fal.ai")
    r.add_argument("voice_id", help="custom_voice_id со страницы результата")
    r.add_argument("--name", required=True)
    r.add_argument("--model", default="speech-02-hd",
                   choices=("speech-02-hd", "speech-02-turbo", "speech-01-hd", "speech-01-turbo"))
    r.add_argument("--sample", help="откуда образец, для памяти")
    r.set_defaults(fn=cmd_register)

    s = sub.add_parser("say", help="озвучить текст готовым голосом")
    s.add_argument("text", nargs="?"); s.add_argument("--file")
    s.add_argument("--voice", required=True); s.add_argument("-o", "--out", required=True)
    s.add_argument("--speed", type=float, default=1.0)
    s.add_argument("--volume", type=float, default=1.0)
    s.add_argument("--pitch", type=int, default=0)
    s.add_argument("--emotion", choices=("happy","sad","angry","fearful","disgusted","surprised","neutral"))
    s.add_argument("--language", default=None, choices=LANGUAGES,
                   help="по умолчанию определяется по тексту")
    s.add_argument("--sample-rate", type=int, default=32000,
                   choices=(8000,16000,22050,24000,32000,44100))
    s.set_defaults(fn=cmd_say)

    b = sub.add_parser("batch", help="озвучить набор реплик (двуязычный сценарий)")
    b.add_argument("script", help="JSON со списком реплик или текстовый файл построчно")
    b.add_argument("--voice", required=True)
    b.add_argument("--out", default="voice-out", help="папка для файлов")
    b.add_argument("--language", default=None, choices=LANGUAGES,
                   help="один язык на все реплики; по умолчанию — по тексту каждой")
    b.add_argument("--speed", type=float, default=1.0)
    b.add_argument("--volume", type=float, default=1.0)
    b.add_argument("--pitch", type=int, default=0)
    b.add_argument("--emotion", choices=("happy","sad","angry","fearful","disgusted","surprised","neutral"))
    b.add_argument("--sample-rate", type=int, default=32000,
                   choices=(8000,16000,22050,24000,32000,44100))
    b.add_argument("--yes", action="store_true", help="подтверждение: реплики уйдут на озвучку")
    b.set_defaults(fn=cmd_batch)

    sub.add_parser("list", help="какие голоса есть и когда сгорят").set_defaults(fn=cmd_list)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
