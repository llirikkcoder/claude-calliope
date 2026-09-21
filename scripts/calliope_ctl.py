"""Управление связкой Calliope + шим + ComfyUI и прогон шотов из командной строки.

    python calliope_ctl.py status
    python calliope_ctl.py workflows
    python calliope_ctl.py import-workflow граф_API.json --name "MLX Keyframe"
    python calliope_ctl.py shot --workflow 1 --prompt "…" --duration 4
    python calliope_ctl.py job 1 --wait

Адреса переопределяются через CALLIOPE_URL, COMFYUI_URL, CLAUDE_SHIM_URL.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request

CALLIOPE = os.environ.get("CALLIOPE_URL", "http://127.0.0.1:8247")
COMFYUI = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
SHIM = os.environ.get("CLAUDE_SHIM_URL", "http://127.0.0.1:8317")


def _req(url: str, body=None, method="GET", timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode()
    return json.loads(raw) if raw else {}


def _alive(url: str, timeout=5) -> bool:
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except urllib.error.HTTPError:
        return True  # ответил, пусть и кодом ошибки — значит живой
    except Exception:
        return False


def cmd_status(_args) -> None:
    checks = [("Calliope", f"{CALLIOPE}/api/health"),
              ("ComfyUI", f"{COMFYUI}/system_stats"),
              ("шим Claude", f"{SHIM}/health")]
    for name, url in checks:
        print(f"  {name:<14} {'работает' if _alive(url) else 'НЕ ОТВЕЧАЕТ':<12} {url}")
    try:
        s = _req(f"{CALLIOPE}/api/settings", timeout=20)
        print(f"\n  активная модель : {s.get('llm_model')}")
        roles = s.get("agent_llm_assignments") or {}
        profiles = {p["id"]: p.get("name") for p in s.get("llm_profiles") or []}
        print(f"  роли по профилям: "
              f"{ {r: profiles.get(pid, pid) for r, pid in roles.items()} or 'все на активной модели'}")
    except Exception as exc:
        print(f"  настройки прочитать не удалось: {exc}")


def cmd_workflows(_args) -> None:
    for w in _req(f"{CALLIOPE}/api/workflows", timeout=30):
        ins = w.get("input_schema") or []
        roles = ",".join(sorted({i.get("role") or "?" for i in ins}))
        print(f"  [{w['id']}] {w['name']:<40} {w.get('kind','?'):<7} слоты: {roles or '—'}")


def cmd_import(args) -> None:
    graph = json.load(open(args.graph))
    res = _req(f"{CALLIOPE}/api/workflows",
               {"name": args.name, "kind": args.kind, "workflow_json": graph,
                "description": args.description or "", "prompt_profile": "prose"},
               method="POST", timeout=120)
    print(f"  импортирован как id={res.get('id')}")
    for slot in res.get("input_schema") or []:
        print(f"   вход  {slot.get('nodeId'):>4}  {slot.get('role'):<10} {slot.get('label')}")
    for slot in res.get("output_schema") or []:
        print(f"   выход {slot.get('nodeId'):>4}  {slot.get('role'):<10} {slot.get('label')}")
    if not (res.get("input_schema") or res.get("output_schema")):
        print("   ни одного слота не распознано — проверь ролевые теги в заголовках нод")


def cmd_shot(args) -> None:
    project = _req(f"{CALLIOPE}/api/projects",
                   {"title": args.title, "idea": args.prompt,
                    "target_duration": f"{args.duration} секунд"},
                   method="POST", timeout=60)
    pid = project["id"]
    scene = _req(f"{CALLIOPE}/api/projects/{pid}/scenes",
                 {"order_index": 1, "heading": args.title, "action": args.prompt,
                  "duration_sec": args.duration, "workflow_id": args.workflow},
                 method="POST", timeout=60)
    clip = _req(f"{CALLIOPE}/api/projects/{pid}/scenes/{scene['id']}/clips",
                {"order_index": 1, "description": args.prompt,
                 "duration_sec": args.duration, "workflow_id": args.workflow},
                method="POST", timeout=60)
    res = _req(f"{CALLIOPE}/api/jobs/projects/{pid}/generate-videos",
               {"clip_ids": [clip["id"]], "workflow_id": args.workflow},
               method="POST", timeout=180)
    for job in res.get("jobs") or []:
        print(f"  задание {job['id']} поставлено в очередь (проект {pid})")
        print(f"  значения по ролям: "
              f"{json.dumps((job.get('payload') or {}).get('input_values'), ensure_ascii=False)}")
        print(f"  следить: python calliope_ctl.py job {job['id']} --wait")


def cmd_job(args) -> None:
    started = time.time()
    while True:
        job = _req(f"{CALLIOPE}/api/jobs/{args.job_id}", timeout=30)
        status = job.get("status")
        print(f"  [{time.time()-started:5.0f}с] {status} {job.get('error') or ''}")
        if status in ("done", "completed", "failed", "error") or not args.wait:
            for path in job.get("output_paths") or []:
                print(f"  готово: {path}")
            return
        time.sleep(args.interval)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="живы ли сервисы и куда назначены роли").set_defaults(fn=cmd_status)
    sub.add_parser("workflows", help="какие графы знает Calliope").set_defaults(fn=cmd_workflows)

    i = sub.add_parser("import-workflow", help="загрузить размеченный API-граф")
    i.add_argument("graph"); i.add_argument("--name", required=True)
    i.add_argument("--kind", default="video"); i.add_argument("--description")
    i.set_defaults(fn=cmd_import)

    s = sub.add_parser("shot", help="прогнать один шот выбранным графом")
    s.add_argument("--workflow", type=int, required=True)
    s.add_argument("--prompt", required=True)
    s.add_argument("--duration", type=int, default=4)
    s.add_argument("--title", default="Шот")
    s.set_defaults(fn=cmd_shot)

    j = sub.add_parser("job", help="статус задания")
    j.add_argument("job_id", type=int); j.add_argument("--wait", action="store_true")
    j.add_argument("--interval", type=float, default=20.0)
    j.set_defaults(fn=cmd_job)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
