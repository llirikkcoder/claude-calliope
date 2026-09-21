"""OpenAI-совместимый шим поверх Claude Code CLI.

Calliope (и любой другой клиент, умеющий OpenAI /v1/chat/completions) ходит сюда,
а мы транслируем запрос в `claude -p`. Работает на подписке Claude Code —
API-ключ не нужен.

Запуск:  .venv/bin/python shim.py --port 8317
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

log = logging.getLogger("claude-shim")

# Пустая рабочая директория: чтобы CLI не втягивал CLAUDE.md и файлы проекта.
# Переопределяется через CLAUDE_SHIM_WORKDIR.
WORKDIR = os.environ.get("CLAUDE_SHIM_WORKDIR") or os.path.join(
    tempfile.gettempdir(), "claude-shim-workdir"
)

# Инструменты не нужны — это чисто текстовая генерация. Отключение
# срезает их схемы из системного промпта.
_DISABLED_TOOLS = [
    "Bash", "Read", "Write", "Edit", "Glob", "Grep", "WebFetch", "WebSearch",
    "Task", "Agent", "TodoWrite", "NotebookEdit", "Artifact", "Skill", "ToolSearch",
]

DEFAULT_MODEL = "claude-opus-5"
# Один вызов CLI за раз: параллельные сессии только дерутся за лимиты.
_LOCK = asyncio.Semaphore(1)


def _split_messages(messages: list[dict[str, Any]]) -> tuple[str, str]:
    """Разложить OpenAI-сообщения на системный промпт и текст запроса."""
    system_parts: list[str] = []
    convo: list[str] = []
    for msg in messages or []:
        role = msg.get("role")
        content = msg.get("content")
        # Мультимодальные части схлопываем в текст — картинки сюда не идут.
        if isinstance(content, list):
            content = " ".join(
                str(p.get("text") or "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        text = (content or "").strip()
        if not text:
            continue
        if role == "system":
            system_parts.append(text)
        elif role == "assistant":
            convo.append(f"[Previous assistant reply]\n{text}")
        else:
            convo.append(text)
    return "\n\n".join(system_parts), "\n\n".join(convo)


async def _run_claude(system_prompt: str, prompt: str, model: str, timeout: float) -> str:
    cmd = [
        "claude", "-p", prompt,
        "--output-format", "json",
        "--model", model,
        # Срезаем обвес Claude Code: динамические секции, MCP-серверы, инструменты.
        "--exclude-dynamic-system-prompt-sections",
        "--strict-mcp-config",
        "--disallowed-tools", *_DISABLED_TOOLS,
    ]
    if system_prompt:
        cmd += ["--system-prompt", system_prompt]

    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=WORKDIR,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise RuntimeError(f"claude CLI не ответил за {timeout:.0f} с")

    if proc.returncode != 0:
        raise RuntimeError(f"claude CLI завершился с кодом {proc.returncode}: {err.decode()[:400]}")

    try:
        payload = json.loads(out.decode())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"CLI вернул не-JSON: {out.decode()[:300]}") from exc

    if payload.get("is_error"):
        raise RuntimeError(f"CLI сообщил об ошибке: {str(payload.get('result'))[:400]}")
    return payload.get("result") or ""


def _strip_fence(text: str) -> str:
    """Снять ```json…``` — модель часто оборачивает JSON в блок кода."""
    t = text.strip()
    if not t.startswith("```"):
        return t
    t = t.split("\n", 1)[1] if "\n" in t else ""
    if t.rstrip().endswith("```"):
        t = t.rstrip()[:-3]
    return t.strip()


def create_app(default_model: str, timeout: float) -> FastAPI:
    app = FastAPI(title="Claude Code OpenAI shim")

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [
            {"id": m, "object": "model", "owned_by": "anthropic"}
            for m in ("claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5-20251001")
        ]}

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        system_prompt, prompt = _split_messages(body.get("messages") or [])

        # response_format=json_object штатного аналога у CLI не имеет —
        # просим JSON словами и на выходе снимаем ограждение блока кода.
        wants_json = (body.get("response_format") or {}).get("type") == "json_object"
        if wants_json:
            system_prompt = (system_prompt + "\n\n" if system_prompt else "") + \
                "Reply with a single valid JSON value and nothing else. " \
                "No prose, no explanation, no markdown code fences."

        model = body.get("model") or default_model
        if not str(model).startswith("claude"):
            model = default_model

        started = time.time()
        try:
            async with _LOCK:
                text = await _run_claude(system_prompt, prompt, model, timeout)
        except Exception as exc:
            log.error("вызов не удался: %s", exc)
            return JSONResponse(status_code=502, content={"error": {"message": str(exc), "type": "shim_error"}})

        if wants_json:
            text = _strip_fence(text)
        log.info("ответ за %.1f с, %d символов (модель %s)", time.time() - started, len(text), model)

        created = int(time.time())
        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        # Токены честно посчитать нечем; клиенту эти поля не важны.
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        if body.get("stream"):
            async def event_stream():
                first = {"id": cid, "object": "chat.completion.chunk", "created": created,
                         "model": model,
                         "choices": [{"index": 0, "delta": {"role": "assistant", "content": text},
                                      "finish_reason": None}]}
                yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"
                last = {"id": cid, "object": "chat.completion.chunk", "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                yield f"data: {json.dumps(last, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(event_stream(), media_type="text/event-stream")

        return {
            "id": cid, "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": usage,
        }

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8317)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    os.makedirs(WORKDIR, exist_ok=True)

    import uvicorn
    uvicorn.run(create_app(args.model, args.timeout), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
