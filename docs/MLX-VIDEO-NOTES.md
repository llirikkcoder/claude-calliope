# Заметки по локальной видеогенерации на Apple Silicon

Практические находки, которых нет в README проектов и в официальной документации.
Железо: **MacBook Pro M2 Max, 32 GB unified memory**, macOS 15.7.4, ComfyUI 0.36.0.
Дата: сентябрь 2026.

---

## 1. PyTorch MPS — что на нём не работает

### FP8 не поддерживается вообще
```
TypeError: Trying to convert Float8_e4m3fn to the MPS backend
           but it does not have support for that dtype
```
Любые `*_fp8_scaled.safetensors` на Mac бесполезны. Нужны bf16 или GGUF.
Это убивает большинство «оптимизированных» весов, которые выкладывают под NVIDIA.

### Тихая порча картинки на видеомоделях
ComfyUI issue **#15793**: Wan 2.1/2.2 на MPS даёт прогрессирующую порчу цвета.
В issue протестированы M4 Pro (ломается с 3 латентных кадров) и M3 Pro (чист),
**M1/M2 не тестировались**.

Проверено здесь: **M2 Max ломается**. 97 кадров 512×320 → цветовой шум,
лиса неузнаваема. Важно: видео при этом создаётся без единой ошибки,
метаданные корректны. Проверять надо ГЛАЗАМИ, разложив на кадры.

**Вывод: для видео на Apple Silicon брать MLX, а не PyTorch MPS.**

---

## 2. LTX-2.3 через MLX (`ltx-2-mlx`)

### LTX2_GEMMA_MAX_LENGTH — жёсткий потолок 128
Самый неочевидный параметр. Симптом:
```
RuntimeError: [METAL] Command buffer execution failed: Internal Error (0000000e)
```
Падает в `feature_extractor.py` на `_materialize(audio_embeds)` — audio-проекция
коннектора Gemma (конкатенация 49 слоёв, 188160 измерений).

- **128 — работает**
- **256 — падает даже при 18 GB свободной памяти**
- **512 — падает всегда**
- **значение должно быть кратно 128**; на 160 другая ошибка:
  `ValueError: [broadcast_shapes] Shapes (160,1) and (128,4096) cannot be broadcast`

Это НЕ нехватка памяти. Я дважды ошибочно гонял освобождать RAM — не помогает.

**Следствие:** промпт ограничен ~125 токенами. Официальные примеры (627 токенов)
не влезают. Обрезка идёт СЛЕВА (`tokens[-max_length:]`), то есть движок выкинет
начало промпта — в IC-LoRA это описание панелей. Сжимать промпт надо вручную.

### В версии 0.14.4 переменной нет — нужен патч
`ltx-core-mlx` 0.14.4 (ставится для ComfyUI) хардкодит `max_length: int = 1024`.
Патчить надо ДВЕ функции в
`.venv/.../ltx_core_mlx/text_encoders/gemma/encoders/base_encoder.py`:
- `tokenize(self, text, max_length=1024)`
- `encode_all_layers(self, text, max_length=1024)` ← её легко пропустить,
  нода вызывает именно её, и она перекрывает патч в `tokenize`

Патч слетает при обновлении пакета. Бэкап: `base_encoder.py.bak`.

### Модели: что брать на 32 GB
| вариант | DiT | RAM | вердикт |
|---|---|---|---|
| LTX-2.5 ditq8 | 20.6 GB | ~25 GB + энкодер 23.8 GB | **не влезает** |
| **LTX-2.3-q4** | 11.3 GB | умещается | **рабочий** |

LTX-2.5 рассчитан на 64 GB — это написано в карточке модели (int8 занимает 57%
на 64 GB). Энкодер у 2.5 — `gemma4-12b-ltx-v1` (23.8 GB) внутри репозитория,
у 2.3 — отдельный `mlx-community/gemma-3-12b-it-4bit` (8.1 GB).

Качать выборочно: `transformer-distilled` + `connector` + `vae_*` ≈ 19 GB
вместо всего репозитория (59.7 GB).

### keyframe требует dev-трансформер
`ltx-2-mlx keyframe` падает с distilled:
> Keyframe interpolation requires the dev (non-distilled) model.
> The distilled model hallucinates unrelated content during interpolation.

Нужны `transformer-dev.safetensors` + `ltx-2.3-22b-distilled-lora-384.safetensors`
и `--cfg-scale 3.0`. Это +19 GB к выборочной загрузке.

### Замеры (M2 Max, 512×320, 97 кадров)
| задача | шагов | время |
|---|---|---|
| keyframe (два кадра) | 20+3 | **9 мин 11 с** |
| ic-lora Ingredients | 8+3 | **8 мин 18 с** |
| a2v (говорящая голова) | 30 | **25–50 мин** |

`--low-ram` мапит трансформер через mmap: 11.3 GB грузятся за 2.3 с.

---

## 3. IC-LoRA Ingredients — как оно на самом деле работает

### `reference_strength` — не то, что кажется
По коду `VideoConditionByReferenceLatent.apply()`:
```python
mask_value = 1.0 - strength
new_latent = concatenate([target, reference])
```
Параметр задаёт, насколько референс **защищён от денойза**, а НЕ его вес
в композиции. Снижать бесполезно — я пробовал 0.15 и 0.2, эффекта нет.

### Число кадров референса важнее силы
`RepeatImageBatch` при 97 кадрах даёт референсу столько же токенов, сколько
у целевого видео → модель копирует лист покадрово. **Ставить 9.**

### Формат промпта решает
Работает только структура с привязкой панелей к местам:
```
### Reference Sheet Description
Top-left: ... Top-right: ... Bottom: ...
### Target Description
<герой> from top-left sits ... in the studio from the bottom panel ...
```
Ссылки «из верхней панели», «из нижней панели» — то, что заставляет
собрать сцену. Без них модель анимирует сам лист.

### Лист должен иметь МНОГО панелей
С двумя панелями модель принимает раскладку за образец кадра. Официальный
пример — шесть панелей (персонаж в трёх ракурсах, реквизит, одежда, локация).

### Главное ограничение MLX-обёртки
Даже с правильным листом и промптом **выход часто повторяет раскладку листа**.
Вероятная причина: `reference_downscale_factor = 1` в метаданных LoRA —
позиции референс-токенов ложатся ровно на целевые. В нативном ComfyUI иначе.

**Практический вывод: для «поместить героя в сцену» надёжнее композитинг:**
вырезать фигуру (`rembg`, модель `u2net_human_seg`) → вставить на фон →
подать как первый кадр в `LTXVMLXBaseSampler`. Модель сама уберёт следы
вклейки, посадит за стол и оживит. В тесте она даже убрала телефон из руки
(он был в негативном промпте).

---

## 4. MiniMax-H3 — не для 32 GB

33B DiT + 32B энкодер. Даже в GGUF (Q3_K_M 15.6 + Q2_K 8.5 = 24 GB весов):
- **1 ч 50 мин** на 5 секунд 512×320
- результат — цветовой шум, лиса угадывается лишь местами
- своп 27 GB, 703 с на шаг

Нативная поддержка в ComfyUI 0.36 есть (`MiniMaxH3ImageToVideo` с `first_frame`
и `last_frame`), GGUF читается, тип CLIP `minimax` существует. Технически
запускается — практически бесполезно.

---

## 5. ComfyUI: мелочи, стоившие времени

### Gated-репозитории HuggingFace
Krea2 (`krea/Krea-2-Turbo`) и IC-LoRA Lightricks закрыты (`gated: auto`).
Метаданные видны всем, файлы — нет. Нода Krea2-MLX качает голым `requests`
**без Authorization**, поэтому даже после `hf auth login` будет 401.
Обход: скачать вручную с токеном и указать `KREA2_BASE_DIR` — тогда
`_base_dir()` не вызывается вообще (`nodes.py:72`).

### HF режет скорость без токена
0.2 MB/s без авторизации → 7 MB/s с `HF_TOKEN`. Разница в 35 раз,
и это не про канал.

### Xet виснет и пишет кеш на системный диск
`HF_HUB_DISABLE_XET=1` — когда системный том заполнен или загрузка встала.

### Фоновые задачи гибнут вместе с сессией
Многочасовые загрузки запускать через `nohup ... & disown`,
иначе перезапуск сессии убивает их (у меня так вышло четыре раза).

### widgets_values в workflow-JSON
Это позиционный массив. У каждого поля `seed` ComfyUI добавляет скрытый
виджет `control_after_generate` — если его нет в массиве, всё после seed
съезжает, и получаешь `"transformer-dev.safetensors" couldn't be converted to INT`.

### Расширения ломаются целиком из-за одного импорта
`ComfyUI-LTXVideo-MLX-Extensions` падал на
`ImportError: cannot import name 'interleaved_freqs_cis'` — и не грузились
все 15 MLX-нод, хотя они от сломанного модуля не зависят.
Лечится заменой `__init__.py` на импорт только нужного подпакета.

### Пакеты с устаревшими метаданными не ставятся
`ltx-2-mlx` (xocialize): `License :: OSI Approved :: MIT License` в classifiers
ломает сборку на новом setuptools (PEP 639). `dgrauet/ltx-2-mlx`:
`readme = "../../README.md"` — путь вне директории пакета.
Лечится правкой `pyproject.toml` в локальном клоне.

---

## 6. kdenlive на macOS

### MCP-сервер требует ФОРК, а не обычный kdenlive
`mcp-kdenlive` работает только с `D-Ogi/kdenlive` — форком с D-Bus scripting API.
В официальной сборке методов `scriptGetProjectFps` и прочих просто нет,
поэтому `org.kde.kdenlive` на шине не появляется. Это написано в README сервера.
**Ставить dbus бессмысленно** — я потратил на это час и уронил приложение.

### Установка dbus через brew ломает kdenlive
`KF6Solid` находит системный `libdbus` и падает. Первый креш — через минуту
после `brew install dbus`. Лечится `brew uninstall dbus`.
У kdenlive свой D-Bus внутри бандла (`Contents/Frameworks/QtDBus.framework`),
и его конфиг требует launchd-сокет: `<listen>launchd:env=DBUS_LAUNCHD_SESSION_BUS_SOCKET</listen>`.

### exFAT-том через fskit роняет kdenlive
Симптом: `EXC_BREAKPOINT` в `Solid::Device::parent()` при работе файлового
диалога. Причина — том `Untitled` (exFAT, драйвер `fskit` из macOS 15).
Solid обходит устройства и падает на нём. **Отключить том — падения прекращаются.**
17 крешей подряд были именно из-за этого.

### Управление через AppleScript работает
Права Accessibility у `osascript` есть, меню доступно:
```applescript
tell application "System Events" to tell process "kdenlive"
  click menu item "Revert" of menu 1 of menu bar item "File" of menu bar 1
end tell
```
Так можно: Revert, Save, Insert Track, подтверждать диалоги
(кнопки лежат внутри `group`, искать `button "OK" of g`).

---

## 7. Формат .kdenlive (MLT XML) — подводные камни

### Что рендерить
Корневой атрибут `<mlt producer="...">`:
- kdenlive пишет `producer="main_bin"` — это КОРЗИНА, не таймлайн
- `melt` с таким атрибутом отрендерит клипы из корзины подряд
- для рендера таймлайна подменять на главный tractor

### Структура дорожки
Каждая дорожка — **tractor из ПАРЫ плейлистов** (вторая нужна для операций
вставки). Таймлайн обёрнут в секвенцию с UUID. Одиночный плейлист на дорожку
вызывает `Timeline clip without bin reference found and removed`.

### Каждый producer нуждается в kdenlive:id
Иначе: `Project corrupted. Clip black_track () with invalid Id.`
Плейлист чёрной дорожки должен называться `black_track`, а producer —
иначе (например `black`), иначе дубликат id ломает melt.

### Самозакрывающиеся теги ломают regex-правки
Пустые плейлисты пишутся как `<playlist id="playlist5"/>`. Регулярка
`<playlist id="X">(.*?)</playlist>` захватит содержимое СЛЕДУЮЩЕГО плейлиста.
**Править только XML-парсером** (`xml.etree.ElementTree`).
Из-за этого аудиоклип у меня дважды попадал в видеодорожку.

### Дорожки должен создавать сам редактор
Попытка добавить tractor'ы правкой XML → `EXC_BAD_ACCESS` при открытии.
Melt при этом рендерит нормально — редактор строже.
Порядок: `Sequence → Tracks → Insert Track…` в kdenlive, потом правка содержимого.

### Аудио: avformat, НЕ avformat-novalidate
Для wav-клипа `mlt_service` должен быть `avformat`.
С `avformat-novalidate` (как у видеоклипов) — **тишина, −91 dB**, без ошибок.
Это стоило мне тринадцати попыток.

### Аудио требует transition mix в главном тракторе
```xml
<transition id="mix_audio">
  <property name="a_track">0</property>
  <property name="b_track">1</property>
  <property name="mlt_service">mix</property>
  <property name="always_active">1</property>
  <property name="sum">1</property>
</transition>
```
Без него звук с дорожки не поднимается в выход.

### Проще собрать проект заново, чем чинить
Добавление аудиодорожки в уже построенный kdenlive-проект у меня не заработало
ни одним способом (13 подходов). Сборка нового проекта с нуля по проверенной
схеме — с первого раза.

---

## 8. Рабочие рецепты

### Видео между двумя кадрами
```bash
~/ComfyUI-Shared/scripts/ltx-keyframe.sh start.png end.png "промпт"
```
97 кадров = 4 с, 121 = 5 с, **193 = 8 с требуют 64 GB RAM** (данные автора).

### Говорящая голова
```bash
ltx-2-mlx a2v --image портрет.png --audio голос.wav --prompt "..." --frames 97
```
Аудио должно совпадать по длине с видео. 4 секунды — максимум за проход
на 32 GB; минута речи ≈ 5–6 часов счёта.

### Сборка сегментов
```bash
~/ComfyUI-Shared/scripts/assemble-video.sh <папка> [выход.mp4] [переход, с]
```
Нумеровать файлы `01_`, `02_`. Переход `0` — склейка без перекодирования.

### Рендер проекта kdenlive из терминала
```bash
/Applications/kdenlive.app/Contents/MacOS/melt проект.kdenlive \
  -consumer avformat:out.mp4 vcodec=libx264 acodec=aac pix_fmt=yuv420p
```
В бандле лежит melt 7.37.0 — отдельно ставить не нужно.

---

## 9. Диагностика: чему нельзя верить

**Метаданным видео.** Wan и MiniMax отдавали корректные `nb_frames`,
`duration`, кодеки — при полностью испорченной картинке. Всегда раскладывать
на кадры и смотреть:
```bash
ffmpeg -i out.mp4 -vf "select=eq(n\,50)" -frames:v 1 chk.png
```

**Размеру папки при загрузке.** Xet сбрасывает на диск рывками; `du` в паузе
показывает ноль прироста. Мерить ростом самого `.incomplete`-файла.

**`pgrep -f kdenlive`.** Находит процессы MCP-сервера, у которых слово
в пути. Проверять `ps aux | grep "[k]denlive.app"`.

**Отсутствию ошибок в рендере.** melt молча выдаёт чёрное видео или тишину,
если структура XML не та. Проверять уровень звука:
```bash
ffmpeg -i out.mp4 -af volumedetect -f null - 2>&1 | grep mean_volume
```
Тишина = −91 dB.

---

## Calliope (benjiyaya) + наш MLX-конвейер

Установлено в `<корень>/calliope`, данные — в
`~/ComfyUI-Shared/calliope-data`. Бэкенд :8247, фронт :5173, LLM — локальная Ollama.

### Установка на macOS (в репозитории только Windows-скрипты)

`setup.bat`/`start.bat`/`stop.bat` — единственные скрипты в репо. Написаны
`start.sh` и `stop.sh` рядом с ними (пишут pid в `logs/`, логи туда же).

Ставить бэкенд ТОЛЬКО через `pip install -e ".[dev]"` из `calliope-backend`.
Установка одного `requirements.txt` не кладёт сам пакет (src-layout), и запуск
падает с `ModuleNotFoundError: No module named 'calliope'`.

`comfy-mcp>=0.10.0` из requirements — это НЕ наш comfy-cli-шный comfy-mcp 1.20.
Живёт в своём venv, конфликта нет.

### Ролевые теги: как Calliope находит слоты

Граф подаётся в **API-формате**, роли читаются из `_meta.title`:
регэксп `\((Input|Output)(?::([a-zA-Z0-9_-]+))?\)`, регистронезависимо.
Есть алиасы: `face|ref|portrait → character`, `seconds|length → duration`,
`scene|background → location` и т.д. (`comfyui/roles.py`).

**Главная засада.** Патчер (`comfyui/patcher.py`) пишет **одно значение на ноду**,
а имя поля выводит из `class_type` через `registry.class_to_patch_field`.
Наши MLX-классы реестру неизвестны, и эвристика по подстроке даёт неверное поле:
`LTXVMLXTextEncode` содержит «text» → патчер полез бы в `inputs["text"]`, тогда как
реальные виджеты называются `prompt` и `negative_prompt`. Значение молча ушло бы в никуда.

Вывод: **вешать теги прямо на кастомные MLX-ноды нельзя.** Точками входа должны быть
`Primitive*`-ноды (у них поле `value`, и реестр это знает — `startswith("Primitive")`),
подключённые ссылками ко входам сэмплера:

| роль | нода | куда идёт |
|---|---|---|
| `(Input:prompt)` / `(Input:negative)` | `PrimitiveStringMultiline` | `LTXVMLXTextEncode.prompt` / `.negative_prompt` |
| `(Input:duration)` | `PrimitiveFloat` | `duration` — именно FLOAT, не INT |
| `(Input:seed)` / `(Input:width)` / `(Input:height)` | `PrimitiveInt` | одноимённые входы сэмплера |
| `(Input:image)` | `LoadImage` | тег вешается на саму ноду — `LoadImage` реестру известен |
| `(Output:video)` | `SaveVideo` | известен как видеовыход |

Несколько `(Input:image)` заполняются **по возрастанию node id** — порядок нод задаёт,
какая картинка станет первым кадром, а какая последним.

Готовый размеченный граф: `~/ComfyUI-Shared/workflows-api/MLX-Keyframe-FLF_API.json`.

`SaveVideo` требует `format.codec` (dynamic-combo). Без него `comfy validate` ругается
`required_input_missing`, а исполнение падает. Ставить `"auto"`.

### Локальная LLM

Ollama отдаёт рассуждения reasoning-моделей **отдельным полем** `reasoning`, а `content`
остаётся чистым JSON. Поэтому qwen3/gemma4 работают без обрезки `<think>`.

Отключить «думанье» из конфига нельзя: `extra_body` проброшен только для одного агента
(`h3_rewrite_extra_body`). Плата — скорость: на JSON-задачу qwen3:14b тратит ~18 с,
gemma4:12b-mlx ~36 с. Выбран **qwen3:14b**.

### Тайминги и грабли прогона (M2 Max, 32 ГБ)

- `generate-story` — **624 с** (10,5 мин), 10 последовательных вызовов LLM.
- Видео 4 с / 512×320 через наш MLX-граф — **~11,5 мин**, включая копирование в assets.

**Обрыв HTTP-клиента убивает генерацию.** FastAPI отменяет корутину, когда клиент
отвалился по таймауту, — первый прогон story умер на 5-м вызове, проект остался
полузаполненным. Долгие запросы запускать отцепленно (`nohup`), а не ждать в curl.

`generate-story` **игнорирует `target_duration`**: на заказ «12 сек» выдал 60 битов.
Количество сцен придётся резать вручную.

### Что подтверждено прогоном

Calliope корректно разложил роли по нодам и отправил в ComfyUI:
`{"10": prompt, "11": negative, "12": 4, "13": 42, "14": 512, "15": 320}`.
Кадры проверены глазами — первый и последний совпадают с заданными keyframe'ами.
Середина уходит в посторонний план (лес вместо лисы) — это поведение самой LTX
при далёких друг от друга кадрах, к Calliope отношения не имеет.

### Story-агент переведён на Claude (подписка, без API-ключа)

Вместо локальной Ollama раскадровку пишет Claude — через шим
`~/ComfyUI-Shared/scripts/claude-shim/shim.py`: OpenAI-совместимый сервер на :8317,
который транслирует `/v1/chat/completions` в `claude -p --output-format json`.
Работает на подписке Claude Code, API-ключ не нужен. Поднимается и гасится
вместе с Calliope (`start.sh` / `stop.sh`), лог — `calliope/logs/shim.log`.

Шим умеет non-stream, SSE и `response_format: json_object` (JSON выпрашивается словами,
на выходе снимается ```-ограждение).

**Обвес CLI надо срезать.** Без флагов каждый вызов тащит ~40k токенов системного
промпта Claude Code со схемами инструментов и MCP-серверов. Комбинация
`--system-prompt` + `--exclude-dynamic-system-prompt-sections` + `--strict-mcp-config`
+ `--disallowed-tools <все>` ужимает до ~27k, и при повторных вызовах это читается
из кеша: ответ за 1,2–4 с вместо 18–35 с у qwen3:14b.

CLI запускается в пустой `workdir/`, чтобы не втянуть CLAUDE.md и файлы проекта.

#### Две правки в коде Calliope (бэкапы `.bak` рядом)

1. **Роль `story` была объявлена, но не подключена.** `AGENT_LLM_ROLES` содержит
   `story`/`script`, в настройках их можно назначить — но `generate_structured()`
   создавала `LLMClient()` без роли, то есть всегда активный профиль. Назначение
   молча не работало. Добавлен параметр `role`, и он проброшен из `routers/story.py`
   (5 вызовов, `role="story"`) и `agent/script_agent.py` (2 вызова, `role="script"`).

2. **Парсер длительности не знал русских единиц.** `estimate_target_seconds()` искал
   только `min|sec|s|m`. «12 сек» не распознавалось, число уходило в ветку «голое
   число = минуты» → 720 секунд → **60 битов вместо 4**. А «1 мин» читалось как
   1 секунда. Добавлены `мин\w*|м` и `сек\w*|с`.

После обеих правок: «40 секунд» → 4 бита за **19 с** (было 624 с и 60 битов).

#### Что осталось на Ollama

Активный профиль по-прежнему `qwen3:14b` — на нём роли `assets`, `video`, `main`,
`planner`. На Claude переведены только `story` и `script`. Переключается в
Settings → LLM profiles, без правки кода.
