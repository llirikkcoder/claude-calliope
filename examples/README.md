# Пример размеченного графа

`MLX-Keyframe-FLF_API.json` — граф LTX-2.3 MLX: переход между первым и последним
кадром, 512×320, ~11 минут на 4 секунды видео на M2 Max.

Перед импортом подставь свои пути к моделям в нодах `1` и `2` (сейчас там
плейсхолдеры `/ЗАМЕНИ/…`) и положи `flf_start.png` / `flf_end.png` в `input`
своего ComfyUI.

Разметка устроена так, как описано в навыке: промпт, негатив, длительность, сид и
размер вынесены в `Primitive`-ноды — вешать теги прямо на MLX-ноды нельзя, значение
потеряется молча. Теги `(Input:image)` стоят на двух `LoadImage`: нода `4` даёт
первый кадр, нода `5` — последний, порядок определяется возрастанием id.

| нода | класс | тег |
|---|---|---|
| 4 | LoadImage | `(Input:image)` первый кадр |
| 5 | LoadImage | `(Input:image)` последний кадр |
| 8 | SaveVideo | `(Output:video)` |
| 10 | PrimitiveStringMultiline | `(Input:prompt)` |
| 11 | PrimitiveStringMultiline | `(Input:negative)` |
| 12 | PrimitiveFloat | `(Input:duration)` — именно FLOAT |
| 13 | PrimitiveInt | `(Input:seed)` |
| 14 / 15 | PrimitiveInt | `(Input:width)` / `(Input:height)` |
