# pipeline_softsegmenter_longyaal vs longyaal2 分词/对齐对比

## Ref（参考）侧

| 项目 | longyaal | longyaal2 (use_unit_punct_tokenize=True 默认) |
|------|----------|-----------------------------------------------|
| 入口 | `_load_reference_qwen_units` 唯一路径 | `_load_reference_unit_punct` |
| 是否按 char_level 切 ref | **否**，无 char_level 参数 | **是**：`char_level=True`（ja/zh）时 ref = **逐字符** `[(c,i,i+1)]` |
| 切分方式 | 直接 `qwen_tok.encode_timestamp(ref_sentence, ref_unitizer_language)` → 日语=nagisa 词 | `_ref_units_with_spans(..., char_level)` → 若 char_level 则按字，否则 Qwen + norm+find 得 span |
| 是否插标点 | **否**，只有 units | **是**，units 之间插标点 token（纯空格不插） |
| 日语 ref 粒度 | **词**（nagisa） | **字**（与 longyaal 不一致） |

## Hyp（假设）侧

| 项目 | longyaal | longyaal2 (use_unit_punct_tokenize=True) |
|------|----------|-----------------------------------------|
| 入口 | `_load_hypothesis_prediction_units` | `_load_hypothesis_unit_punct` |
| 数据来源 | `prediction_units` + delays + elapsed | 同上 + `prediction_unit_char_starts_ends` |
| 是否插标点 | **否** | **是**，unit 之间按 starts_ends 取 gap 作标点 token |
| 粒度 | 与 hypothesis 的 prediction_units 一致（多为词） | 同上 |

## 结论

- longyaal：ref = **Qwen（nagisa）词**，hyp = **prediction_units 词**，**词对词**，无标点 token。
- longyaal2 默认：ref = **char_level 时按字**，hyp = 词+标点 → **字对词**粒度不一致，导致对齐乱（如 す–こんにちは）。
- longyaal2 在「longyaal 基础上」应只增加：标点 token、span 切片、延迟只算词。**ref 不应改成按字切**。

## 修复

在 longyaal2 的 `_load_reference_unit_punct` 中，ref 的 unit 切分与 longyaal 保持一致：**始终用 Qwen 切 ref**，不按 char_level 把 ref 切成字。即调用 `_ref_units_with_spans(..., char_level=False)` 用于 ref 的 unit 切分；`char_level` 仅用于对齐度量、prediction 拼接、latency_unit 等，不用于 ref 分词。
