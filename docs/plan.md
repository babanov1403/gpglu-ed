# MiniRT — план разработки

Версия 5.0. Цель — небольшой single-GPU C++20/CUDA inference runtime для
Qwen2.5-0.5B semantics и воспроизводимых kernel-экспериментов.
Основной workflow: implement → validate → benchmark → integrate → explain.

## 1. Результат и границы

Целевой v0.1 включает TinyModel, contiguous и paged KV-cache, greedy decode,
token-budget scheduler с continuous batching и одно законченное исследование.
Операторное ускорение проверяется на model/engine path; отрицательный результат
тоже считается результатом, если гипотеза проверена и ограничения объяснены.

Три public extension points: `RmsNormBackend`, `GemmBackend`,
`AttentionStageBackend`. Поддерживаемый backend выбирается через config и static
registry без веток по его имени в `ModelRunner`, KV manager или scheduler.
RoPE, KV write, SwiGLU, embedding и argmax остаются internal primitives.

PyTorch используется для reference, генерации fixtures и экспорта весов.
Runtime и CUDA launchers не зависят от `at::Tensor`.

Единые источники требований:

| Документ | Что фиксирует |
|---|---|
| [ADR-0001](ADR/ADR-0001.md) | Qwen semantics, TinyModel и независимый reference |
| [ADR-0002](ADR/ADR-0002.md) | RTX 2060 SUPER, sm_75 и toolchain |
| [ADR-0003](ADR/ADR-0003.md) | FP16, accumulation, tolerances и layouts |
| [ADR-0004](ADR/ADR-0004.md) | Backend boundary, lifecycle и API freeze |
| [ADR-0005](ADR/ADR-0005.md) | Paged KV, admission и scheduler progress |
| [Проверки и измерения](validation.md) | Correctness, benchmark и release evidence |
| [Работа с тикетами](workflow.md) | `[epic]` → `[US]` → один `[research]` + `[core]` |

## 2. Время и реальное сокращение scope

Лимит — 10 часов в неделю, 24 рабочие недели, всего 240 часов. Это бюджет,
а не обещание закончить весь целевой scope за любое количество отладки.
Планируется 180 часов работы и 60 часов резерва внутри этих 240 часов.
Обычно в неделю берётся 7–8 часов задач; оставшееся время поглощает отладку.
Чтение, проверки, profiling и документация входят в оценки задач.

Прогноз пересматривается после M1 и M3 по фактически затраченным часам.
Инфраструктура расширения начинается с одного работающего RMSNorm backend;
новые общие механизмы появляются только по требованиям следующего stage.

| Режим | Что заканчиваем | Что переносим |
|---|---|---|
| Целевой v0.1 | M1–M6, включая paged KV и scheduler | Только optional work |
| Сокращённый v0.1-alpha | M1–M3, contiguous TinyModel и одно исследование | M4 и M5 целиком; API остаётся candidate |
| Отставание после готового M4 | Paged TinyModel и исследование, v0.1-alpha | M5 целиком |

Если после M3 прогноз оставшихся обязательных работ вместе с исследованием и
проверками не помещается в остаток бюджета, выбирается сокращённый режим.
Если к концу недели 16 paged parity не достигнута, scheduler GPU integration
не начинается. Последние 36 часов на исследование, исправления и оформление
результата защищены от новых подсистем. Недостающие release gates нельзя
объявить пройденными ради даты; v0.1 требует полного целевого результата.

FP32 GEMM laboratory — optional, максимум 8 часов из свободного остатка после
текущего gate. Наличие naive/tiled GEMM не блокирует ни один milestone.

## 3. Этапы

M1–M6 — этапы roadmap, а не шесть обязательных GitHub milestones.
Каждый этап соответствует одному `[US]`; подробные `[core]` создаются только
для текущего US как маленькие шаги на одну сессию (ориентир 1–3 часа).
US задаёт порядок; каждый шаг можно проверить и закрыть до следующего.
В каждом US ровно один `[research]` с конкретным вопросом.
Исследования M1–M5 — короткие проверки решений, а не пять дополнительных papers.
Один выбранный вопрос получает глубокое исследование в M6.

| Этап / US | Ориентир | Работа / резерв | Проверяемый результат |
|---|---|---:|---|
| M1 — Первый kernel-эксперимент | Недели 1–4 | 24 / 16 ч | Runtime, FP16 cuBLASLt, зарегистрированный RMSNorm, correctness и JSONL |
| M2 — Contiguous attention | Недели 5–8 | 30 / 10 ч | RoPE, causal prefill, KV append и sequential decode с GQA |
| M3 — TinyModel | Недели 9–12 | 36 / 4 ч | Transformer block, 2-layer model, greedy decode и реальные hotspots |
| M4 — Paged model path | Недели 13–16 | 30 / 10 ч | Prefill → paged decode, request isolation и динамические shapes |
| M5 — Continuous batching | Недели 17–20 | 24 / 16 ч | Admission, меняющийся decode batch и deterministic workloads |
| M6 — Исследование и релиз | Недели 21–24 | 36 / 4 ч | Проверенная гипотеза, raw results, profiler evidence и demo |
| Всего | 24 недели | 180 / 60 ч | 240 часов |

### M1 — Первый воспроизводимый kernel-эксперимент

- Завершить RAII wrappers, `DeviceBuffer`, public-header compile checks и
  физическую public/internal boundary.
- Реализовать `ExecutionContext`: engine-owned stream, cuBLASLt handle,
  workspace, CUDA errors и allocation counters.
- Получить корректный FP16 linear path через cuBLASLt с FP32 accumulation.
- Подключить FP32 RMSNorm через static registry и `supports()`; один общий
  correctness/benchmark path сохраняет JSONL, environment и reproduction command.
- Проверить non-default stream, input immutability и отсутствие device allocation
  в `run()`; снять один Systems trace и targeted Compute report.

Research: какие reference, baseline и timing protocol позволяют честно измерять
малые RMSNorm/linear workloads на sm_75? Результат — короткое решение до реализации
зависимых kernels и harness. Smoke measurements, профили и итоговый отчёт
выполняются последующими core-шагами по выбранному протоколу.

Gate: оба операторных пути корректны; RMSNorm выбирается через registry;
результат воспроизводится документированной командой. Manifests, scaffold,
универсальный CLI и custom GEMM не входят в gate.

### M2 — Contiguous attention

- Реализовать точную RoPE semantics, FP16 core path, contiguous KV population
  и append, GQA mapping и naive decode attention.
- Baseline prefill: QKᵀ → scale/causal mask → stable softmax → V.
  Materialized scores допустимы; maximum prompt задаётся по доступной памяти.
- Ввести candidate `AttentionStageBackend` с отдельными prefill/decode paths.
  Добавить FP16 RMSNorm до интеграции блока.

Research: как выбрать простой корректный prefill/decode baseline и workspace
для target shapes, сохранив место для fusion внутри attention stage?

Gate: prompt → KV population → первый decode → несколько следующих шагов
совпадают с reference; пройдены GQA, causal boundary и allocation checks.

### M3 — Transformer block и TinyModel

- Сгенерировать TinyModel из ADR-0001, простой versioned weight format и C++ loader.
- Собрать embedding → decoder blocks → final norm → LM head → greedy token IDs.
- Проверить intermediate tensors, logits и KV относительно закреплённого
  upstream Qwen2 reference с теми же синтетическими весами.
- Реализовать prepared plans, persistent storage, reusable step workspace и
  buffer reuse между layers. Model GEMMs используют cuBLASLt.
- Снять target-shape one-block profile и профиль TinyModel; пройти выбранную
  Compute Sanitizer matrix и steady-state allocation audit.

Research: какие operations ограничивают prefill/decode на target shapes и какой
один вопрос имеет измеримый model-level смысл? TinyModel timing не выдаётся за
производительность полного Qwen checkpoint.

Gate: один блок и 2-layer TinyModel проходят parity, несколько greedy steps
работают, hotspots измерены. Здесь выбирается полный или сокращённый scope.

### M4 — Paged KV и проверка API

- Реализовать model-level BlockPool, generation handles, rollback, per-request
  block tables и preallocated host/device metadata по ADR-0005.
- Добавить prompt KV scatter в paged storage и paged append/decode через тот же
  registry/config path. Оптимизированный paged kernel необязателен.
- Проверить новый запрос → prefill → paged decode на реальном model path,
  страницы B−1/B/B+1, несколько requests, release/reuse и KV contents parity.
- До API freeze прогнать batch `1 → 4 → 2`, growing contexts и non-default stream
  на подготовленных resources без новых allocations и lazy prepare.

Research: как block size влияет на allocated slots, metadata и decode latency
при фиксированных shapes? Начать с representative cases, расширять matrix
только для выбранного глубокого исследования.

Gate: весь переход prefill → paged decode корректен, isolation подтверждена,
capacity contract проверен динамическим batch. Только затем API получает v1.

### M5 — Scheduler и continuous batching

- Реализовать lifecycle `Waiting → Admitted → Prefilling → Decoding → Finished`
  с явными `Rejected` и `Failed` по ADR-0005.
- Сначала проверить policy через fake executor, затем подключить GPU.
- `BatchPlan` содержит IDs, token/position IDs, lengths, slot mappings,
  block tables и output slots. Новые запросы допускаются между iterations.
- Prefill и decode выполняются отдельными sub-batches; chunked и mixed packed
  execution не требуются.
- Прогнать uniform short, mixed-length и KV-pressure traces, включая arrival
  во время decode, EOS, MaxOutLen, невозможный prompt и освобождение резервов.

Research: как continuous batching меняет TTFT, TPOT, throughput и KV utilization
относительно static batching на одинаковом workload?

Gate: каждый принятый запрос продвигается до terminal state; budgets соблюдаются;
batch меняется между iterations; compatible backends не требуют правок scheduler.

### M6 — Одно глубокое исследование

Выбрать один track по profiling: paged decode, small-M GEMM, RoPE + KV write или
KV block size. Для сокращённого scope выбрать вопрос на contiguous model path.

Внутри 36 часов: 24 часа — гипотеза, baseline, 2–3 изолированных variants,
correctness, benchmark и profiling; 8 — report и воспроизводимость;
4 — исправления. `[research]` один; реализация variants и интеграции — `[core]`.

Gate: operator и model/engine results, отдельные Systems/Compute reports,
limitations, negative results, raw data, reproduction command и clean-build demo.
Ускорение не является обязательным условием закрытия вопроса.

## 4. Итоговая проверка

v0.1 требует всех M1–M6 gates и проверок из [validation.md](validation.md).
В частности: TinyModel; contiguous/paged parity; dynamic batches; request
isolation; zero steady-state device allocations; три config-selected backend
families; API v1 после M4; одно законченное исследование и воспроизводимый demo.

v0.1-alpha явно перечисляет завершённые gates и перенесённые подсистемы.
Contributor-ready v0.2 возможен после реального опыта использования API.

## 5. Что остаётся после v0.1

Полный checkpoint, FP32/Tensor Core GEMM laboratory, optimized prefill,
chunked prefill, mixed packed batches, CUDA Graphs, sampling, prefix caching,
copy-on-write, второй dtype/GPU и дополнительные public extension points.

Manifest ecosystem, scaffold, dynamic plugins, out-of-tree packaging, polished
onboarding и автоматические reports появляются только при подтверждённой
потребности. Stable binary ABI, network server и distributed execution не
входят в текущий продукт.
