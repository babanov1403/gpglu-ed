# Correctness и измерения MiniRT

Это целевой контракт проверок. Suite, test target и CLI-команда добавляются
вместе с работающим path; их наличие в документе не означает, что они реализованы.

## Reference и fixtures

- Operator reference — Python FP32/FP64; runtime precision задаётся ADR-0003.
- Model reference — закреплённый upstream Qwen2 из ADR-0001 с теми же
  синтетическими весами, включая ненулевые Q/K/V biases. Собственный Python
  reference дополнительно проверяется против upstream.
- Fixture identity: generator version, problem signature, seed, input hash,
  reference output hash. Failing case печатает эти данные.
- TinyModel не требует checkpoint download. Target-shape operator/block fixtures
  обязательны отдельно от TinyModel; его малые dimensions не заменяют GQA 14/2.
- Weight format: version, endianness, alignment, architecture/config, tensor name,
  dtype, shape, layout, offset, byte length и checksum. Loader проверяет bounds,
  shapes/orientations, обязательные bias tensors и tied embeddings.

## Уровни correctness

| Уровень | Обязательные сценарии по мере реализации |
|---|---|
| Operator | RMSNorm, RoPE, GEMM, softmax, SwiGLU, embedding, KV append, argmax |
| Composition | Norm + QKV + bias; RoPE + KV write; causal attention; residuals; prefill + first decode |
| Layer | Prefill/decode блока, intermediate tensors, KV contents, repeated steps |
| Model | TinyModel logits, hidden states, top-1/top-k и greedy sequence |
| Engine | Arrivals/completions, changing batches, budgets, EOS/MaxOutLen, request isolation |

Conformance public backend-а проверяет semantic, memory, stream и lifecycle
contract; internal tests дополнительно проверяют allocator и request state.

Общие cases: minimum и maximum supported shapes, target shapes, tile/vector
boundaries, non-multiple sizes, invalid dtype/layout/shape, insufficient workspace,
misaligned buffers, repeated prepare/run, input immutability и non-default stream.
Проверяются truthful `supports()`, actionable errors, cleanup после неуспешного
prepare, selected canaries и отсутствие host/device allocations в `run()`.

Attention matrix включает GQA, causal boundary, prompt/context length 1,
short/long contexts, first decode и несколько последовательных шагов.
Paged matrix дополнительно требует:

- свежий запрос → prefill/scatter → paged decode на model path;
- B−1/B/B+1 и partial final page;
- multiple requests, mixed context lengths и несколько страниц;
- release/reuse sentinel и contiguous/paged KV/logits parity;
- batch `1 → 4 → 2` с growing contexts на заранее подготовленных resources;
- отсутствие записи в free/foreign blocks.

Scheduler fake-executor tests покрывают невозможный prompt, запрос сверх полной
KV capacity, временный недостаток reservation, progress после освобождения,
MaxOutLen/EOS, budget saturation и batch failure. GPU trace подтверждает те же
semantics и отсутствие steady-state device allocations.

## Численные результаты

Сохраняются `max_abs_error`, `max_relative_error`, `relative_l2_error`,
`mean_abs_error`, `allclose`, `atol`, `rtol`, NaN/Inf counts. Cosine similarity —
дополнительная метрика. Для logits: top-1 agreement, top-k overlap и reference
logit margin. Околонулевые значения не оцениваются одним relative error.

Tolerances versioned по ADR-0003; их ослабление требует failing fixture и
объяснённого решения. Для greedy fixtures фиксируются tie-breaking и достаточный
logit margin; расхождение токенов анализируется вместе с ошибкой logits.
Чистое копирование уже квантованных KV проверяется побитово; fused RoPE + write
допускает только численную погрешность самой RoPE.

Результаты: `PASS`, `FAIL`, `UNSUPPORTED`, `SKIPPED`, `INFRA_ERROR`.
Unsupported разрешён для declared optional/unsupported cases, но не заменяет
обязательное покрытие model path. Mandatory failure даёт non-zero exit code.

## Memory и stream safety

Проверки включают partial model-load/prepare failure, repeated create/destroy,
randomized BlockPool state machine, rollback, stale handles, out-of-range positions,
block-table overflow, request reuse и insufficient workspace.

После prepare/warmup запрещены device allocations в backend/model/engine iteration
и host allocations в per-layer execution. Counters дополняются targeted Systems
trace, чтобы увидеть allocations и synchronization внутри библиотек.
Zero device allocations не запрещает выдачу логических blocks из готового pool.

Compute Sanitizer запускается на выбранных regular, non-multiple, GQA, decode,
page-boundary и reuse cases. Режимы memcheck/racecheck/initcheck/synccheck
применяются там, где соответствующий класс ошибок возможен; полный набор на
каждой shape не требуется.

## Benchmark protocol

Один общий timing harness используется всеми public backend families.

1. Зафиксировать suite/config, backend и baseline; проверить conformance и support.
2. Подготовить одинаковые inputs, seeds и weights; выполнить warmup.
3. Измерить несколько независимых samples. Для коротких kernels допускается
   несколько launches на sample; количество сохраняется.
4. GPU events и wall time измерять отдельно с явной synchronization policy.
   Setup/prepacking/prepare отделяются от steady-state, но не теряются из отчёта.
5. Для stateful KV/model workloads явно определить reset/replay. У candidate и
   baseline одинаковые начальные KV, positions и generated-token trace; стоимость
   reset указывается, iteration не выдаётся за повтор того же shape после роста KV.
6. Сохранить raw samples, p50/p90; p99 публиковать только при достаточной выборке.
   При шуме повторить paired/interleaved candidate/baseline runs и описать разброс.

Profiler запускается отдельно от final timing. Failure/unsupported не получает
нулевую latency. Conformance failure запрещает publishable speedup claim.

Suite фиксирует version, semantic contract, fixture version, shapes/dtypes/layouts,
required conformance, baselines, warmup/samples/launches, measurement mode и
GPU-state policy. Suites создаются по этапам: RMSNorm, small-M GEMM, contiguous
attention, decoder block, paged decode, scheduler mixed lengths.

Baseline должен быть указан по имени и версии и реально работать на sm_75.
Python mathematical reference сам по себе не является strong GPU baseline.
Если доступен только naive baseline, claim явно ограничивается этим сравнением.
Выбор baseline проверяется в research текущего US до оптимизаций.

## Result bundle

```text
results/raw/<experiment_id>/
  results.jsonl
  conformance.json
  environment.json
  config.json
  command.txt
```

JSONL содержит schema/suite/backend/API versions, problem signature,
shape/layout/dtypes/compute type, seed/input/weights hashes, measurement scope,
raw samples, summary metrics, workspace/persistent/transient bytes, correctness,
status/error и reproduction command. Resolved config и environment связаны с run.

Environment: GPU/SM/VRAM, driver/toolkit, compiler/CMake, OS, build flags,
git commit и dirty state; для measurements — доступные clock/power/temperature
данные и сведения о конкурирующей GPU-нагрузке. Неизвестное значение отмечается.
Опубликованные raw data и suites immutable; изменения получают новую версию.

GEMM metrics: M/N/K, algorithm ID, GFLOP/s, ratio к cuBLASLt. Memory-bound kernels:
estimated bytes и effective bandwidth. Attention: phase, heads, context, batch,
block size/layout. Engine: queue time, TTFT, inter-token latency/TPOT, throughput,
accepted/rejected/failed counts, KV utilization и peak memory.

TTFT считается от arrival до доступности первого output token; queue time — до
начала исполнения. Inter-token latency измеряется между доступными output tokens;
TPOT усредняет интервалы после первого токена и не определён для одного токена.
Сравнения static/continuous используют одинаковые arrivals и output limits;
учёт отклонённых запросов и знаменатель throughput записываются явно.

Systems evidence: launch gaps, copies, sync, allocations и NVTX layer/stage/engine
ranges. Compute evidence: representative kernel, occupancy/registers, DRAM/L1/L2,
stalls, instruction mix и Tensor Core usage, где применимо. Каждый вывод связан
с командой, commit, shape, report и отдельно измеренным benchmark result.

## CLI и gates

Целевые команды добавляются по мере реализации:

```text
minirt backends list
minirt conformance --backend <name> --suite <suite>
minirt bench --backend <name> --baseline <name> --suite <suite> --output <dir>
minirt model run --config <model> --backend-config <backends>
minirt workload run --config <engine> --trace <trace>
```

Human summary отделён от machine-readable output. Ошибка содержит backend,
case и причину; runner печатает reproduction command. Universal compare UI,
manifest/scaffold ecosystem и отдельные tutorials не являются gates.

На PR запускаются только существующие проверки, подходящие изменению:
format/build, standalone public headers, boundary check, CPU references/fixtures
и fake scheduler. CUDA compile требует toolkit даже без GPU; CPU-only tests
не должны требовать запуск CUDA. GPU gate: smoke conformance, TinyModel,
non-default stream, allocation audit и выбранный sanitizer subset.

Release gate — этапы заявленного scope, clean-build reproduction, проверенный
result bundle и главное исследование. Performance claims подтверждаются
operator и model/engine evidence; неприменимость объясняется в отчёте.
