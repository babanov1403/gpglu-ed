# MiniRT — kernel-first single-GPU LLM inference laboratory

> Версия 4.0. Шестимесячный план разработки C++/CUDA inference runtime, в котором главным продуктом является работающий engine и среда для честных kernel-экспериментов. Contributor-friendly API остаётся архитектурным ограничением, но полноценная contributor platform не является release-gate для первых шести месяцев.

## 1. Миссия проекта

Создать **kernel-first single-GPU experimental LLM inference runtime на C++/CUDA** для одной точно определённой decoder-only архитектуры.

Главный результат проекта — работающий inference engine, в котором можно:

```text
Implement → Validate → Benchmark → Integrate → Explain
```

То есть:

1. реализовать новый operator или model stage;
2. проверить его математическую, memory- и stream-correctness;
3. измерить на фиксированных shapes и workload-ах;
4. заменить backend без переписывания `ModelRunner`;
5. увидеть влияние не только на microbenchmark, но и на layer/model/engine path;
6. объяснить результат через Nsight Systems, Nsight Compute и измеримые ограничения.

Contributor-friendly архитектура сохраняется, но является **средством**, а не отдельным продуктом первого шестимесячного релиза. MiniRT не обязан к v0.1 предоставлять полноценный onboarding для внешнего contributor-а, генератор backend-ов, стабильную manifest ecosystem или community-ready release process.

Главная ценность проекта:

- небольшой и читаемый C++/CUDA execution path;
- возможность заменять GEMM, RMSNorm и attention backends;
- точные semantic, layout, ownership и stream contracts;
- PyTorch-независимые CUDA launchers;
- воспроизводимая correctness и performance infrastructure;
- связь operator-level ускорения с model- и scheduler-level эффектом;
- один законченный research result с profiler evidence;
- возможность публиковать как положительные, так и отрицательные результаты.

### 1.1 Главное продуктовое обещание v0.1

> Автор runtime может добавить или заменить `RmsNormBackend`, `GemmBackend` или `AttentionStageBackend` через ограниченный public API и registry/config path, не добавляя backend-specific ветки в `ModelRunner`, KV allocator или scheduler.

Это обещание является архитектурным invariant v0.1.

Полноценное обещание для внешнего contributor-а — clean clone, scaffold, tutorials, out-of-tree build и самостоятельный PR journey — переносится в post-v0.1 roadmap.

### 1.2 Golden path разработчика kernels

Целевой workflow первого релиза:

```bash
git clone <repo>
cmake --preset release
cmake --build --preset release

./build/bin/minirt backends list

./build/bin/minirt conformance \
    --kind rmsnorm \
    --backend my_rmsnorm \
    --suite rmsnorm-smoke-v1

./build/bin/minirt bench \
    --kind rmsnorm \
    --backend my_rmsnorm \
    --baseline reference,cuda_naive \
    --suite rmsnorm-target-v1 \
    --output results/raw/exp-001

./build/bin/minirt model run \
    --config configs/model/tiny.json \
    --backend-config configs/backend/my_rmsnorm.json
```

Минимальный output:

```text
correctness verdict
unsupported cases with reasons
latency table
speedup versus baseline
operator- или model-level impact
raw JSONL
environment manifest
reproduction command
```

`minirt compare`, автоматический Markdown report, manifests и scaffolding полезны, но не обязательны для шестимесячного v0.1.

### 1.3 Что означает «доступный» в v0.1

Доступность оценивается следующими проверяемыми свойствами:

- operator experiments не требуют real checkpoint;
- есть deterministic synthetic fixtures;
- есть tiny generated model той же архитектурной семантики;
- public и internal headers физически разделены;
- как минимум для RMSNorm и attention есть понятные reference/example implementations;
- backend возвращает actionable `UNSUPPORTED`, а не падает obscure runtime error;
- correctness и benchmark запускаются одной командой;
- результаты сохраняются машинно-читаемо;
- замена поддерживаемого backend-а не требует правок `ModelRunner`;
- backend hot path не выполняет device allocation и global synchronization.

Не являются критериями v0.1:

```text
внешний contributor dry run;
scaffolding tool;
manifest schema ecosystem;
несколько polished tutorials;
stable binary ABI;
community-ready release process.
```

### 1.4 Целевой inference execution path

```text
Token IDs
   ↓
Engine / Scheduler
   ↓
BatchPlan
   ↓
ModelRunner
   ├── embedding
   ├── N × DecoderBlock
   │      ├── AttentionStageBackend
   │      │      ├── RMSNorm
   │      │      ├── QKV projection
   │      │      ├── RoPE
   │      │      ├── KV-cache write/read
   │      │      └── prefill/decode attention
   │      └── MLP path
   │             ├── RMSNorm
   │             ├── gate/up projections
   │             ├── activation / SwiGLU
   │             └── down projection
   ├── final norm
   └── LM head
   ↓
Logits / generated token IDs
```

PyTorch находится **за пределами runtime** и используется только для:

- загрузки исходного checkpoint;
- экспорта весов;
- математических reference-реализаций;
- генерации deterministic fixtures;
- получения intermediate reference tensors;
- проверки logits и token IDs;
- сравнительных benchmarks.

CUDA launchers, backend API, `ModelRunner`, KV-cache и scheduler не зависят от `at::Tensor`.

---

## 2. Scope, бюджет и критерии завершения

План рассчитан на:

```text
10 часов в неделю
24 рабочие недели
240 часов суммарно
```

Оставшиеся календарные недели шестимесячного периода считаются резервом на рабочие авралы, болезни, flaky tooling и CUDA debugging.

Рекомендуемый недельный ритм:

```text
6 часов — runtime/model/kernel implementation;
2 часа  — correctness, benchmark и profiling;
1 час   — paper/documentation reading или research notes;
1 час   — report, README, ADR и GitHub issue planning.
```

Contributor-friendly infrastructure получает примерно **15–20 часов внутри 240 часов**. Эти часы тратятся только на минимальные extension seams, registry, conformance и benchmark path. Полноценная contributor platform не конкурирует с attention, KV-cache, model path и scheduler.

### 2.1 Mandatory scope v0.1

К концу шести месяцев обязательно существуют:

#### Runtime и model path

- один зафиксированный `ArchitectureContract`;
- одна target GPU architecture;
- один обязательный low-precision dtype;
- собственный C++/CUDA runtime core;
- explicit `cudaStream_t` и engine-owned memory;
- custom RMSNorm и RoPE;
- low-precision cuBLASLt linear backend;
- короткий учебный FP32 GEMM laboratory;
- baseline prefill path;
- contiguous KV-cache и decode attention;
- один корректный Transformer block к концу двенадцатой недели;
- tiny 2-layer model той же архитектурной семантики;
- paged KV-cache и paged decode attention;
- token-budget scheduler;
- continuous batching между iterations.

#### Минимальная experimentation/extension infrastructure

- физическая public/internal header boundary;
- source-level `BackendApiVersion`;
- static `BackendRegistry`;
- public extension points только для `RmsNormBackend`, `GemmBackend` и `AttentionStageBackend`;
- `supports(problem, device)` с actionable reason;
- config-driven backend selection;
- один RMSNorm example backend;
- один attention example/reference backend;
- conformance runner для реализованных public extension points;
- benchmark runner с единым timing protocol;
- deterministic synthetic fixtures;
- machine-readable JSONL и environment manifest;
- минимальные CLI-команды для discovery, conformance, benchmark и model/workload run.

#### Research quality

- operator/composition/layer/model/engine correctness;
- selected Compute Sanitizer matrix;
- Nsight Systems и Nsight Compute reports;
- один глубокий research result;
- strong baseline и isolated variants;
- explicit limitations и negative results;
- воспроизводимый GitHub repository.

### 2.2 Deferred contributor platform

Следующие задачи **не являются mandatory scope первых шести месяцев**:

- backend manifest schema и отдельное manifest versioning;
- scaffolding generator `tools/new_backend.py`;
- polished out-of-tree backend package;
- отдельные tutorials для каждого extension point;
- полноценный `minirt compare` и автоматическая генерация plots/Markdown reports;
- comprehensive suite discovery/matrix UI;
- external contributor dry run;
- UX time targets для стороннего разработчика;
- community-ready governance и release gates;
- stable binary ABI и dynamic plugins;
- поддержка нескольких architectures или GPU generations.

Эти пункты допускаются только как stretch goals после завершения engine path и research result.

### 2.3 Stretch scope

Только после mandatory scope:

- полный checkpoint класса 0.5–1B;
- custom fused prefill / FlashAttention-like kernel;
- Tensor Core GEMM laboratory;
- chunked prefill;
- mixed prefill и decode в одном packed GPU batch;
- CUDA Graphs;
- prefix caching;
- copy-on-write;
- sampling;
- второй low-precision dtype;
- public `MlpStageBackend`;
- public scheduler policy API;
- contributor platform items из раздела 2.2.

### 2.4 Уровни завершённости

#### Kernel-first v0.1

Проект имеет working runtime, TinyModel, contiguous/paged KV, scheduler, три заменяемых backend families, correctness/benchmark path и один законченный research result. Автор воспроизводит основные эксперименты из clean build.

#### Contributor-ready v0.2

Дополнительно появляются:

```text
manifest schema;
scaffolding;
out-of-tree examples;
polished tutorials;
comparison/report tooling;
external contributor dry run;
friction metrics и migration policy.
```

Отсутствие этих элементов не понижает шестимесячный релиз до alpha, если engine и research goals выполнены.

---

## 3. Пользователи и типы работы

### 3.1 Kernel/runtime researcher — основной пользователь v0.1

Хочет:

```text
реализовать CUDA operator или stage;
подключить его к runtime;
сравнить с reference и strong baseline;
проверить numerical и memory correctness;
увидеть end-to-end effect;
объяснить profiler metrics.
```

Именно под этот workflow оптимизируется шестимесячный roadmap.

### 3.2 Backend author

Реализует один из public extension points:

```text
RmsNormBackend
GemmBackend
AttentionStageBackend
```

Ему не требуется менять `ModelRunner`, scheduler или allocator ownership. Полный внешний onboarding не обещается до v0.2, но API не должен намеренно привязывать backend к internal types.

### 3.3 Research reproducer

Не меняет runtime core, а:

- запускает фиксированную benchmark matrix;
- сравнивает candidate и baseline;
- воспроизводит Nsight workflow;
- проверяет raw JSONL, environment и reproduction command;
- читает limitations и negative results.

### 3.4 Runtime contributor

Меняет:

```text
ModelRunner
memory planner
KV allocator
scheduler
public contracts
benchmark infrastructure
```

Такие изменения требуют ADR и отдельного correctness impact analysis.

### 3.5 Будущий external contributor

External contributor journey является целью post-v0.1. Архитектура не должна его блокировать, но сроки v0.1 не зависят от наличия scaffolding, polished docs или внешнего dry run.

---

## 4. Public extension surface

Не каждая часть runtime является plugin point. В v0.1 поддерживается минимальное число extension points, необходимое для kernel research.

### 4.1 Tier 1 — обязательные public extension points v0.1

```text
RmsNormBackend
GemmBackend
AttentionStageBackend
```

Для каждого Tier 1 interface обязательны:

```text
public header;
semantic specification;
reference или example implementation;
support query с reason;
correctness/conformance path;
benchmark path;
config-driven selection;
model-level A/B path, если interface входит в ModelRunner.
```

### 4.2 Internal primitives v0.1

Следующие операции остаются внутренними CUDA primitives:

```text
RoPE
activation / SwiGLU
KV append
embedding
argmax / logits helpers
masked softmax
```

Они имеют tests и benchmarks по необходимости, но не обязаны иметь registry, public backend API, отдельные canonical suites и contributor tutorials.

Это сознательно уменьшает platform scope и не блокирует fusion внутри `AttentionStageBackend`.

### 4.3 Experimental extension points после working model path

После M3/M4 могут появиться:

```text
MlpStageBackend
KvAppendBackend
NormLinearBackend
LogitsBackend
```

Они помечаются `experimental` и не входят в Final Definition of Done v0.1.

### 4.4 Не является extension point v0.1

```text
arbitrary model architecture
arbitrary computation graph
memory allocator ABI
scheduler policy ABI
binary plugin loader
network server
tokenizer
distributed execution
```

### 4.5 No-backend-specific-core-branches invariant

Выбор поддерживаемого backend-а происходит через один registry/config path.

`ModelRunner`, paged KV manager и scheduler не должны содержать ветки вида:

```cpp
if (backend_name == "my_attention_v3") { ... }
```

Если новый backend выявляет недостаток public contract, изменение API оформляется отдельно через ADR. Для v0.1 допускается source-level evolution до конца M4; стабильность API не должна тормозить paged attention design.

---

## 5. Backend API

### 5.1 Public/internal boundary

```text
include/minirt/backend_api/    // разрешённая зависимость backend-а
include/minirt/core/           // минимальные runtime value/view types
src/internal/                  // запрещённая зависимость backend-а
```

Public backend может использовать:

```text
DType
LayoutId
TensorView
MutableTensorView
WorkspaceView
PersistentWorkspaceView
ExecutionContextView
DeviceProperties
SupportResult
BackendDescriptor
problem structs
args structs
prepared plan base types
registry API
```

Backend не должен зависеть от:

```text
конкретного ModelRunner
scheduler queues
internal RequestState
BlockPool implementation
PreparedLayer internals
engine-owned STL containers
private arena implementation
```

### 5.2 Source-level versioning

До прохождения contiguous/paged attention parity используется:

```cpp
inline constexpr std::uint32_t kBackendApiVersionCandidate = 1;
```

API считается candidate и может меняться через ADR.

Source-level `Backend API v1` замораживается **после M4**, когда public attention views прошли:

```text
contiguous prefill/decode;
paged decode;
multi-request cases;
GQA;
partial final page;
workspace и lifetime validation.
```

Stable binary ABI не обещается.

### 5.3 Общий backend lifecycle

Public API не передаёт backend-у allocator implementation.

```cpp
struct SupportResult {
    bool supported;
    std::string reason;
};

struct ResourceRequirements {
    std::size_t persistent_bytes;
    std::size_t persistent_alignment;
    std::size_t workspace_bytes;
    std::size_t workspace_alignment;
};

class ExampleBackend {
public:
    virtual ~ExampleBackend() = default;

    virtual SupportResult supports(
        const Problem& problem,
        const DeviceProperties& device) const = 0;

    virtual ResourceRequirements requirements(
        const Problem& problem,
        const DeviceProperties& device) const = 0;

    virtual Result<std::unique_ptr<PreparedPlan>> prepare(
        const Problem& problem,
        const DeviceProperties& device,
        PersistentWorkspaceView persistent_storage) = 0;

    virtual Status run(
        const PreparedPlan& plan,
        const Args& args,
        WorkspaceView workspace,
        cudaStream_t stream) = 0;
};
```

Engine:

1. запрашивает requirements;
2. выделяет memory через internal arena;
3. передаёт backend-у ограниченные views;
4. сохраняет ownership внутри runtime.

Обязательные свойства:

- `supports()` возвращает причину;
- `prepare()` находится вне steady-state hot path;
- `run()` не выполняет device/host allocation;
- `run()` не выполняет global synchronization;
- все launches происходят в переданном stream;
- inputs не изменяются без mutable contract;
- preprocessing и execution time измеряются отдельно;
- plan lifetime и thread-safety задокументированы.

### 5.4 Attention-specific public views

Attention backend не получает `BlockPool` или internal request objects.

Public args содержат immutable или explicitly mutable views:

```text
ContiguousKvView
PagedKvView
SequenceMetadataView
BlockTableView
SlotMappingView
```

`PagedKvView` описывает минимум:

```text
K/V base pointers
logical dtype
physical layout ID
layer-local strides
block size in tokens
number of physical blocks
block table pointer and stride
sequence/context lengths
alignment guarantees
```

Ownership и allocation остаются у engine.

---

## 6. Backend registry и discovery

### 6.1 Registry model v0.1

Используется static source-level registration. Динамическая загрузка `.so` не входит в mandatory scope.

Минимальный backend directory:

```text
src/backends/<kind>/<name>/
├── backend.h
├── backend.cu
├── register.cc
├── CMakeLists.txt
└── README.md
```

Central `if/else` factory chains запрещены. Допускается одна декларативная CMake registration entry.

### 6.2 Backend descriptor

```cpp
struct BackendDescriptor {
    std::string_view name;
    std::string_view kind;
    std::string_view implementation_version;
    std::uint32_t backend_api_version;
    BackendStability stability;
    BackendFactory factory;
};
```

Runtime всегда вызывает `supports()` для конкретного problem; descriptor не заменяет capability check.

### 6.3 Mandatory discovery CLI

```bash
minirt backends list
minirt conformance --backend <name> --suite <suite-id>
minirt bench --backend <name> --suite <suite-id>
minirt model run --config <config>
minirt workload run --config <config> --trace <trace>
```

`minirt backends describe`, `suites list`, отдельный `compare` subcommand и matrix UI являются post-v0.1 polish.

### 6.4 Manifest policy

`backend.json` и отдельная manifest schema не обязательны для v0.1. Capability metadata может жить в `BackendDescriptor` и code-owned config.

Manifest вводится после того, как:

- появились реальные внешние backend packages;
- descriptor перестал быть достаточным;
- определены стабильные capability names;
- есть время поддерживать schema/version migration.

---

## 7. Kernel Experimentation Kit

Цель этого слоя — не построить отдельный contributor product, а обеспечить честный цикл:

```text
correctness → benchmark → profiler → integration → interpretation
```

### 7.1 Minimal Conformance Runner

Команда:

```bash
minirt conformance \
    --backend <name> \
    --suite <suite-id>
```

Result classifications:

```text
PASS
FAIL
UNSUPPORTED
SKIPPED
INFRA_ERROR
```

Общие checks:

- semantic correctness;
- declared support совпадает с поведением;
- boundary и non-multiple shapes;
- invalid input rejection;
- workspace validation;
- repeated execution;
- non-default stream;
- no device allocation в `run()`;
- input immutability;
- output bounds/canaries для selected cases;
- Compute Sanitizer-compatible smoke cases.

Attention checks добавляются по мере реализации:

```text
causal semantics
GQA
short/long contexts
contiguous/paged parity
page boundary
partial final page
multiple requests
request isolation
```

Suite versioning используется только для опубликованных experiment matrices. Не требуется заранее создавать suite family для каждой потенциальной операции.

### 7.2 Benchmark Runner

Benchmark runner фиксирует:

```text
problem matrix
backend/config
baseline
warmup
samples
launches per sample
GPU-event time
wall time
correctness tolerances
environment
raw JSONL
reproduction command
```

Обязательные v0.1 suites создаются по milestone-ам:

```text
rmsnorm-smoke-v1
rmsnorm-target-v1
small-m-gemm-v1
decode-attention-v1
paged-decode-v1
decoder-block-v1
scheduler-mixed-length-v1
```

Полный universal compare/report framework не требуется. Достаточно Python-скрипта, который строит таблицу candidate vs baseline из JSONL.

### 7.3 Synthetic fixtures

Repository предоставляет:

```text
deterministic tensor generator
fixed seeds
boundary-value fixtures
contiguous KV fixtures
paged KV/block-table fixtures
request-isolation fixtures
```

Operator-level experiments не требуют скачивания checkpoint.

### 7.4 Tiny generated model

`TinyModelSpec` использует ту же architecture semantics, что target model:

- 2 layers;
- GQA, если target architecture использует GQA;
- та же RoPE semantics;
- тот же activation kind;
- deterministic generated weights;
- малые dimensions и vocab;
- полный prefill/decode path без external download.

### 7.5 Example backends v0.1

Обязательно:

```text
examples/backends/rmsnorm_naive/
examples/backends/decode_attention_naive/
```

GEMM implementations могут жить как built-in backends в `src/backends/gemm`.

Example backend должен быть небольшим, понятным и документировать ownership, stream, workspace и unsupported cases.

### 7.6 Post-v0.1 platform items

Переносятся:

```text
scaffolding generator
manifest validation tool
out-of-tree package template
полные contributor tutorials
external dry run
UX time targets
automatic Markdown/HTML reports
plots generation
community PR checklist automation
```

---

## 8. Решения, замораживаемые до kernel work

До завершения первой недели создать четыре ADR:

```text
docs/adr/0001-target-model.md
docs/adr/0002-target-gpu-and-toolchain.md
docs/adr/0003-dtype-numerics-and-layouts.md
docs/adr/0004-backend-api-boundary-and-registration.md
```

Остальные ADR создаются только при реальном архитектурном decision point.

### 8.1 ArchitectureContract

Формулировка «Qwen/Llama-like» не является contract. Выбирается одна точная архитектура и одна synthetic-конфигурация той же семантики.

`ArchitectureContract` содержит минимум:

```text
architecture_name
reference_checkpoint и revision/hash, если используется
vocab_size
hidden_size
intermediate_size
num_layers
num_query_heads
num_kv_heads
head_dim
max_position_embeddings
rms_norm_epsilon
norm placement
activation kind
RoPE variant/theta/scaling
bias flags
tied embeddings
attention scale
BOS/EOS IDs
weight/activation/accumulation dtype
```

Инварианты:

```text
hidden_size == num_query_heads * head_dim
num_query_heads % num_kv_heads == 0
weight names и orientations документированы
position semantics одинаковы в Python и C++
GQA является общим случаем; MHA — частный случай
```

### 8.2 Target GPU contract

Зафиксировать:

```text
GPU model
SM architecture
VRAM
driver
CUDA toolkit
host compiler
CMake
Linux distribution
CMAKE_CUDA_ARCHITECTURES
```

Основные performance claims делаются только для target GPU.

### 8.3 Dtype и numerics policy

```text
reference tensors       — FP32/FP64 в Python
weights/activations     — один selected low-precision dtype
GEMM accumulation      — FP32, где возможно
RMSNorm accumulation   — FP32
softmax accumulation   — FP32
logits comparison      — в FP32
TF32                    — выключен или явно записан
fast math               — выключен или явно записан
quantization            — вне scope
```

Второй low-precision dtype — stretch goal.

### 8.4 Named layout contract

Первая версия поддерживает небольшой набор именованных layouts:

```text
HiddenRowMajor
QkvTokenHeadDim
ContiguousKvLayerHeadTokenDim
PagedKvBlockHeadTokenDim
LinearWeightOutIn
BlockTableRequestLogicalBlock
PackedTokenMetadata
```

Backend не получает произвольную stride system без необходимости. Unsupported layout возвращает `SupportResult{false, reason}`.

### 8.5 API evolution policy

До конца M4:

- public API имеет status `candidate`;
- breaking change требует ADR;
- example backend обновляется в том же commit;
- conformance cases обновляются;
- core refactor не должен без причины протекать в backend API.

После paged/contiguous parity API получает source-level version 1. Migration docs и manifest versioning остаются post-v0.1.

---

## 9. Основные инженерные принципы

1. **Engine и kernel research — главный продукт v0.1.** Platform polish не блокирует attention, KV-cache, model path или scheduler.
2. **Contributor-friendly API — архитектурное ограничение, а не отдельный шестимесячный продукт.**
3. **Correctness before performance.** Оптимизация начинается после correctness/conformance pass.
4. **Early vertical slices.** Сначала operator, затем attention stage, Transformer block, paged path и scheduler.
5. **Одна гипотеза за эксперимент.** Несколько оптимизаций не объединяются в один performance commit.
6. **PyTorch — reference, а не runtime dependency.**
7. **Явный `cudaStream_t`.** Backend не выполняет скрытую global synchronization.
8. **No allocation in hot path.** После prepare/warmup внутри `run`, model step и scheduler iteration нет device allocation.
9. **Воспроизводимые измерения.** JSONL связан с environment, config и git commit.
10. **Profiler не заменяет benchmark.** Финальные latency numbers получаются отдельным run.
11. **Semantic stages не должны блокировать fusion.** `AttentionStageBackend` может выполнять несколько kernels или fused kernel.
12. **Минимальное число public extension points.** Internal primitive не становится plugin point без use case.
13. **Unsupported лучше silent fallback.** Fallback фиксируется явно.
14. **Synthetic first.** Основной correctness path не зависит от checkpoint.
15. **Public/internal boundary проверяется сборкой.**
16. **Один законченный vertical slice лучше трёх недоделанных subsystems.**
17. **Каждый research issue заканчивается интерпретацией.** Код без вывода не считается research result.
18. **Scope сокращается сначала за счёт platform polish и optional features, а не tests и measurements.**
19. **Hotspot сначала измеряется.** Deep research target выбирается после real-shape profile.
20. **Backend-specific core branches запрещены.**
21. **Backend API замораживается после paged attention parity, а не раньше.**

---

## 10. Runtime architecture

### 10.1 Слои системы

```text
Python reference/export layer
   ├── checkpoint loader/exporter
   ├── reference operators
   ├── deterministic fixture generator
   └── result analysis

Experimentation layer
   ├── public backend API
   ├── static BackendRegistry
   ├── conformance runner
   ├── benchmark runner
   └── JSONL/environment output

C++ engine layer
   ├── request API accepting token IDs
   ├── scheduler
   ├── BatchPlan builder
   └── ModelRunner

Model execution layer
   ├── PreparedModel
   ├── PreparedLayer
   ├── AttentionStageBackend
   ├── internal MLP path
   └── primitive kernels/backends

Memory layer
   ├── persistent model storage
   ├── per-request KV storage
   ├── transient step arena
   └── host/device metadata buffers

CUDA runtime layer
   ├── streams and events
   ├── cuBLASLt handle
   ├── workspace
   ├── launchers
   ├── NVTX
   └── error handling
```

Manifest tooling, scaffolding, external package templates и report UI не являются отдельным обязательным слоем v0.1.

### 10.2 Primitive и composite backends

Public v0.1:

```text
GemmBackend
RmsNormBackend
AttentionStageBackend
```

Internal primitives:

```text
RoPE
Activation/SwiGLU
KV append
Embedding
Masked softmax
Argmax/logits helpers
```

Reference `AttentionStageBackend` может выполнять:

```text
RMSNorm → QKV GEMM → RoPE → KV write → attention → output GEMM
```

Optimized backend может fusion-ить:

```text
RoPE + KV write
RoPE + KV write + decode attention
Norm + QKV projection
```

`ModelRunner` вызывает semantic stage и не предполагает фиксированное число launches.

### 10.3 Memory lifetimes

#### Persistent model storage

```text
weights
prepacked weights
backend plans
constant tables
optional RoPE tables
cuBLASLt algorithm state
```

#### Persistent request storage

```text
KV-cache blocks
block tables
sequence state
generated token buffers
request metadata
```

#### Per-step transient storage

```text
hidden activations
QKV outputs
attention workspace
MLP intermediate
logits workspace
packed batch metadata
temporary reductions
```

Обязательные components:

```text
PersistentArena
StepArena или ScratchPlanner
WorkspaceView
allocation counters
peak memory accounting
```

Hot-path invariants:

```text
cuda allocations during steady-state prefill/decode = 0
host allocations inside per-layer execution          = 0
hidden global synchronizations                       = 0
```

### 10.4 Execution phases

```text
Configure
  - загрузить ArchitectureContract;
  - проверить target GPU;
  - выбрать backends по config;
  - проверить Backend API version.

Prepare
  - загрузить/generated weights;
  - создать prepared plans;
  - выделить persistent/transient storage;
  - выбрать cuBLASLt algorithms;
  - выполнить warmup.

Execute
  - scheduler создаёт BatchPlan;
  - ModelRunner вызывает selected semantic backends;
  - request и KV metadata обновляются;
  - device allocations отсутствуют.

Validate
  - runner создаёт problems и fixtures;
  - сравнивает с reference;
  - сохраняет result и unsupported reasons.

Measure
  - benchmark runner использует фиксированную suite/config;
  - profiler mode запускается отдельно;
  - result bundle содержит environment и reproduction command.
```

---

## 11. Структура repository

Директории создаются **по мере появления работающего vertical slice**, а не заранее пустыми.

```text
minirt/
├── LICENSE
├── README.md
├── plan.md
├── CONTRIBUTING.md
├── CMakeLists.txt
├── CMakePresets.json
├── configs/
│   ├── model/
│   ├── backend/
│   └── workload/
├── docs/
│   ├── adr/
│   ├── architecture/
│   ├── numerics/
│   ├── benchmark-methodology.md
│   └── research/
├── include/minirt/
│   ├── backend_api/
│   └── core/
├── src/
│   ├── internal/
│   │   ├── runtime/
│   │   ├── memory/
│   │   ├── model/
│   │   ├── kv_cache/
│   │   └── scheduler/
│   ├── kernels/
│   ├── backends/
│   │   ├── rmsnorm/
│   │   ├── gemm/
│   │   └── attention/
│   ├── registry/
│   └── cli/
├── examples/backends/
│   ├── rmsnorm_naive/
│   └── decode_attention_naive/
├── bindings/torch/
├── python/
│   ├── references/
│   ├── model_export/
│   ├── fixtures/
│   └── analysis/
├── suites/
│   ├── conformance/
│   └── benchmarks/
├── tests/
│   ├── unit/
│   ├── correctness/
│   ├── composition/
│   ├── integration/
│   ├── scheduler/
│   ├── api_boundary/
│   └── sanitizer/
├── benchmarks/
├── workloads/
├── scripts/
├── results/
└── reports/
```

Post-v0.1 directories/tools:

```text
tools/new_backend.py
manifests/
out_of_tree_examples/
docs/contributors/tutorials/
report_ui/
```

### 11.1 API boundary enforcement

- backend targets получают include path только к `include/minirt`;
- include из `src/internal` является compile error;
- example backend собирается отдельным target;
- public headers проходят standalone include test;
- public API не экспортирует internal ownership/container types.

---

## 12. Correctness и Conformance Kit

### 12.1 Назначение

Conformance отвечает на вопрос:

> Соблюдает ли backend semantic, memory, stream и lifecycle contract своего extension point?

Conformance не доказывает высокую производительность и является gate перед publishable benchmark claim.

### 12.2 Suites v0.1

Suites создаются тогда, когда соответствующий path реализован:

```text
rmsnorm-conformance-v1
gemm-conformance-v1
attention-decode-conformance-v1
paged-decode-conformance-v1
```

Prefill и full-stage cases могут входить в attention suite или получить отдельный ID, если их semantics достаточно различаются.

Suite фиксирует:

```text
schema/suite version
operation или stage
semantic contract version
fixture generator version
shapes, dtypes, layouts
seeds
tolerances
required/optional/expected-unsupported cases
reference backend
memory и stream checks
```

Не требуется заранее создавать suite для каждого internal primitive.

### 12.3 Общие проверки backend-а

- descriptor и API version;
- `supports()` для valid и invalid problems;
- понятная причина unsupported case;
- requirements/workspace validation;
- insufficient workspace handling;
- repeated `prepare()` и `run()`;
- работа в non-default stream;
- отсутствие global synchronization в targeted trace;
- отсутствие device allocation в `run()`;
- отсутствие host allocation в per-launch hot path;
- отсутствие input mutation;
- output bounds/canaries для representative cases;
- alignment contracts;
- deterministic output, если заявлен;
- cleanup после неуспешного `prepare()`;
- Compute Sanitizer-compatible smoke cases.

### 12.4 RMSNorm cases

```text
rows = 1, 2, 8, 32, 128
hidden = TinyModel и target shapes
hidden не кратен vector width
малые/большие значения
near-zero variance
repeated runs
non-default stream
```

Metrics:

```text
max_abs_error
max_relative_error
relative_l2_error
cosine_similarity
NaN/Inf count
```

### 12.5 GEMM cases

Cases ограничены declared capabilities:

```text
M = 1, 2, 4, 8, 32, 128, 512
real N/K
boundary N/K
selected layout/op combination
workspace limit
Identity epilogue
unsupported configuration
```

### 12.6 Attention cases

Contiguous prefill/decode:

```text
prompt/context length 1
short и long context
causal mask boundary
GQA mapping
KV population
first decode after prefill
several sequential steps
```

Paged decode:

```text
first page
within-page append
page boundary
partial last page
multiple pages
multiple requests
mixed sequence lengths
release/reuse sentinel fixture
contiguous/paged parity
```

### 12.7 Machine-readable result

```json
{
  "schema_version": 1,
  "backend": "warp_split_attention",
  "backend_version": "0.1.0",
  "backend_api_version": 1,
  "suite": "paged-decode-conformance-v1",
  "passed": 41,
  "failed": 0,
  "skipped": 3,
  "unsupported": [
    {
      "problem": "head_dim=96",
      "reason": "supported head dimensions: 64, 128"
    }
  ],
  "allocation_count_after_prepare": 0,
  "stream_order_probe_passed": true
}
```

### 12.8 Definition of Done

- [ ] Reference указан для каждого public extension point.
- [ ] Synthetic fixtures deterministic.
- [ ] Failures содержат seed и problem signature.
- [ ] Unsupported отличается от failure.
- [ ] Allocation и stream-order checks автоматизированы для representative cases.
- [ ] Machine-readable result сохраняется.
- [ ] CLI возвращает ненулевой exit code при mandatory failure.
- [ ] Example backends проходят соответствующие suites.

---

## 13. Benchmark platform

### 13.1 Назначение

Benchmark runner отвечает на вопрос:

> Как candidate backend ведёт себя на фиксированной problem matrix относительно strong baseline при одинаковой correctness и environment methodology?

Автор backend-а не пишет отдельный timing harness для каждого эксперимента.

### 13.2 Suites v0.1

```text
rmsnorm-target-v1
small-m-gemm-v1
attention-decode-v1
paged-decode-v1
decoder-block-v1
scheduler-mixed-length-v1
```

Suite фиксирует:

```text
suite/version
required conformance
problem matrix
warmup policy
samples
launches per sample
measurement mode
baselines
metrics
GPU-state requirements
```

### 13.3 Publishable benchmark gates

- required correctness/conformance прошёл;
- environment manifest сохранён;
- git commit и dirty state сохранены;
- candidate и baseline используют одинаковые inputs/seeds;
- preprocessing отделён от steady-state;
- unsupported problems перечислены;
- raw JSONL сохранён;
- profiler run не используется как final timing.

### 13.4 CLI behavior

```bash
minirt bench \
  --suite attention-decode-v1 \
  --backend warp_split_attention \
  --baseline naive_attention \
  --output results/raw/exp-001
```

Runner:

1. проверяет descriptor и support;
2. запускает/проверяет smoke correctness;
3. загружает suite;
4. генерирует fixtures;
5. выполняет warmup;
6. измеряет candidate и baselines;
7. сохраняет JSONL и environment;
8. печатает reproduction command.

Comparison выполняется минимальным Python-скриптом:

```bash
python python/analysis/compare_runs.py \
  results/raw/exp-001 \
  results/raw/exp-000
```

Полноценный `minirt compare`, plots и generated Markdown report — post-v0.1.

### 13.5 Common JSONL fields

```text
schema_version
experiment_id
timestamp_utc
suite_name/suite_version
benchmark_scope
operation/backend/backend_version
backend_api_version
backend_config
problem_signature
shape/layout/dtypes/compute_type
warmup/samples/launches_per_sample
measurement_mode/synchronization_policy
workspace/persistent/transient bytes
random_seed/input_hash/weights_hash
command_line/status/error_reason
GPU-event min/p50/p90/p99
wall min/p50/p90/p99
correctness metrics
environment and git metadata
```

### 13.6 Operation-specific metrics

GEMM:

```text
M/N/K
algorithm_id
tile_shape
achieved GFLOP/s
ratio to cuBLASLt
```

Memory-bound operations:

```text
estimated bytes read/written
effective bandwidth
vector width
```

Attention:

```text
phase
batch size
context/prompt length
query/KV heads
head_dim
block size
KV layout
```

Model/engine:

```text
prefill latency
TTFT
decode latency per token
input/output tokens per second
peak memory
kernel launches
allocation count after warmup
scheduler latency distributions
KV utilization
```

### 13.7 Run directory

```text
results/raw/<experiment_id>/
├── results.jsonl
├── conformance.json
├── environment.json
├── config.json
├── command.txt
├── stdout.log
└── stderr.log
```

Processed tables и plots могут генерироваться локально, но не являются release gate v0.1.

---

## 14. Reference fixtures и tiny model

### 14.1 Synthetic fixtures

Каждый реализованный public extension point имеет deterministic fixture generator.

Fixture identity:

```text
fixture_generator_version
operation
problem signature
seed
input hash
reference output hash
```

Fixtures генерируются локально и не требуют больших binary artifacts в repository.

### 14.2 TinyModelSpec

Tiny model сохраняет semantics target architecture, но использует малые dimensions:

```text
2–4 layers
hidden_size <= 256
малое число query/KV heads
head_dim, поддерживаемый target attention API
малый vocab
max sequence length <= 256
та же norm/RoPE/activation semantics
```

Точные параметры замораживаются в ADR и versioned config.

Tiny model используется для:

- end-to-end correctness;
- model-level backend A/B comparison;
- CI smoke tests;
- contributor quickstart;
- scheduler simulator integration;
- reproduction без model license/download.

### 14.3 Real checkpoint path

```text
source checkpoint
→ Python loader
→ architecture validation
→ tensor rename / transpose / reorder
→ dtype conversion
→ model.json + weights.bin
→ hash manifest
→ C++ inspect tool
→ reference parity test
```

Export format:

```text
format_version
endianness
alignment
ArchitectureContract
tensor name
dtype
shape
byte offset
byte length
logical layout
weight orientation
checksum
```

Real checkpoint является обязательным только для final real-shape benchmark, если он укладывается в scope. Tiny model остаётся mandatory fallback.

---

## 15. Roadmap overview

| Milestone | Недели | Главный engine result | Минимальная extension/measurement infrastructure |
|---|---:|---|---|
| M1 | 1–4 | Runtime core, RMSNorm, cuBLASLt и FP32 GEMM lab | Public boundary, static registry, RMSNorm conformance/benchmark, JSONL |
| M2 | 5–8 | RoPE, baseline prefill, contiguous KV, decode attention | Provisional `AttentionStageBackend`, attention correctness/benchmark |
| M3 | 9–12 | Transformer block, TinyModel, static memory, real-shape profile | Backend API candidate, model-level A/B path |
| M4 | 13–16 | Paged KV и paged decode attention | Paged view through same API; API v1 freeze after parity |
| M5 | 17–20 | Scheduler и continuous batching | Compatible backends participate in engine workloads without special branches |
| M6 | 21–24 | One deep study, final report и portfolio release | Reproducible commands and concise backend documentation; external dry run is stretch |

---

## 16. Milestone 1 — Runtime Core, RMSNorm & GEMM Laboratory

**Срок:** недели 1–4  
**Бюджет:** 40 часов

### 16.1 Цель

Построить минимальную платформу, в которой один CUDA operator проходит полный цикл:

```text
reference → backend → correctness → benchmark → profiler → JSONL
```

Первый месяц не обязан предоставлять внешний contributor journey.

### 16.2 Engine scope

- C++17 + CUDA CMake project;
- Debug, Release и profiling presets;
- CUDA/cuBLASLt error handling;
- move-only `CudaStream`, `CudaEvent`, `DeviceBuffer`;
- limited contiguous `TensorView`;
- `ExecutionContext` с одним stream и cuBLASLt handle;
- NVTX;
- allocation counters;
- benchmark timing skeleton;
- custom RMSNorm FP32;
- selected low-precision cuBLASLt linear;
- timeboxed FP32 GEMM laboratory:

```text
NaiveCudaGemm
TiledCudaGemmV0
CublasLtGemm baseline
```

Custom GEMM budget: **не более 8 часов**.

### 16.3 Minimal extension/measurement scope

- public/internal include boundary;
- `BackendDescriptor` и candidate API version;
- static `BackendRegistry`;
- `minirt backends list`;
- `minirt conformance` для RMSNorm;
- `minirt bench` для RMSNorm/GEMM;
- `naive_rmsnorm` example backend;
- deterministic RMSNorm fixtures;
- JSONL и environment manifest;
- compile test preventing internal includes.

Не делать в M1:

```text
manifest schema
scaffolding
out-of-tree package
full compare command
polished report generator
RoPE public extension point
attention API
```

### 16.4 RMSNorm contract

```text
input  [rows, hidden_size]
weight [hidden_size]
output [rows, hidden_size]
FP32 first implementation
FP32 accumulation
explicit epsilon
explicit stream
no input mutation
```

Selected low-precision custom RMSNorm переносится в M2/M3, если FP32 path завершён.

### 16.5 Benchmark matrix

RMSNorm:

```text
rows   = 1, 2, 8, 32, 128, 512
hidden = TinyModel/target shapes
boundary hidden not divisible by vector width
```

GEMM:

```text
M = 1, 2, 4, 8, 32, 128, 512
N/K = несколько target linear shapes
boundary dimensions
```

### 16.6 Definition of Done

- [ ] Четыре обязательных ADR созданы.
- [ ] Repository собирается одной командой.
- [ ] Runtime wrappers проходят smoke tests.
- [ ] Public/internal headers разделены.
- [ ] Registry обнаруживает RMSNorm backend.
- [ ] RMSNorm проходит numerical correctness и non-default-stream test.
- [ ] One-command RMSNorm benchmark создаёт JSONL.
- [ ] cuBLASLt backend работает на selected low-precision dtype.
- [ ] Naive и один tiled FP32 GEMM корректны на ограниченной matrix.
- [ ] Есть один Nsight Systems trace и один targeted Nsight Compute report.
- [ ] Опубликован короткий M1 report с negative results и next step.

### 16.7 Stop condition

Если RMSNorm vertical slice не завершён к концу недели 4, M2 начинается с его закрытия. Attention не блокируется отсутствием manifest, scaffolding, report UI или external contributor workflow.

### 16.8 Stretch goals

- warp-shuffle RMSNorm;
- fused residual + RMSNorm;
- second tiled GEMM configuration;
- selected low-precision RMSNorm.

---

## 17. Milestone 2 — Baseline Prefill, Contiguous KV & Decode Attention

**Срок:** недели 5–8  
**Бюджет:** 40 часов

### 17.1 Цель

Построить первую LLM-specific execution stage:

```text
Q/K/V inputs
→ RoPE
→ contiguous KV population/append
→ prefill baseline
→ single-token decode attention
→ output
```

Полный Transformer block переносится в M3.

### 17.2 Scope

- internal RoPE kernel;
- selected low-precision support для core path;
- documented contiguous KV layout;
- prompt KV population;
- single-token KV append;
- baseline causal prefill attention;
- naive contiguous decode attention;
- GQA mapping как основной contract;
- variable context length;
- PyTorch references и tensor dumps;
- several sequential decode steps в isolated attention stage.

Допустимый baseline prefill:

```text
QKᵀ                — cuBLASLt/simple batched baseline
scale + causal mask — custom CUDA
stable softmax      — custom CUDA, FP32 accumulation
softmax × V         — cuBLASLt/simple baseline
```

Materialized score matrix допустима.

### 17.3 Provisional AttentionStageBackend

API поддерживает отдельные methods/arguments для:

```text
prefill
decode
contiguous KV view
GQA metadata
workspace
explicit stream
```

API имеет status `candidate`; его нельзя замораживать до paged path.

### 17.4 Correctness matrix

```text
position 0 и boundary positions
prompt length 1/short/medium
context length 1/128/512/2048, если feasible
GQA mapping
causal mask boundary
stable softmax
KV contents after prefill
single-token append
first decode after prefill
several sequential decode steps
```

### 17.5 Definition of Done

- [ ] RoPE совпадает с reference.
- [ ] Baseline prefill совпадает с reference.
- [ ] Contiguous KV population и append корректны.
- [ ] Decode attention совпадает с reference.
- [ ] GQA является основным semantic contract.
- [ ] Several sequential decode steps проходят parity.
- [ ] Attention backend выбирается через registry/config.
- [ ] Замена attention backend не требует backend-specific branch в caller-е.
- [ ] Есть attention conformance/benchmark suites на synthetic fixtures.
- [ ] Hot path не выполняет device allocation.
- [ ] Есть Nsight Systems trace prefill + decode.
- [ ] Опубликован M2 report.

### 17.6 Stop condition

M3 model integration не начинается без:

```text
prefill correctness
contiguous KV correctness
several decode steps
GQA case
zero device allocations in attention hot path
```

### 17.7 Stretch goals

- fused RoPE + contiguous KV write;
- second decode kernel variant;
- custom optimized prefill prototype;
- selected low-precision RMSNorm.

---

## 18. Milestone 3 — Transformer Block, TinyModel & Runtime Hardening

**Срок:** недели 9–12  
**Бюджет:** 40 часов

### 18.1 Цель

Собрать end-to-end model path поверх уже корректных operators/stages:

```text
generated weights
→ embedding
→ one Transformer block
→ 2-layer TinyModel
→ logits
→ greedy decode steps
```

Одновременно завершить memory planning и получить real-shape profile. Backend API остаётся candidate.

### 18.2 Weight and model scope

- `TinyModelSpec` generator;
- versioned `model.json + weights.bin` или эквивалентный простой format;
- C++ loader без PyTorch runtime dependency;
- `ModelConfig`, `LayerWeights`, `PreparedModel`;
- embedding;
- final norm;
- LM head;
- weight orientation/shape validation;
- deterministic intermediate tensor dumps.

### 18.3 Transformer block

```text
input
 → RMSNorm
 → QKV projection
 → RoPE
 → contiguous KV write/read
 → attention
 → output projection
 → residual
 → RMSNorm
 → gate/up projections
 → activation/SwiGLU
 → down projection
 → residual
```

Model GEMMs используют cuBLASLt. Custom GEMM не блокирует model path.

### 18.4 Prepared lifecycle и static memory

- `supports()`;
- resource requirements;
- `prepare()` и prepared plans;
- `PersistentArena`;
- `StepArena`/`ScratchPlanner`;
- buffer reuse across layers;
- allocation audit;
- no `cudaMalloc/cudaFree` after warmup;
- explicit fallback reporting;
- cleanup tests.

### 18.5 Real-shape baseline

- target operator shapes;
- one-block prefill/decode;
- 2-layer TinyModel;
- per-layer timeline;
- launch count;
- persistent/transient/KV memory;
- top hotspots for prefill/decode;
- research-track candidate list.

Full real checkpoint — stretch goal.

### 18.6 Compute Sanitizer

Representative matrix:

```text
regular shape
boundary/non-multiple shape
single-token decode
longer context
GQA
repeated execution
```

### 18.7 Definition of Done

- [ ] TinyModel генерируется без external checkpoint.
- [ ] C++ loader не зависит от PyTorch.
- [ ] Один Transformer block проходит intermediate parity.
- [ ] 2-layer TinyModel выдаёт корректные logits/top-1 на fixed cases.
- [ ] Several greedy decode steps работают.
- [ ] Model path использует config-selected backends.
- [ ] Steady-state device allocations равны нулю.
- [ ] Compute Sanitizer representative suite проходит.
- [ ] Есть real-shape Nsight profile и top-hotspot table.
- [ ] Выбран основной research track candidate.
- [ ] Backend API остаётся `candidate`, limitations записаны.
- [ ] Опубликован M3 report.

### 18.8 Stretch goals

- 4-layer TinyModel;
- full checkpoint;
- selected weight prepacking;
- model-level backend autotuning;
- CUDA Graph feasibility note.

---

## 19. Milestone 4 — Paged KV Cache & Backend API v1

**Срок:** недели 13–16  
**Бюджет:** 40 часов

### 19.1 Цель

Перейти к block-based KV memory management и подключить paged decode через тот же `AttentionStageBackend`/registry path, что contiguous implementation.

API v1 замораживается только после paged/contiguous parity.

### 19.2 Allocation semantics

Первая версия использует model-level block identity:

```text
one logical block ID
→ K/V slot layer 0
→ K/V slot layer 1
→ ...
→ K/V slot layer N-1
```

Model-level block bytes:

```text
num_layers × 2 × num_kv_heads × block_size_tokens × head_dim × bytes_per_element
```

### 19.3 Host-side KV manager

- `BlockPool`;
- free list;
- `BlockHandle {index, generation}`;
- request-local logical block table;
- allocate/append/release;
- logical token → block + offset;
- OOM status и rollback;
- high-water mark;
- fragmentation counters;
- deterministic reuse in tests.

### 19.4 Metadata и device storage

```text
block_table[request, logical_block]
sequence_lengths[request]
context_lengths[request]
last_block_fill[request]
slot_mapping[token]
request_ids[batch_slot]
```

Host metadata зеркалируется в preallocated device buffers.

### 19.5 Invariants

- free block не принадлежит request;
- occupied block отсутствует во free list;
- independent requests не разделяют block;
- generation обнаруживает stale handle;
- release возвращает все blocks;
- logical position отображается однозначно;
- partial last block учитывается;
- OOM выполняет rollback;
- release одного request не меняет данные другого.

### 19.6 Paged attention backend

Mandatory reference implementation:

- читает K/V через block table;
- поддерживает multiple requests и mixed context lengths;
- поддерживает GQA;
- обрабатывает partial final page;
- совпадает с contiguous backend;
- не читает free/foreign blocks;
- не аллоцирует в `run_decode()`.

Optimized paged kernel не обязателен, если не выбран как M6 research track.

### 19.7 API v1 freeze gate

После прохождения:

```text
contiguous prefill/decode
paged decode
multi-request isolation
GQA
page boundary
partial final page
workspace/lifetime checks
```

фиксируются:

- source-level Backend API version 1;
- `PagedKvView` semantics;
- resource requirements contract;
- registry/config selection path.

Manifest, out-of-tree package и migration tooling не обязательны.

### 19.8 Benchmark matrix

```text
block_size_tokens = 8, 16, 32, 64
context_length    = 16, 128, 512, 2048, max feasible
active_requests   = 1, 2, 4, 8
heads/dtype       = TargetModelSpec
```

Metrics:

```text
useful/allocated token slots
fragmentation
metadata/block-table/KV bytes
allocation operations
high-water mark
decode latency
effective bandwidth
page-table overhead
OOM events
```

### 19.9 Definition of Done

- [ ] BlockPool unit и randomized tests проходят.
- [ ] Generation IDs и OOM rollback корректны.
- [ ] Paged KV contents совпадают с contiguous.
- [ ] Paged decode совпадает с contiguous.
- [ ] Multi-request isolation доказана sentinel tests.
- [ ] Paged backend выбирается через обычный registry/config path.
- [ ] `ModelRunner` не содержит backend-name branches.
- [ ] Public API не раскрывает allocator ownership internals.
- [ ] Block-size benchmark воспроизводим.
- [ ] Compute Sanitizer включает page-boundary cases.
- [ ] Backend API v1 заморожен после parity gate.
- [ ] Опубликован M4 report.

### 19.10 Stop condition

Scheduler GPU integration не начинается, пока paged/contiguous parity не проходит mandatory correctness matrix.

### 19.11 Stretch goals

- optimized paged attention variant;
- alternative physical KV layout;
- vectorized block-table loads;
- prefix block sharing;
- copy-on-write;
- out-of-tree paged backend example.

---

## 20. Milestone 5 — Scheduler & Continuous Batching

**Срок:** недели 17–20  
**Бюджет:** 40 часов

### 20.1 Цель

Построить минимальный inference engine для нескольких requests. Compatible attention/GEMM/RMSNorm backends должны участвовать в engine workloads без специальных веток.

Mandatory semantics:

> Новые requests допускаются, а завершённые удаляются между model iterations. Decode batch динамически меняется.

Mixed prefill/decode в одном fused GPU batch не обязателен.

### 20.2 Request lifecycle

```text
Waiting
Admitted
Prefilling
Decoding
Finished
Failed
Rejected
```

`RequestState` содержит IDs, arrival time, token buffers, lengths, status, EOS, KV handles, timestamps и error status.

### 20.3 Scheduler architecture

```text
RequestQueue
   ↓
Scheduler::step()
   ↓
BatchPlan
   ↓
BatchBuilder
   ↓
ModelRunner
   ↓
ExecutionResult
   ↓
RequestState update
```

`BatchPlan` содержит request IDs, token/position IDs, sequence/context lengths, slot mapping, block tables, output slots и required KV blocks.

### 20.4 Token-budget limits

```text
max_active_requests
max_batched_tokens
max_prefill_tokens_per_step
max_decode_requests_per_step
max_sequence_length
max_total_kv_blocks
```

Admission учитывает request и KV capacity.

### 20.5 Baseline scheduling policy

```text
1. завершить bookkeeping previous iteration;
2. освободить KV finished/failed requests;
3. сформировать decode batch;
4. при оставшемся budget допустить prefill requests;
5. выполнить prefill/decode отдельными sub-batches;
6. обновить state и timestamps.
```

### 20.6 Required workload traces

Mandatory:

```text
uniform short
mixed prompt/output lengths
KV-pressure
```

Stretch:

```text
bursty arrivals
steady arrivals
high-concurrency throughput
priority/fairness cases
```

### 20.7 Engine-level backend evaluation

Promise v0.1:

> Backend, прошедший public correctness и объявивший compatible capabilities, запускается в model-step и scheduler workloads без изменений scheduler-а или `BatchBuilder`-а.

Добавить:

- `decoder-block-v1`;
- `scheduler-mixed-length-v1`;
- backend matrix runner;
- automatic capability filtering;
- report/table, связывающую microbenchmark с TTFT/TPOT/throughput.

### 20.8 Metrics

Per-request:

```text
queue time
TTFT
TPOT/inter-token latency
end-to-end latency
input/output tokens
finish reason
```

Aggregates:

```text
p50/p90/p99 latency distributions
input/output tokens/s
completed requests/s
accepted/rejected/failed counts
KV utilization/high-water mark
OOM events
GPU busy fraction estimate
```

### 20.9 Definition of Done

- [ ] Request transitions покрыты unit tests.
- [ ] Scheduler тестируется без GPU через fake executor.
- [ ] `BatchPlan` deterministic.
- [ ] Token budget и KV admission соблюдаются.
- [ ] Requests приходят в разное время.
- [ ] Decode batch меняется между iterations.
- [ ] EOS и MaxOutLen работают.
- [ ] Completed requests освобождают KV blocks.
- [ ] Backend failure не смешивает request state.
- [ ] Compatible backends участвуют в engine workloads без special branches.
- [ ] Выполнено static vs continuous comparison.
- [ ] Сохранены latency distributions и memory metrics.
- [ ] Опубликован M5 report.

### 20.10 Stretch goals

- chunked prefill;
- mixed prefill/decode packed execution;
- priorities/fairness;
- cancellation;
- CUDA Graph buckets;
- public scheduling policy interface.

---

## 21. Milestone 6 — Deep Research & Kernel-First v0.1 Release

**Срок:** недели 21–24  
**Бюджет:** 40 часов

### 21.1 Цель

Не добавлять новые крупные subsystems. Завершить одно глубокое CUDA-исследование поверх работающего engine и упаковать reproducible portfolio release.

Рекомендуемое распределение:

```text
28 часов — research implementation, benchmark, profiling, model/engine impact;
8 часов  — report, README, architecture diagrams, reproducibility;
4 часа  — bugfix/reserve.
```

External contributor dry run — stretch goal и не отнимает время у research.

### 21.2 Research tracks

Выбрать ровно один на основании M3/M4 profiling:

#### Track A — Optimized paged decode attention

```text
one-CTA-per-head vs split-sequence
GQA mapping
vectorized K/V loads
page-table overhead
online softmax
warp reductions
block-size interaction
```

#### Track B — Specialized small-M GEMM

```text
M = 1–8
exact target shapes
register/shared-memory tiling
selected Tensor Core experiment, если feasible
epilogue fusion
cuBLASLt baseline
```

#### Track C — Fused RoPE + KV write

```text
launch count
eliminated memory traffic
vectorized stores
contiguous vs paged
prefill vs decode
model-level impact
```

#### Track D — KV block-size trade-off

```text
fragmentation
metadata
allocator frequency
page locality
latency
maximum concurrency
scheduler throughput
```

#### Track E — CUDA Graphs

Только если shapes, pointers и metadata достаточно стабильны.

#### Track F — FP16/BF16

Только если target GPU и model path позволяют честное сравнение.

### 21.3 Research loop

```text
Problem statement
Related work notes
Hypothesis
Scope and non-goals
Reference implementation
Strong baseline
2–3 isolated variants
Correctness/conformance
Benchmark matrix
Nsight Systems
Nsight Compute
Operator result
Model/scheduler impact
Interpretation
Threats to validity
Limitations
Negative results
Reproduction command
```

### 21.4 Release packaging

Mandatory:

```text
README
architecture diagram
backend lifecycle diagram
memory/KV layout diagram
scheduler diagram
build instructions
supported model/GPU/CUDA assumptions
correctness methodology
benchmark methodology
reproducible commands
raw results
research report
known limitations
short demo
```

Not mandatory:

```text
external contributor tutorial set
out-of-tree guide
scaffolding
manifest docs
community governance
HTML report
external dry-run report
```

### 21.5 Definition of Done

- [ ] Один research question завершён.
- [ ] Есть strong baseline и isolated variants.
- [ ] Все variants проходят required correctness.
- [ ] Есть operator и model/engine-level results.
- [ ] Есть Nsight Systems/Compute evidence.
- [ ] Raw results и scripts сохранены.
- [ ] Главный experiment воспроизводится одной командой.
- [ ] Working engine demo воспроизводим из clean build.
- [ ] Limitations и unsupported scope описаны.
- [ ] Release `v0.1` или `v0.1-alpha` подготовлен по критериям раздела 28.

### 21.6 Stretch goals

- internal clean-room contributor exercise;
- один внешний dry run;
- friction fixes;
- polished tutorial;
- automatic report generation.

---

## 22. Недельный календарь

| Неделя | Главный deliverable |
|---:|---|
| 1 | Exact architecture/GPU/dtype/layout contracts, repository bootstrap |
| 2 | Runtime wrappers, `ExecutionContext`, JSONL benchmark skeleton, registry skeleton |
| 3 | RMSNorm backend, correctness, non-default-stream checks |
| 4 | cuBLASLt + timeboxed FP32 GEMM lab, first profiles, M1 report |
| 5 | RoPE, attention semantics, contiguous KV layout |
| 6 | Baseline prefill and causal softmax path |
| 7 | Contiguous KV population/append, GQA cases |
| 8 | Decode attention, sequential decode parity, M2 report |
| 9 | TinyModel generator, weight format/loader |
| 10 | One Transformer block and intermediate parity |
| 11 | 2-layer TinyModel, static memory planning |
| 12 | Real-shape profile, sanitizer, research candidate, M3 report |
| 13 | BlockPool, generation handles, OOM rollback |
| 14 | Device paged storage and metadata |
| 15 | Paged decode backend and multi-request isolation |
| 16 | Contiguous/paged parity, block-size benchmark, Backend API v1 freeze |
| 17 | Request lifecycle and fake scheduler |
| 18 | `BatchPlan`, token/KV budgets, deterministic traces |
| 19 | GPU integration and changing decode batches |
| 20 | Static vs continuous batching study, M5 report |
| 21 | Research baseline and first variant |
| 22 | Additional isolated variants and benchmark matrix |
| 23 | Nsight analysis and model/engine-level effect |
| 24 | Final report, reproducibility, demo and release |

### 22.1 Strict gates

```text
After week 4:
  RMSNorm correctness/benchmark path works;
  runtime core and JSONL harness exist.

After week 8:
  contiguous prefill/decode and GQA parity pass;
  attention backend is selectable without special caller branch.

After week 12:
  one Transformer block and 2-layer TinyModel work;
  steady-state allocations are zero;
  real hotspots are profiled.

After week 16:
  paged/contiguous parity passes;
  Backend API v1 can be frozen.

After week 20:
  continuous batching works on deterministic workloads.

After week 24:
  one deep research result and reproducible engine demo are complete.
```

Platform polish никогда не блокирует следующий engine milestone, если public boundary, registry, correctness и benchmark path уже достаточны для текущего backend-а.

---

## 23. Correctness и test strategy

Conformance проверяет public backend contract. Внутренние runtime tests дополняют его и проверяют composition, ownership и request lifecycle.

### 23.1 Уровни correctness

#### Level 1 — Operator

```text
RMSNorm
RoPE
GEMM wrapper
masked softmax
activation / SwiGLU
embedding
KV append
argmax
```

#### Level 2 — Composition

```text
RMSNorm + QKV projection
RoPE + KV write
QKᵀ + causal mask + softmax + V
attention + output projection + residual
RMSNorm + MLP + residual
prefill + first decode step
```

#### Level 3 — Layer

Проверяются:

- full decoder block prefill;
- full decoder block decode;
- intermediate tensors в debug mode;
- KV contents;
- residual paths;
- repeated decode steps.

#### Level 4 — Model

Проверяются:

- final hidden state;
- logits;
- top-1 agreement;
- top-k overlap;
- greedy tokens;
- divergence по нескольким steps;
- TinyModel;
- real checkpoint, если он вошёл в scope.

#### Level 5 — Engine

Проверяются:

- multiple requests;
- block ownership;
- scheduler transitions;
- arrivals и completions;
- EOS и MaxOutLen;
- static/continuous equivalence там, где ordering одинаков;
- backend replacement внутри scheduler workload.

### 23.2 Numerical metrics

```text
max_abs_error
max_relative_error
relative_l2_error
mean_abs_error
cosine_similarity
allclose_passed
atol
rtol
nan_count
inf_count
```

Для logits:

```text
top1_agreement
top5_overlap
top10_overlap
argmax_reference
argmax_runtime
reference_logit_margin
```

Tolerance хранится в versioned config. Его нельзя ослаблять без issue, failing fixture и объяснения причины.

### 23.3 Required shape classes

Каждый operator/conformance suite включает:

- минимальный valid shape;
- real-model shape;
- tile/vector boundary;
- non-multiple shape;
- maximum supported shape;
- invalid shape;
- unsupported dtype/layout;
- repeated execution;
- GQA case, если применимо;
- page boundary, если применимо.

### 23.4 Determinism

- default fixed seed;
- seed и fixture hash в каждом result;
- failing seed печатается;
- scheduler traces immutable после publication;
- TinyModel weights deterministic;
- nondeterministic backend явно помечается.

### 23.5 Memory-safety matrix

```text
prepare allocation failure
partial model-load failure
repeated engine create/destroy
repeated request allocate/release
block-pool randomized state machine
out-of-range position
block-table overflow
stale handle
insufficient workspace
misaligned buffer
buffer canary failure
steady-state allocation audit
```

### 23.6 Test entry points

```bash
./scripts/test.sh unit
./scripts/test.sh correctness
./scripts/test.sh integration
./scripts/test.sh scheduler
./scripts/conformance.sh smoke
./scripts/conformance.sh full
./scripts/sanitize.sh fast
./scripts/sanitize.sh milestone
```

---

## 24. Benchmark и profiling protocol

### 24.1 Benchmark rules

- warmup before measurement;
- CUDA-event и wall time сохраняются отдельно;
- preparation, data setup и steady-state разделены;
- several independent samples;
- several launches per sample для short kernels;
- p50 и p90 обязательны;
- p99 только при достаточном числе samples;
- raw JSONL immutable;
- benchmark failure не записывается как numeric success;
- unsupported case не получает latency `0`;
- dirty tree и environment mismatch явно маркируются;
- conformance failure блокирует publishable performance claim.

### 24.2 Nsight Systems

Используется для:

```text
CPU launch gaps
implicit synchronizations
H2D/D2H copies
kernel/library sequence
allocation activity
NVTX stage/layer ranges
end-to-end critical path
```

NVTX levels:

```text
engine iteration
prefill batch
decode batch
model layer
attention stage
MLP stage
custom operator
KV update
scheduler step
```

### 24.3 Nsight Compute

Собирается только для representative kernels/shapes:

```text
occupancy
register usage
shared memory
DRAM throughput
L1/L2 behavior
global load/store efficiency
warp stalls
branch behavior
bank conflicts
instruction mix
Tensor Core utilization
```

Не запускать exhaustive sections на всей matrix.

### 24.4 Evidence contract

Profiler-based вывод содержит:

```text
command
commit
backend/version
suite/case
kernel name
shape
GPU environment
selected metrics
interpretation
link/path to report
relation to benchmark number
limitations
```

Profiler-derived latency не является финальным headline timing.

---

## 25. GitHub workflow и issue structure

### 25.1 Milestones

```text
M1 — Runtime Core, RMSNorm & GEMM Lab
M2 — Contiguous KV & Decode Attention
M3 — Transformer Block & TinyModel
M4 — Paged KV & Backend API v1
M5 — Scheduler & Continuous Batching
M6 — Research & Kernel-First v0.1
```

### 25.2 Labels

```text
area:backend-api
area:runtime
area:memory
area:gemm
area:attention
area:kv-cache
area:model
area:scheduler
area:correctness
area:benchmark
area:profiling
area:fixtures
area:docs

type:feature
type:test
type:benchmark
type:research
type:bug
type:refactor
type:adr

priority:p0
priority:p1
priority:p2

scope:mandatory
scope:stretch
scope:post-v0.1
```

### 25.3 Issue sizing

Один engineering issue должен занимать **2–4 часа**. Более крупная задача разбивается по observable deliverables.

Одновременно в `In Progress` находится один основной technical issue и, при необходимости, один небольшой docs/test issue.

### 25.4 Engineering issue template

```markdown
## Context

## Deliverable

## Semantic contract

Inputs, outputs, layouts, dtype, ownership, stream и error behavior.

## Public API impact

None / candidate-compatible / breaking. Для breaking change приложить ADR.

## Scope

## Non-goals

## Acceptance criteria

- [ ] Code builds.
- [ ] Tests added.
- [ ] Required correctness/conformance passes.
- [ ] Unsupported cases documented.
- [ ] Hot path adds no allocation/global sync.
- [ ] Benchmark added or not required with reason.

## Verification

Commands, fixtures, suite IDs, tolerances.

## Estimate

2–4 hours.
```

### 25.5 Research issue template

```markdown
## Question

## Hypothesis

## Scope / non-goals

## Reference

## Strong baseline

## Variants

## Correctness gate

## Benchmark matrix

## Model/engine-level evaluation

## Profiler evidence

## Deliverables

Raw results, report, reproduction command.

## Definition of Done

- [ ] Correctness passes.
- [ ] Results reproducible.
- [ ] Baseline documented.
- [ ] Negative results retained.
- [ ] Limitations stated.
```

### 25.6 Required CLI contract v0.1

```bash
minirt backends list
minirt conformance --backend <name> --suite <suite-id>
minirt bench --backend <name> --suite <suite-id>
minirt model run --config <config>
minirt workload run --config <config> --trace <trace>
```

`minirt doctor` — useful stretch goal. `backends describe`, `suites list`, `compare` и `serve-bench` — post-v0.1 polish.

CLI requirements:

- resolved config сохраняется в result bundle;
- unsupported отличается от infrastructure failure;
- error содержит backend, case и actionable reason;
- command печатает reproduction line;
- conformance failure возвращает non-zero exit code;
- machine-readable output отделён от human summary.

### 25.7 CI and release gates

#### Every PR without GPU

```text
format/lint
public-header standalone compile
public/internal boundary test
CMake configure/build
Python reference tests
fixture determinism tests
suite schema validation
```

#### GPU-required gate

```text
smoke correctness/conformance
TinyModel integration
non-default-stream test
allocation audit
Compute Sanitizer selected subset
benchmark smoke for gross regression
```

#### v0.1 release gate

```text
mandatory engine correctness
paged/contiguous parity
scheduler deterministic trace suite
result-bundle validation
clean-build reproduction
main research experiment
```

External contributor dry run не является release gate v0.1.

### 25.8 Backend acceptance checklist

- [ ] Backend API version совместима.
- [ ] Используются только public headers.
- [ ] `supports()` и reasons протестированы.
- [ ] Workspace/prepared lifecycle протестированы.
- [ ] Non-default stream test проходит.
- [ ] Declared-scope correctness проходит.
- [ ] Benchmark result приложен, если делается performance claim.
- [ ] Model-level effect измерен или неприменимость объяснена.
- [ ] Hot path не аллоцирует и не синхронизирует device глобально.
- [ ] `ModelRunner`, scheduler и benchmark runner не получили backend-specific branches.

---

## 26. Правила сокращения scope

При отставании scope сокращается в следующем порядке:

1. manifest schema, scaffolding, out-of-tree packaging;
2. external contributor dry run и polished tutorials;
3. automatic compare/report/plots tooling;
4. дополнительные kernel variants вне research track;
5. full target checkpoint, если TinyModel и target shapes доступны;
6. второй dtype;
7. custom fused prefill;
8. Tensor Core GEMM laboratory;
9. sampling;
10. prefix sharing/copy-on-write;
11. chunked prefill;
12. mixed prefill/decode packed execution;
13. CUDA Graphs;
14. public scheduler API.

Нельзя убирать:

- exact architecture/GPU/dtype contracts;
- public/internal boundary;
- static registry для трёх mandatory backend families;
- correctness/reference path;
- machine-readable benchmark results;
- TinyModel;
- baseline prefill/decode;
- one correct Transformer block;
- contiguous/paged parity;
- request isolation tests;
- allocation audit;
- selected sanitizer runs;
- scheduler deterministic tests;
- one completed research report.

### 26.1 Emergency scope

Если к концу недели 12 есть существенное отставание:

```text
оставить TinyModel вместо full checkpoint;
оставить cuBLASLt для всех model GEMMs;
оставить naive paged attention;
ограничить scheduler 4 active requests;
не делать mixed prefill/decode;
выбрать fused RoPE + KV write или block-size study как low-risk research track;
сохранить только minimal registry/conformance/benchmark infrastructure;
перенести весь contributor-platform polish в post-v0.1.
```

Если paged parity не достигнута к концу недели 16, GPU scheduler integration останавливается до исправления. Scheduler unit work может продолжаться через fake executor.

---

## 27. Risk register

| Риск | Ранний признак | Mitigation | Stop condition |
|---|---|---|---|
| Platform work вытесняет engine | много CLI/schema/docs, но нет attention path | platform cap 15–20h, deferred list | остановить platform polish до следующего engine gate |
| Over-abstraction | много interfaces, нет working vertical slice | три Tier 1 interfaces, example-first | удалить unused extension points |
| API слишком мелкий | fusion требует менять `ModelRunner` | composite `AttentionStageBackend` | refactor до deep optimization |
| API слишком широкий | backend видит allocator/scheduler internals | semantic views, engine-owned memory | убрать internal types из signatures |
| API заморожен слишком рано | paged path требует breaking changes | candidate до M4 parity | не объявлять v1 до paged gate |
| End-to-end path опаздывает | week 10 без block parity | stop optional kernels, use cuBLASLt | paged KV не начинать |
| Benchmark comparisons нечестны | разные inputs/environment | fixed suites, hashes, manifests | claim не публиковать |
| Hidden allocations | Nsight показывает allocations in iteration | prepare lifecycle, arenas, counters | milestone gate not passed |
| Paged cache corruption | data mixing after reuse | generation handles, sentinels, randomized tests | scheduler integration stop |
| Full model съедает время | exporter/checkpoint issues dominate | TinyModel mandatory | отказаться от full checkpoint |
| Performance noise | contradictory runs | repeated samples, GPU-state metadata | no headline claim |
| Research выбран заранее | выбранная operation не hotspot | M3/M4 profile gate | выбрать другой track |
| Учебный GEMM поглощает roadmap | неделя 4 уходит на tiling | hard timebox 8h, cuBLASLt model backend | остановить GEMM lab |
| Burnout | регулярный перенос задач и ночные марафоны | 10h cap, 2–4h issues, reserve weeks | сократить scope, не увеличивать weekly load |

### 27.1 Decision rule перед оптимизацией

```text
Какова доля operation в end-to-end latency?
Какой strong baseline?
Какой correctness gate?
Какой bottleneck предполагается?
Какой profiler metric подтвердит hypothesis?
Какой минимальный variant может её опровергнуть?
Каков timebox и stop condition?
```

---

## 28. Final Definition of Done

### 28.1 Runtime

- [ ] Собственный C++/CUDA execution path.
- [ ] Одна документированная decoder-only architecture.
- [ ] Одна target GPU architecture.
- [ ] Explicit dtype/numerics/layout policy.
- [ ] Mandatory prefill и decode paths.
- [ ] One correct Transformer block.
- [ ] Tiny 2-layer model.
- [ ] Contiguous и paged KV-cache.
- [ ] Token-budget scheduler и continuous batching.
- [ ] Zero steady-state device allocations.

### 28.2 Kernel experimentation infrastructure

- [ ] Public/internal headers физически разделены.
- [ ] Static `BackendRegistry`.
- [ ] Source-level Backend API v1 заморожен после M4 parity.
- [ ] `RmsNormBackend`, `GemmBackend`, `AttentionStageBackend` selectable через config.
- [ ] Example RMSNorm и attention backends используют только public headers.
- [ ] Correctness/conformance runner работает для mandatory public interfaces.
- [ ] Benchmark runner создаёт JSONL и environment manifest.
- [ ] Unsupported configurations имеют actionable reasons.
- [ ] Backend replacement не добавляет special branches в `ModelRunner`/scheduler.
- [ ] Main operator/model/workload runs воспроизводятся documented commands.

Не входят в Final DoD v0.1:

```text
manifest schema
scaffolding
out-of-tree polished package
external contributor dry run
community-ready tutorials
full compare/report UI
stable binary ABI
```

### 28.3 Correctness и research

- [ ] Operator/composition/layer/model/engine tests.
- [ ] Selected Compute Sanitizer suite.
- [ ] Nsight Systems и Nsight Compute reports.
- [ ] Один deep research result.
- [ ] Strong baseline и isolated variants.
- [ ] Negative results сохранены.
- [ ] Limitations и threats to validity описаны.
- [ ] Main experiment reproducible by one command.

### 28.4 Release naming

```text
v0.1-alpha:
  часть mandatory engine path или correctness gates ещё не завершена;
  результаты полезны, но roadmap выполнен не полностью.

v0.1:
  kernel-first engine, scheduler, paged path и research goals complete;
  minimal backend API/correctness/benchmark path reproducible.

v0.2 contributor-ready:
  manifests, scaffolding, out-of-tree workflow, tutorials,
  external contributor dry run и friction fixes complete.
```

---

## 29. Первые 10 часов

### Session 1 — Product and contracts, 2 часа

- выбрать exact architecture и target GPU;
- определить selected dtype;
- зафиксировать scope v0.1 и post-v0.1;
- создать ADR 0001–0004 skeletons;
- создать repository, LICENSE, README, plan и GitHub milestones.

### Session 2 — Build and runtime skeleton, 2 часа

- настроить CMake/CUDA presets;
- создать `include/minirt/backend_api`, `include/minirt/core`, `src/internal`;
- добавить CUDA error helpers;
- создать `CudaStream`, `CudaEvent`, `DeviceBuffer` skeletons;
- добавить standalone public-header compile test.

### Session 3 — Execution context and measurement, 3 часа

- реализовать `ExecutionContext`;
- создать cuBLASLt handle;
- добавить NVTX smoke range;
- реализовать CUDA-event + wall timing skeleton;
- добавить JSONL и environment manifest;
- проверить один vector-add smoke kernel.

### Session 4 — First backend vertical slice, 3 часа

- создать minimal `BackendDescriptor` и static registry;
- добавить `naive_rmsnorm` reference/problem/args skeleton;
- написать PyTorch reference;
- реализовать первую корректную FP32 CUDA version или минимальный runnable kernel;
- добавить one-case correctness test;
- закрыть первый issue с exact reproduction command.

Result after 10 hours:

```text
frozen kernel-first direction
buildable CUDA repository
runtime resource wrappers
public/internal boundary
first registered backend
first machine-readable timing/correctness result
```

Не делать в первые 10 часов:

```text
manifest schema
scaffolding
full CLI
report generator
attention abstraction
пустую большую directory tree
```

---

## 30. Post-v0.1 roadmap

После kernel-first v0.1 направление выбирается по measured engine hotspots и реальному API friction.

### 30.1 Contributor-ready v0.2

```text
backend manifest schema
scaffolding generator
out-of-tree backend package
polished RMSNorm/attention tutorials
suite discovery/matrix CLI
full compare/report tooling
external contributor dry run
friction metrics
source migration policy
community PR/release checklist
```

### 30.2 Engine/performance extensions

```text
optimized paged attention continuation
custom fused prefill
CUDA Graph decode buckets
full checkpoint support
quantization
prefix caching
chunked prefill
Tensor Core small-M GEMM
second GPU architecture
experimental scheduler policies
```

### 30.3 Decision questions

Следующий roadmap начинается не с новой feature list, а с:

```text
какие kernels реально являются hotspots;
какие extension points использовались;
какие API changes потребовал paged/model path;
какие benchmark suites дали полезные выводы;
какой contributor tooling действительно нужен;
что можно стабилизировать без потери читаемости;
какой следующий research question имеет model-level value.
```
