# gpglu-ed

Single-GPU experimental LLM inference runtime на C++/CUDA для одной decoder-only архитектуры с заменяемыми backend-ами операторов, собственным KV-cache, простым scheduler и воспроизводимыми исследованиями производительности.

[План разработки](docs/plan.md) · [Правила тикетов](docs/workflow.md) ·
[Correctness и измерения](docs/validation.md)

## Требования

- CMake 3.25 или новее;
- Ninja;
- C++ компилятор и CUDA Toolkit с `nvcc`, поддерживающие C++20;
- NVIDIA GPU и совместимый драйвер для запуска CUDA-тестов.

По умолчанию CUDA-код собирается для архитектуры `75`.

## Debug-сборка и тесты

На чистом клоне используются три команды:

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug --output-on-failure
```

## Подключение заголовков

Все проектные заголовки подключаются через `<minirt/...>`:

```cpp
#include <minirt/core/cuda_error.hpp>
#include <minirt/internal/memory/device_buffer.hpp>
#include <minirt/internal/runtime/cuda_stream.hpp>
#include <minirt/test/test_require.hpp>
```

Публичные заголовки находятся в `include/minirt/`. Внутренние — в
`src/minirt/internal/`: корень `src/` подключается с видимостью `PRIVATE` к runtime
и явно добавляется внутренним тестам. Тестовые заголовки находятся в
`tests/support/minirt/test/` и доступны только тестовым targets.

## Структура тестов

```text
tests/
  CMakeLists.txt
  support/minirt/test/test_require.hpp
  runtime/runtime_wrappers_test.cpp
  memory/device_buffer_test.cu
```

Во всех тестах используй общий макрос:

```cpp
#include <minirt/test/test_require.hpp>

// Внутри тестовой функции:
MINIRT_TEST_REQUIRE(actual == expected);
```

Он вычисляет условие один раз и при провале бросает `minirt::test::Failure`
с текстом условия, файлом и строкой. Проверки работают в `void`-функциях и в
Release-сборках. `main()` должен ловить `const std::exception&`, печатать
`error.what()` и возвращать `EXIT_FAILURE`, как в существующих тестах.
В `.cu` макрос предназначен для CPU-кода вокруг запуска kernel.

Новые тесты регистрируй через `minirt_add_test` или `minirt_add_internal_test`
в `tests/CMakeLists.txt`: оба helper-а автоматически дают доступ к общему заголовку.

`memory/device_buffer_test.cu` содержит CUDA-ядро, поэтому собирается через `nvcc`.

Сборка и запуск только теста буфера:

```bash
cmake --preset debug
cmake --build --preset debug --target minirt_device_buffer
ctest --preset debug -R '^device_buffer$' --output-on-failure
```

## Release-сборка и тесты

```bash
cmake --preset release
cmake --build --preset release
ctest --preset release --output-on-failure
```

## Сборка только runtime-библиотеки

Если тестовые targets не нужны:

```bash
cmake --preset debug -DBUILD_TESTING=OFF
cmake --build --preset debug --target minirt_runtime
```

## Compilation database для IDE

Configure presets включают `CMAKE_EXPORT_COMPILE_COMMANDS`, поэтому отдельная команда генерации не требуется. После конфигурации файл находится по адресу:

```text
build/debug/compile_commands.json
```

Для release-конфигурации используется `build/release/compile_commands.json`.

Файл `.clangd` указывает на `build/debug`, поэтому clangd получает `-std=c++20`
из команд компиляции CMake. После изменения стандарта достаточно выполнить
`cmake --preset debug`; отдельно задавать стандарт в `.clangd` или настройках
VS Code не требуется. Если редактор продолжает показывать старую диагностику,
выполните команду `clangd: Restart language server`.

Чтобы удалить старый CMake cache и заново сконфигурировать debug-сборку:

```bash
cmake --preset debug --fresh
```

## Форматирование C++ и CUDA

Проект использует `clang-format` и настройки из `.clang-format`. После конфигурации
debug-сборки доступны две команды:

```bash
# Переформатировать файлы на месте
cmake --build --preset debug --target format

# Только проверить форматирование, ничего не изменяя
cmake --build --preset debug --target format-check
```

В VS Code форматирование C++, CUDA и C включено при сохранении через расширение
`clangd`. Если `clang-format` ещё не установлен в Fedora:

```bash
sudo dnf install clang-tools-extra
```
