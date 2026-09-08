# gpglu-ed

Single-GPU experimental LLM inference runtime на C++/CUDA для одной decoder-only архитектуры с заменяемыми backend-ами операторов, собственным KV-cache, простым scheduler и воспроизводимыми исследованиями производительности.

[План разработки](docs/plan.md) · [Правила тикетов](docs/workflow.md) ·
[Correctness и измерения](docs/validation.md)

## Требования

- CMake 3.25 или новее;
- Ninja;
- CUDA Toolkit с `nvcc`;
- NVIDIA GPU и совместимый драйвер для запуска CUDA-тестов.

По умолчанию CUDA-код собирается для архитектуры `75`.

## Debug-сборка и тесты

На чистом клоне используются три команды:

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug --output-on-failure
```

Основной CUDA smoke-тест после сборки можно запустить отдельно:

```bash
./build/debug/minirt_cuda_smoke
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
