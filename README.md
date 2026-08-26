# Gemma.c

A pure C implementation of Gemma 1 / 2 / 3 inference, no third-party dependencies, single file, builds with `gcc` out of the box. Comes with a set of Python export scripts to convert Hugging Face Gemma checkpoints into this project's own `.bin` format.
 
Inspired by [llama2.c](https://github.com/karpathy/llama2.c). The goal is to make Gemma inference as clear and hackable as possible with minimal code.

## Features
 
- Pure C (C11), no runtime dependencies, compiles with a single `gcc` invocation
- Supports Gemma 1 / 2 / 3 (including the local/global alternating sliding-window attention, QK-Norm, and attention/logit softcapping introduced in Gemma 2/3)
- GQA (Grouped-Query Attention) + RoPE (separate local/global theta)
- KV cache inference with streaming, long-context generation
- Built-in INT8 weight quantization (per-channel quantization for `Linear`/`Embedding`), paired with a one-shot `--quantize` flag on the Python export side
- OpenMP multithreading + `-march=native` compile-time optimizations
- Runs on Windows (with UTF-8 console handling) and Linux/macOS
- Interactive chat mode following Gemma's official chat template (`<start_of_turn>` / `<end_of_turn>`)
- A minimal BPE tokenizer (re-implemented in C, with weights embedded into the `.bin` file by the export script)

## Project layout

```
.
├── gemma.c        # The inference engine itself (model loading, forward pass, sampling, CLI)
├── Makefile       # Cross-platform build script (Linux/macOS/Windows)
├── export.py      # HuggingFace -> .bin weight exporter
├── model.py       # Reference PyTorch implementation (for correctness checks before export)
├── sample.py      # PyTorch-side sampling logic (for parity testing against gemma.c)
└── tokenizer.py   # Reference tokenizer implementation
```

## Quick start
 
### 1. Export a model
 
```bash
pip install torch transformers
 
python export.py google/gemma-3-4b-it \
    -o exported/gemma-3-4b-it.bin \
    -d float16
```
 
Export an INT8-quantized version (smaller footprint, noticeably faster on CPU):
 
```bash
python export.py google/gemma-3-4b-it \
    -o exported/gemma-3-4b-it_q.bin \
    -d float16 \
    -q
```

`export.py` arguments:
 
| Argument | Description |
| -------- | ----------- |
| `modelfile`        | HuggingFace model id or a local directory |
| `-o, --output`     | Output `.bin` path (required) |
| `-d, --dtype`      | Weight precision: `float16` / `float32` / `bfloat16` (default: `float16`) |
| `-q, --quantize`   | Enable INT8 quantization |
| `-c, --cache-path` | Custom HuggingFace cache directory |

### 2. Build
 
```bash
make
```
 
On Windows (MinGW), just run `make` as well. The Makefile detects the platform automatically and statically links the binary.
 
### 3. Run inference
 
One-shot generation:
 
```bash
./gemma exported/gemma-3-4b-it_q.bin -i "Once upon a time" -t 1.2 -k 50 -p 0.8 -r 1.15
```
 
Interactive chat:
 
```bash
./gemma exported/gemma-3-4b-it_q.bin -c -l 4096
```
 
CLI options:
 
| Option | Description |
| ------ | ----------- |
| `-l, --seqlen <N>`      | Maximum sequence length (default: 16384) |
| `-k, --topk <N>`        | Top-k sampling (default: 0, disabled) |
| `-p, --topp <F>`        | Top-p / nucleus sampling, `0 < p <= 1` (default: 1.0) |
| `-t, --temperature <F>` | Sampling temperature, `>= 0` (default: 1.0; 0 means greedy decoding) |
| `-r, --rpen <F>`        | Repetition penalty, `>= 1.0` (default: 1.0) |
| `-s, --seed <N>`        | Random seed (default: current time) |
| `-i, --prompt <S>`      | Input prompt, ignored in chat mode (default: `"Once upon a time"`) |
| `-c, --chat`            | Enable interactive chat mode |
| `-h, --help`            | Show help message |
 
Generation can be gracefully interrupted with `Ctrl+C` at any time.

## Performance

Test environment:
 
- CPU: `Intel(R) Core(TM) Ultra 5 225H`
- RAM: `31.5 GB`
- OS: `Windows 11`
- Compiler: `gcc 15.2.0 x86_64-w64-mingw32`
- Threads (`OMP_NUM_THREADS`): `6`

| Model | Generation speed (tok/s, 100 tokens) | Prompt processing speed (tok/s, 100 tokens) | Base memory | Memory per tok |
| ----- | ------------------------------------ | ------------------------------------------- | ----------- | -------------- |
| Gemma-1 2B (FP16)   | `<TODO>`   | `<TODO>`   | `<TODO>`    | `<TODO>`   |
| Gemma-1 2B (W8A8)   | `<TODO>`   | `<TODO>`   | `<TODO>`    | `<TODO>`   |
| Gemma-1 7B (FP16)   | `<TODO>`   | `<TODO>`   | `<TODO>`    | `<TODO>`   |
| Gemma-1 7B (W8A8)   | `<TODO>`   | `<TODO>`   | `<TODO>`    | `<TODO>`   |
| Gemma-2 2B (FP16)   | `<TODO>`   | `<TODO>`   | `<TODO>`    | `<TODO>`   |
| Gemma-2 2B (W8A8)   | `<TODO>`   | `<TODO>`   | `<TODO>`    | `<TODO>`   |
| Gemma-2 9B (FP16)   | `<TODO>`   | `<TODO>`   | `<TODO>`    | `<TODO>`   |
| Gemma-2 9B (W8A8)   | `<TODO>`   | `<TODO>`   | `<TODO>`    | `<TODO>`   |
| Gemma-3 270M (FP16) | `23.45`    | `35.60`    | `0.50 GB`   | `18 KB`    |
| Gemma-3 270M (W8A8) | `44.92`    | `50.30`    | `0.25 GB`   | `18 KB`    |
| Gemma-3 1B (FP16)   | `7.64`     | `10.04`    | `1.86 GB`   | `26 KB`    |
| Gemma-3 1B (W8A8)   | `25.15`    | `29.97`    | `0.93 GB`   | `26 KB`    |
| Gemma-3 4B (FP16)   | `2.30`     | `2.89`     | `7.23 GB`   | `136 KB`   |
| Gemma-3 4B (W8A8)   | `9.32`     | `10.88`    | `3.62 GB`   | `136 KB`   |
| Gemma-3-12B (FP16)  | `<TODO>`   | `<TODO>`   | `<TODO>`    | `<TODO>`   |
| Gemma-3 12B (W8A8)  | `3.62`     | `3.80`     | `10.96 GB`  | `384 KB`   |
| Gemma-3-27B (FP16)  | `<TODO>`   | `<TODO>`   | `<TODO>`    | `<TODO>`   |
| Gemma-3-27B (W8A8)  | `<TODO>`   | `<TODO>`   | `<TODO>`    | `<TODO>`   |

## Known limitations
 
- Text-only for now, Gemma 3's image input branch isn't wired into the C inference path yet
- Weight export and tokenizer are built against the official HuggingFace checkpoint layout; other weight sources may need extra adaptation
- Single-machine CPU inference only, no GPU / multi-node distributed support yet

## Roadmap
 
- [ ] KV cache quantization
- [ ] W4A16 quantization
- [ ] Multimodal inference
- [ ] Gemma 4 architecture support
 
## Acknowledgements
 
- [Google Gemma](https://ai.google.dev/gemma) official models and technical reports
- [llama2.c](https://github.com/karpathy/llama2.c) for inspiring this project

## License

MIT
