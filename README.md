# Gemma.c

Gemma 1, 2 and 3 implemented in a single file of pure C.

This is a from-scratch inference engine for Google's Gemma family of
language models: Gemma 1, 2 and 3, text-only and multimodal (SigLIP vision
tower included), in one .c file, no third-party dependencies, builds with
a single gcc/clang/msvc invocation. It is not a wrapper around llama.cpp or
ggml, the transformer, the attention masks, the RoPE tables, the int8
kernels, the BPE tokenizer, and the image decoder / resizer are all
self-included here. I did this mostly to actually understand how Gemma
works end to end, and the by-product is a codebase that's small enough to
read in an afternoon. It's inspired by karpathy/llama2.c, which is the
same idea applied to Llama.

## Features
 
- Gemma 1 / 2 / 3, including multimodal (SigLIP vision tower), single `.c`
  file, no runtime dependencies beyond libc and (optional) OpenMP
- Hand-written GEMM/GEMV kernels for both fpx and int8, packed and tiled,
  parallelized across heads/rows with OpenMP
- W8A8 quantization (per-channel weights, per-token activations), enabled
  with a one-shot `-q` flag on the export side
- Hybrid sliding-window / full attention (Gemma 2/3's local + global
  layers), using a FlashAttention-style block-tiling scheme for the causal
  mask
- Full SigLIP vision encoder with bidirectional attention between image
  soft tokens, plus a pan & scan crop utility for high-aspect-ratio images
- mmap'd weight loading by default (`--disable-mmap` falls back to a plain
  read, useful on filesystems where mmap is flaky)
- KV-cache streaming generation with chunked prefill for long prompts
- Temperature, top-k, top-p, and repetition-penalty sampling
- float32 / float16 / bfloat16 weights, picked at compile time
- A from-scratch BPE tokenizer, with vocab + merges embedded directly into
  the exported `.bin` file
- Runs on Linux, macOS, and Windows (MinGW/MSYS2), with either gcc, clang or
  msvc

## Project layout

```
.
├── gemma.c        # The inference engine itself
├── Makefile       # Cross-platform build script
├── export.py      # HuggingFace -> .bin weight exporter
├── model.py       # Reference PyTorch implementation
├── sample.py      # PyTorch-side sampling logic
├── preprocess.py  # PyTorch-side image pre-processor
└── tokenizer.py   # Reference tokenizer implementation
```

The Python files aren't needed to *run* the model — only `gemma.c` and a
`.bin` file are. They exist to produce that `.bin` file from a real
HuggingFace checkpoint, and serve as a side-by-side reference.

## Quick start
 
### 1. Export a model
 
```bash
pip install torch transformers accelerate safetensors huggingface_hub
 
python export.py google/gemma-3-4b-it \
    -o exported/gemma-3-4b-it.bin \
    -d float16
```
 
Export an INT8-quantized version (roughly a quarter of the FP16 size, and
noticeably faster on CPU. See the benchmarks below):
 
```bash
python export.py google/gemma-3-4b-it \
    -o exported/gemma-3-4b-it_q.bin \
    -d float16 \
    -q
```

`export.py` arguments:
 
| Argument           | Description                                                               |
| ------------------ | ------------------------------------------------------------------------- |
| `modelfile`        | HuggingFace model id or a local directory                                 |
| `-o, --output`     | Output `.bin` path (required)                                             |
| `-d, --dtype`      | Weight precision: `float16` / `float32` / `bfloat16` (default: `float16`) |
| `-q, --quantize`   | Enable INT8 (W8A8) quantization                                           |
| `-c, --cache-path` | Custom HuggingFace cache directory                                        |
 
### 2. Build
 
```bash
make
```
 
`make` auto-detects whether you're on gcc/clang/msvc and picks safe flags for
each (this actually matters here). Override the compiler or dtype from the
command line if you need to:
 
```bash
make CC=clang
make DTYPE=BF16      # must match whatever -d you exported with
make debug           # -O0 -g -fsanitize=address
```
 
On Windows, install MinGW-w64 or use MSYS2, then run `make` the same way.
The Makefile detects `Windows_NT` and links a static binary so it doesn't
need MinGW's DLLs sitting next to it.

### 3. Run inference
 
One-shot generation:
 
```bash
./gemma exported/gemma-3-4b-it_q.bin -i "Once upon a time" \
    --temperature 1.2 --topk 50 --topp 0.8 --rpen 1.15
```
 
Interactive chat:
 
```bash
./gemma exported/gemma-3-4b-it_q.bin --chat --seqlen 4096
```
 
Multimodal models take images inline in the prompt with `@image{...}`:
 
```bash
./gemma exported/gemma-3-4b-it_q.bin --chat
> Describe what you see in @image{photo.jpg}.
```
 
For very wide or tall images, `@image_pas{...}` runs Gemma 3's pan & scan
utility first, feeding the model both the full downsized image and a few
higher-resolution crops:
 
```bash
> What's happening at the bottom of @image_pas{long_screenshot.png}?
```
 
CLI options:

## Usage
 
```text
./gemma <modelfile> [options]
```
 
### Arguments
 
| Argument | Description |
| --- | --- |
| `modelfile` | Path to the model file |
 
### Options
 
| Option              | Description                                       | Default               |
| ------------------- | ------------------------------------------------- | --------------------- |
| `--seqlen <N>`      | Set sequence length                               | 16384                 |
| `--topk <N>`        | Set top-k sampling value                          | 0                     |
| `--seed <N>`        | Set random seed                                   | current time          |
| `--chunk <N>`       | Set prefilling chunk size, must be >= 1           | 1024                  |
| `--temperature <F>` | Set temperature value, must be >= 0.0             | 1.0                   |
| `--topp <F>`        | Set top-p sampling value, must be 0.0 < p <= 1.0  | 1.0                   |
| `--rpen <F>`        | Set repetition penalty, must be >= 1.0            | 1.0                   |
| `--prompt <S>`      | Set input prompt, ignored if chat mode is enabled | "Once upon a time"    |
| `--chat`            | Enable chat mode                                  | —                     |
| `--disable-mm`      | Disable multimodal capability                     | —                     |
| `--disable-mmap`    | Disable mmap (memory mapped file)                 | —                     |
| `--verbose`         | Print model info                                  | —                     |
| `--help`, `-?`      | Display this help message                         | —                     |
 
### Controls
 
| Control  | Description                              |
| -------- | ---------------------------------------- |
| `Ctrl+C` | Gracefully interrupt generation and exit |
 
### Examples
 
| Command                                                                                   |
| ----------------------------------------------------------------------------------------- |
| `./gemma model.bin -l 2048 -t 0.8 --chat`                                                 |
| `./gemma model.bin -i "Hello I'm a language model," --seqlen 4096 --topk 50 --seed 12345` |

## Performance

Test environment:
 
- CPU: `Intel(R) Core(TM) Ultra 5 225H`
- RAM: `31.5 GB`
- OS: `Windows 11`
- Compiler: `gcc 15.2.0 x86_64-w64-mingw32`
- Compiler arguments: `-Ofast -fopenmp -march=native -mtune=native`
- Run arguments: `--temperature 0 --seqlen 100 --verbose --disable-mmap`
- Threads (`OMP_NUM_THREADS`): `6`

| Model               | Generation speed (tok/s, 100 tokens) | Prompt processing speed (tok/s, 100 tokens) | Base memory | Memory per tok |
| ------------------- | ------------------------------------ | ------------------------------------------- | ----------- | -------------- |
| Gemma-1 2B (FP16)   | `<TODO>`                             | `<TODO>`                                    | `<TODO>`    | `<TODO>`       |
| Gemma-1 2B (W8A8)   | `<TODO>`                             | `<TODO>`                                    | `<TODO>`    | `<TODO>`       |
| Gemma-1 7B (FP16)   | `<TODO>`                             | `<TODO>`                                    | `<TODO>`    | `<TODO>`       |
| Gemma-1 7B (W8A8)   | `<TODO>`                             | `<TODO>`                                    | `<TODO>`    | `<TODO>`       |
| Gemma-2 2B (FP16)   | `<TODO>`                             | `<TODO>`                                    | `<TODO>`    | `<TODO>`       |
| Gemma-2 2B (W8A8)   | `<TODO>`                             | `<TODO>`                                    | `<TODO>`    | `<TODO>`       |
| Gemma-2 9B (FP16)   | `<TODO>`                             | `<TODO>`                                    | `<TODO>`    | `<TODO>`       |
| Gemma-2 9B (W8A8)   | `<TODO>`                             | `<TODO>`                                    | `<TODO>`    | `<TODO>`       |
| Gemma-3 270M (FP16) | `23.45`                              | `35.60`                                     | `0.50 GB`   | `18 KB`        |
| Gemma-3 270M (W8A8) | `44.92`                              | `50.30`                                     | `0.25 GB`   | `18 KB`        |
| Gemma-3 1B (FP16)   | `7.64`                               | `10.04`                                     | `1.86 GB`   | `26 KB`        |
| Gemma-3 1B (W8A8)   | `25.15`                              | `29.97`                                     | `0.93 GB`   | `26 KB`        |
| Gemma-3 4B (FP16)   | `2.30`                               | `2.89`                                      | `7.23 GB`   | `136 KB`       |
| Gemma-3 4B (W8A8)   | `9.32`                               | `10.88`                                     | `3.62 GB`   | `136 KB`       |
| Gemma-3-12B (FP16)  | `<TODO>`                             | `<TODO>`                                    | `<TODO>`    | `<TODO>`       |
| Gemma-3 12B (W8A8)  | `3.62`                               | `3.80`                                      | `10.96 GB`  | `384 KB`       |
| Gemma-3-27B (FP16)  | `<TODO>`                             | `<TODO>`                                    | `<TODO>`    | `<TODO>`       |
| Gemma-3-27B (W8A8)  | `<TODO>`                             | `<TODO>`                                    | `<TODO>`    | `<TODO>`       |

## Known limitations

- Single-machine CPU inference only, no GPU / multi-node distributed support
  yet
- The hand-written GEMM/GEMV kernels are still fairly unoptimized compared
  to BLAS or a real SIMD kernel

## Roadmap
 
- [ ] KV cache quantization
- [ ] W4A16 quantization
- [x] Multimodal inference
- [ ] Gemma 4 architecture support
 
## Acknowledgements
 
- [Google Gemma](https://ai.google.dev/gemma) official models and technical reports
- [llama2.c](https://github.com/karpathy/llama2.c) for inspiring this project

## License

MIT
