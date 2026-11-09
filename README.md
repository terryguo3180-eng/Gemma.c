# Gemma.c

Gemma 1, 2 and 3 implemented in a single file of pure C.

This is a from-scratch inference engine for Google's Gemma family of language
models: Gemma 1, 2 and 3, text-only and multimodal (SigLIP vision tower
included), in one .c file with no dependencies beyond libc and (optionally)
OpenMP. It is not a wrapper around llama.cpp or ggml: the transformer, the
attention masks, the RoPE tables, the int8 kernels and the BPE tokenizer are
all written from scratch in this file. The image decoding uses a minified
build of stb_image and stb_image_resize2 (PNG and JPEG only), embedded in
`gemma.c`. See [Image decoding](#image-decoding).

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
- mmap'd weight loading by default (`--no-mmap` falls back to a plain
  read, useful on filesystems where mmap is flaky)
- KV-cache streaming generation with chunked prefill for long prompts
- Temperature, top-k, top-p, and repetition-penalty sampling
- float32 / float16 / bfloat16 weights, picked at compile time
- A from-scratch BPE tokenizer, with vocab + merges embedded directly into
  the exported `.bin` file
- Runs on Linux, macOS, and Windows (MinGW/MSYS2), with either gcc, clang or
  msvc
- PNG/JPEG image input via embedded stb (see Image decoding)

## Project layout

```
.
├── gemma.c        # The inference engine (minified stb image decoder embedded at the end of the file)
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
 
Gemma weights are gated: accept the license on the model's Hugging Face page
and log in first (`huggingface-cli login`, or `hf auth login` on newer versions).

```bash
pip install torch transformers accelerate safetensors huggingface_hub
 
python export.py google/gemma-3-1b-it \
    -o exported/gemma-3-1b-it.bin \
    -d float16
```
 
Export an INT8-quantized version (about half the FP16 size, and 1.8–3.4x
faster generation depending on the model, see the benchmarks below):
 
```bash
python export.py google/gemma-3-1b-it \
    -o exported/gemma-3-1b-it_q.bin \
    -d float16 \
    -q
```

`export.py` arguments:
 
| Argument           | Description                                                               |
| ------------------ | ------------------------------------------------------------------------- |
| `modelfile`        | HuggingFace model id or a local directory                                 |
| `-o, --output`     | Output `.bin` path (required)                                             |
| `-d, --dtype`      | Weight precision: `float16` / `float32` / `bfloat16` (default: `float32`) |
| `-q, --quantize`   | Enable INT8 (W8A8) quantization                                           |
| `-c, --cache-path` | Custom HuggingFace cache directory                                        |
 
### 2. Build
 
```bash
make
```
 
`make` auto-detects whether you're on gcc/clang/msvc and picks safe flags for
each (this actually matters here). Override the compiler or dtype from the
command line if you need to (MSVC only supports float32):
 
```bash
make CC=clang
make DTYPE=BF16      # must match whatever -d you exported with, FP32 / FP16 / BF16
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
./gemma exported/gemma-3-4b-it_q.bin --chat --seqlen 1024
```
 
Multimodal models take images inline in the prompt with `@image{...}`:
 
```bash
./gemma exported/gemma-3-4b-it_q.bin --chat
> Describe what you see in @image{photo.jpg}.
```
 
For very wide or tall images, `@longimage{...}` runs Gemma 3's pan & scan
utility first, feeding the model both the full downsized image and a few
higher-resolution crops:
 
```bash
> What's happening at the bottom of @longimage{long_screenshot.png}?
```

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
| `--seqlen <N>`      | Set sequence length                               | 4096                  |
| `--tokens <N>`      | Set max number of new tokens to generate          | -1                    |
| `--topk <N>`        | Set top-k sampling value                          | 0                     |
| `--seed <N>`        | Set random seed                                   | current time          |
| `--chunk <N>`       | Set prefilling chunk size, must be >= 1           | 1024                  |
| `--temperature <F>` | Set temperature value, must be >= 0.0             | 1.0                   |
| `--topp <F>`        | Set top-p sampling value, must be 0.0 < p <= 1.0  | 1.0                   |
| `--rpen <F>`        | Set repetition penalty, must be >= 1.0            | 1.0                   |
| `--prompt <S>`      | Set input prompt, ignored if chat mode is enabled | "Once upon a time"    |
| `--threads <N>`     | Set OpenMP thread count, must be >= 1             | 8                     |      
| `--chat`            | Enable chat mode                                  | —                     |
| `--text-only`       | Disable multimodal capability                     | —                     |
| `--no-mmap`         | Disable mmap (memory mapped file)                 | —                     |
| `--verbose`         | Print model info and the perf report              | —                     |
| `--help`, `-?`      | Display this help message                         | —                     |
 
### Controls
 
| Control  | Description                              |
| -------- | ---------------------------------------- |
| `Ctrl+C` | Gracefully interrupt generation and exit |
 
### Examples
 
| Command                                                                                         |
| ----------------------------------------------------------------------------------------------- |
| `./gemma model.bin --seqlen 2048 --temperature 0.8 --chat`                                      |
| `./gemma model.bin --prompt "Hello I'm a language model," --tokens 100 --topk 50 --seed 12345`  |


## Image decoding

Image input uses [stb_image](https://github.com/nothings/stb) v2.30 and
stb_image_resize2 v2.18 by Sean Barrett (public domain / MIT), embedded in
`gemma.c` as a minified build limited to PNG and JPEG (`STBI_ONLY_PNG`,
`STBI_ONLY_JPEG`). This embedded copy was produced with ad-hoc minification
scripts plus some manual edits, so it's probably not byte-reproducible from
upstream.

Supported: PNG, baseline/progressive JPEG.

Not supported: TGA, BMP, PSD, GIF, HDR, etc.

stb_image is not designed as a hardened parser, so only load images you trust.

## Performance

Test environment:
 
- CPU: `Intel(R) Core(TM) Ultra 5 225H`
- RAM: `31.5 GB`
- OS: `Windows 11`
- Compiler: `gcc 15.2.0 x86_64-w64-mingw32`
- Compiler arguments: `-Ofast -fopenmp -march=native -mtune=native`
- Run arguments (generation): `--temperature 0 --tokens 100 --threads 10 --verbose --no-mmap`
- Run arguments (prefill): `--temperature 0 --seqlen 2048 --tokens 1 --threads 10 --verbose --no-mmap`

| Model | Base memory (GB) | KV cache / tok (KB) | Generation (tok/s) | Prefill (tok/s) |
|---|---|---|---|---|
| gemma-2-2b-fp32 | 10.790 | 208.00 | 7.53 (7.38-7.63) | 46.62 (45.69-47.19) |
| gemma-2-2b-fp32_q | 3.490 | 208.00 | 24.72 (21.84-26.47) | 143.51 (142.97-144.02) |
| gemma-3-12b-it-fp32_q | 17.340 | 768.00 | 6.52 (6.31-6.68) | 28.53 (28.50-28.58) |
| gemma-3-1b-it-fp32 | 4.060 | 52.00 | 17.75 (15.90-18.86) | 132.74 (131.13-135.87) |
| gemma-3-1b-it-fp32_q | 1.270 | 52.00 | 69.01 (68.46-69.97) | 399.20 (394.35-404.40) |
| gemma-3-270m-it-fp32 | 1.240 | 36.00 | 64.01 (62.77-64.65) | 665.30 (662.13-670.06) |
| gemma-3-270m-it-fp32_q | 0.490 | 36.00 | 208.96 (205.95-211.57) | 1421.74 (1410.15-1438.73) |
| gemma-3-4b-it-fp32 | 19.240 | 272.00 | 4.61 (3.76-5.09) | 26.62 (22.09-29.28) |
| gemma-3-4b-it-fp32_q | 7.270 | 272.00 | 17.60 (16.24-19.28) | 93.46 (90.41-95.20) |

Notes: each figure is the median of 3 runs (range in parentheses). Prefill
uses a 1000-token prompt. KV cache is fp16 and sized for all layers at full
length (no sliding-window truncation yet). `--no-mmap` is used so that load
time and page cache don't affect the numbers.


## Known limitations

- Single-machine CPU inference only, no GPU / multi-node distributed support
  yet
- The hand-written GEMM/GEMV kernels are still fairly unoptimized compared
  to BLAS or a real SIMD kernel
- Prefilling / SigLIP is significantly faster when compiled with GCC, clang /
  MSVC results in several times slower GEMM performance
- W8A8 quantization is not supported for Gemma 1 models, due to its pre-norm
  nature

## Roadmap
 
- [ ] KV cache quantization
- [ ] W4A8 quantization
- [x] Multimodal inference
- [ ] Gemma 4 architecture support
- [ ] Further optimize GEMM
- [ ] Speculative decoding
- [ ] HTTP server implementation
- [ ] Fancier CLI
- [ ] Runtime dtype dispatch

## Acknowledgements
 
- [Google Gemma](https://ai.google.dev/gemma) official models and technical reports
- [llama2.c](https://github.com/karpathy/llama2.c) for inspiring this project
- [stb](https://github.com/nothings/stb) by Sean Barrett, for the image decoder and resizer

## License

MIT, for the code in this repository. The embedded stb code is public domain / MIT
under its own terms. Model weights are not covered by this license; they are
subject to Google's [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
