
WSL2 / Ubuntu 26.04.1 LTS, Intel(R) Core(TM) Ultra 5 225H (14 threads), gcc (Ubuntu 15.2.0-16ubuntu1) 15.2.0, `-O3 -ffast-math -march=native -mtune=native -fopenmp`.

Every cell: `--seqlen 16384 --threads 10 --temperature 0 --tokens 100 --no-mmap --text-only --chunk 2048`

* **prefill tok/s** - 10000-token prompt, prompt ingestion only
* **decode tok/s** - 100 tokens after a 5-token prompt ("short ctx") or after
  the 10000-token prompt ("10k ctx")

| model | params | dtype | quant | prefill tok/s | decode tok/s (short ctx) | decode tok/s (10k ctx) | ms/token (10k ctx) | n | ±% |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| gemma-3-270m | 268M | fp32 | no | 1309.7 | 68.9 | 59.8 | 16.7 | 4 | 15.2 |
| gemma-3-270m | 268M | fp32 | yes | 1857.7 | 245.9 | 152.3 | 6.6 | 4 | 21.2 |
| gemma-3-270m | 268M | fp16 | no | 1139.6 | 109.3 | 86.0 | 11.6 | 1 | 0.0 |
| gemma-3-270m | 268M | fp16 | yes | 1719.8 | 231.4 | 148.9 | 6.7 | 4 | 21.4 |
| gemma-3-270m | 268M | bf16 | no | 789.0 | 17.6 | 14.5 | 69.0 | 4 | 4.5 |
| gemma-3-270m | 268M | bf16 | yes | 1040.8 | 178.0 | 56.0 | 17.8 | 4 | 14.5 |
| gemma-3-1b-it | 1000M | fp32 | no | 298.7 | 18.9 | 18.3 | 54.7 | 9 | 7.0 |
| gemma-3-1b-it | 1000M | fp32 | yes | 572.5 | 71.9 | 62.2 | 16.1 | 4 | 9.6 |
| gemma-3-1b-it | 1000M | fp16 | no | 296.7 | 32.1 | 30.4 | 32.8 | 4 | 6.6 |
| gemma-3-1b-it | 1000M | fp16 | yes | 541.9 | 66.9 | 57.5 | 17.4 | 4 | 10.9 |
| gemma-3-1b-it | 1000M | bf16 | no | 224.9 | 4.7 | 4.3 | 232.0 | 4 | 0.6 |
| gemma-3-1b-it | 1000M | bf16 | yes | 406.3 | 55.8 | 28.8 | 34.8 | 4 | 14.0 |
| gemma-3-1b-pt | 1000M | fp32 | no | 299.9 | 18.6 | 18.3 | 54.6 | 4 | 6.2 |
| gemma-3-1b-pt | 1000M | fp32 | yes | 571.9 | 72.6 | 59.2 | 16.9 | 4 | 11.9 |
| gemma-3-1b-pt | 1000M | fp16 | no | 294.5 | 32.1 | 29.4 | 34.0 | 4 | 5.4 |
| gemma-3-1b-pt | 1000M | fp16 | yes | 540.8 | 67.0 | 55.8 | 17.9 | 4 | 8.6 |
| gemma-3-1b-pt | 1000M | bf16 | no | 225.9 | 4.7 | 4.3 | 230.9 | 4 | 1.1 |
| gemma-3-1b-pt | 1000M | bf16 | yes | 409.2 | 55.3 | 27.6 | 36.2 | 4 | 7.6 |
| gemma-2-2b | 2.6B | fp32 | no | 84.6 | 7.4 | 6.1 | 163.7 | 4 | 3.3 |
| gemma-2-2b | 2.6B | fp32 | yes | 140.6 | 27.4 | 15.3 | 65.3 | 4 | 5.6 |
| gemma-2-2b | 2.6B | fp16 | no | 84.3 | 13.0 | 9.8 | 101.6 | 4 | 6.6 |
| gemma-2-2b | 2.6B | fp16 | yes | 134.1 | 26.1 | 15.8 | 63.5 | 4 | 9.4 |
| gemma-2-2b | 2.6B | bf16 | no | 58.7 | 1.8 | 1.5 | 666.7 | 4 | 0.9 |
| gemma-2-2b | 2.6B | bf16 | yes | 87.7 | 21.8 | 6.2 | 162.6 | 9 | 10.6 |
| gemma-3-4b-it | 3.9B | fp32 | yes | 141.1 | 19.7 | 15.8 | 63.3 | 9 | 3.8 |
| gemma-3-4b-it | 3.9B | fp16 | no | 69.3 | 9.0 | 8.4 | 118.9 | 4 | 5.5 |
| gemma-3-4b-it | 3.9B | fp16 | yes | 134.3 | 18.5 | 15.9 | 62.9 | 4 | 11.8 |
| gemma-3-4b-it | 3.9B | bf16 | no | 54.5 | 1.2 | 1.2 | 854.7 | 4 | 0.3 |
| gemma-3-4b-it | 3.9B | bf16 | yes | 107.5 | 17.0 | 9.2 | 108.2 | 4 | 5.8 |
| gemma-1-7b-it | 8.5B | fp32 | yes | 42.4 | 8.0 | 3.4 | 296.7 | 1 | 0.0 |
| gemma-1-7b-it | 8.5B | fp16 | yes | 46.4 | 8.8 | 4.7 | 211.0 | 4 | 6.4 |
| gemma-1-7b-it | 8.5B | bf16 | yes | 33.8 | 8.3 | 2.2 | 458.7 | 4 | 9.5 |
| gemma-3-12b-it | 11.8B | fp16 | yes | 42.5 | 6.5 | 5.5 | 180.5 | 4 | 8.8 |
| gemma-3-12b-it | 11.8B | bf16 | yes | 35.2 | 6.0 | 3.4 | 295.9 | 4 | 5.5 |

| model | q/nq (fp32) | q/nq (fp16) | q/nq (bf16) | fp16/fp32 (nq) | fp16/fp32 (q) |
|---|---:|---:|---:|---:|---:|
| gemma-3-270m | 3.57× | 2.12× | 10.13× | 1.59× | 0.94× |
| gemma-3-1b-it | 3.81× | 2.08× | 11.99× | 1.70× | 0.93× |
| gemma-3-1b-pt | 3.91× | 2.08× | 11.89× | 1.73× | 0.92× |
| gemma-2-2b | 3.69× | 2.01× | 11.97× | 1.75× | 0.95× |
| gemma-3-4b-it | — | 2.05× | 13.71× | — | 0.94× |
| gemma-1-7b-it | — | — | — | — | 1.10× |
| gemma-3-12b-it | — | — | — | — | — |

