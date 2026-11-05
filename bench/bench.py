import argparse
import glob
import os
import re
import statistics
import subprocess
import sys
import threading

from pathlib import Path


class Capture:
  def __init__(self):
    self.out = b""
    self.err = b""

  @property
  def out_text(self):
    return self.out.decode("utf-8", "replace")

  @property
  def err_text(self):
    return self.err.decode("utf-8", "replace")


def _emit(dst, data: bytes):
  try:
    dst.write(data)
    dst.flush()
  except TypeError:
    dst.write(data.decode("utf-8", "replace"))
    dst.flush()


def run_tee(cmd, timeout=None, cwd=None, env=None):
  cmd = [str(c) for c in cmd]
  cap = Capture()
  out_chunks, err_chunks = [], []

  proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    stdin=subprocess.DEVNULL,
    bufsize=0,
    cwd=cwd,
    env=env,
  )

  def pump(stream, chunks, dst):
    fd = stream.fileno()
    try:
      while True:
        data = os.read(fd, 4096)
        if not data:
          break
        chunks.append(data)
        _emit(dst, data)
    finally:
      stream.close()

  t_out = threading.Thread(
    target=pump,
    args=(proc.stdout, out_chunks, getattr(sys.stdout, "buffer", sys.stdout)),
    daemon=True,
  )
  t_err = threading.Thread(
    target=pump,
    args=(proc.stderr, err_chunks, getattr(sys.stderr, "buffer", sys.stderr)),
    daemon=True,
  )
  t_out.start()
  t_err.start()

  try:
    proc.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
    proc.kill()
    proc.wait()
    raise
  finally:
    t_out.join()
    t_err.join()
    cap.out = b"".join(out_chunks)
    cap.err = b"".join(err_chunks)

  return cap


RE_DECODER_BUFFER = re.compile(r"decoder buffer\s*:\s*([\d.]+)\s*GB", re.IGNORECASE)
RE_DECODER_WEIGHTS = re.compile(r"decoder weights\s*:\s*([\d.]+)\s*GB", re.IGNORECASE)
RE_ENCODER_BUFFER = re.compile(r"encoder buffer\s*:\s*([\d.]+)\s*GB", re.IGNORECASE)
RE_ENCODER_WEIGHTS = re.compile(r"encoder weights\s*:\s*([\d.]+)\s*GB", re.IGNORECASE)
RE_KV_CACHE_TOK = re.compile(r"kv cache \(tok\)\s*:\s*([\d.]+)\s*KB", re.IGNORECASE)
RE_GENERATED = re.compile(
  r"generated\s+(\d+)\s+tokens?\s+in\s+([\d.]+)\s+seconds\s*\(([\d.]+)\s*tok/s\)",
  re.IGNORECASE,
)
RE_PREFILLED = re.compile(
  r"prefilled\s+(\d+)\s+tokens?\s+in\s+([\d.]+)\s+seconds\s*\(([\d.]+)\s*tok/s\)",
  re.IGNORECASE,
)


def run_gemma(binary, model, extra_args, timeout=None):
  cmd = [binary, model] + extra_args
  try:
    print()
    result = run_tee(cmd, timeout=timeout)
    return result.out_text + "\n" + result.err_text
  except subprocess.TimeoutExpired as e:
    print(f"  ! timeout running: {' '.join(cmd)}", file=sys.stderr)
    return (str(e.stdout) or "") + (str(e.stderr) or "")
  except FileNotFoundError:
    print(f"  ! binary not found: {binary}", file=sys.stderr)
    sys.exit(1)


def extract_float(pattern, text):
  m = pattern.search(text)
  return float(m.group(1)) if m else None


def extract_triplet(pattern, text):
  m = pattern.search(text)
  if not m:
    return None
  return int(m.group(1)), float(m.group(2)), float(m.group(3))


def parse_memory(text):
  decoder_buffer = extract_float(RE_DECODER_BUFFER, text) or 0.0
  decoder_weights = extract_float(RE_DECODER_WEIGHTS, text) or 0.0
  encoder_buffer = extract_float(RE_ENCODER_BUFFER, text) or 0.0
  encoder_weights = extract_float(RE_ENCODER_WEIGHTS, text) or 0.0
  base_memory_gb = decoder_buffer + decoder_weights + encoder_buffer + encoder_weights
  kv_per_tok_kb = extract_float(RE_KV_CACHE_TOK, text)
  return base_memory_gb, kv_per_tok_kb


def bench_generation(binary, model, runs, tokens, threads, log_dir):
  results = []
  base_mem = None
  kv_per_tok = None
  args = [
    "--temperature", "0",
    "--tokens", str(tokens),
    "--threads", str(threads),
    "--verbose",
    "--no-mmap",
  ]
  for i in range(runs):
    text = run_gemma(binary, model, args)
    if log_dir:
      _save_log(log_dir, model, f"gen_run{i+1}", text)
    gen = extract_triplet(RE_GENERATED, text)
    if gen is None:
      print(f"  ! could not parse generation output (run {i+1}) for {model}", file=sys.stderr)
      continue
    results.append(gen)
    if base_mem is None:
      base_mem, kv_per_tok = parse_memory(text)
  return results, base_mem, kv_per_tok


def bench_prefill(binary, model, runs, seqlen, threads, prompt, log_dir):
  results = []
  args = [
    "--temperature", "0",
    "--seqlen", str(seqlen),
    "--tokens", 1,
    "--threads", str(threads),
    "--prompt", prompt,
    "--verbose",
    "--no-mmap",
  ]
  for i in range(runs):
    text = run_gemma(binary, model, args)
    if log_dir:
      _save_log(log_dir, model, f"prefill_run{i+1}", text)
    pre = extract_triplet(RE_PREFILLED, text)
    if pre is None:
      print(f"  ! could not parse prefill output (run {i+1}) for {model}", file=sys.stderr)
      continue
    results.append(pre)
  return results


def _save_log(log_dir, model, tag, text):
  log_dir = Path(log_dir)
  log_dir.mkdir(parents=True, exist_ok=True)
  name = f"{Path(model).stem}_{tag}.log"
  (log_dir / name).write_text(text)


def fmt_tok_s(results):
  if not results:
    return "N/A"
  toks = [r[2] for r in results]
  if len(toks) == 1:
    return f"{toks[0]:.2f}"
  avg = statistics.mean(toks)
  return f"{avg:.2f} ({min(toks):.2f}-{max(toks):.2f})"


def main():
  ap = argparse.ArgumentParser(
    description="Batch benchmark gemma.c across models"
  )
  ap.add_argument("--binary", default="./gemma", help="Path to gemma binary")
  ap.add_argument(
    "--models", nargs="*", default=[], help="Explicit list of model .bin paths"
  )
  ap.add_argument(
    "--models-glob",
    default=None, help="Glob pattern for model files, e.g. 'models/*.bin'"
  )
  ap.add_argument(
    "--prompt-file",
    required=True,
    help="Path to a text file containing the ~1000-token prompt used for the prefill test",
  )
  ap.add_argument(
    "--runs", type=int, default=3, help="Repeated runs per model per test (default: 3)"
  )
  ap.add_argument(
    "--gen-tokens",
    type=int,
    default=100,
    help="tokens for generation test (default 100)",
  )
  ap.add_argument(
    "--threads",
    type=int,
    default=14,
    help="number of OpenMP threads (default 14)",
  )
  ap.add_argument(
    "--prefill-seqlen", type=int, default=2048, help="seqlen for prefill test (default 2048)"
  )
  ap.add_argument(
    "--output", default="bench_results.md", help="Output markdown file"
  )
  ap.add_argument(
    "--log-dir", default=None, help="Optional dir to dump raw per-run output for debugging"
  )
  args = ap.parse_args()

  models = list(args.models)
  if args.models_glob:
    models += sorted(glob.glob(args.models_glob))
  if not models:
    print("No models specified. Use --models or --models-glob.", file=sys.stderr)
    sys.exit(1)

  prompt_path = Path(args.prompt_file)
  if not prompt_path.exists():
    print(f"Prompt file not found: {prompt_path}", file=sys.stderr)
    sys.exit(1)
  prompt_text = prompt_path.read_text()

  rows = []
  for model in models:
    name = Path(model).name
    print(f"== Benchmarking {name} ==")

    print(f"  generation x{args.runs} ...")
    gen_results, base_mem, kv_per_tok = bench_generation(
      args.binary, model, args.runs, args.gen_tokens, args.threads,
      args.log_dir,
    )

    print(f"  prefill x{args.runs} ...")
    pre_results = bench_prefill(
      args.binary, model, args.runs, args.prefill_seqlen, args.threads,
      prompt_text, args.log_dir,
    )

    rows.append({
      "model": name,
      "base_mem_gb": base_mem,
      "kv_per_tok_kb": kv_per_tok,
      "gen_tok_s": fmt_tok_s(gen_results),
      "prefill_tok_s": fmt_tok_s(pre_results),
    })

  lines = [
    "| Model | Base memory (GB) | KV cache / tok (KB) | Generation (tok/s) | Prefill (tok/s) |",
    "|---|---|---|---|---|",
  ]
  for r in rows:
    base_mem_str = f"{r['base_mem_gb']:.3f}" if r["base_mem_gb"] is not None else "N/A"
    kv_str = f"{r['kv_per_tok_kb']:.2f}" if r["kv_per_tok_kb"] is not None else "N/A"
    lines.append(f"| {r['model']} | {base_mem_str} | {kv_str} | {r['gen_tok_s']} | {r['prefill_tok_s']} |")

  table = "\n".join(lines)
  print("\n" + table)

  Path(args.output).write_text(table + "\n")
  print(f"\nWritten to {args.output}")


if __name__ == "__main__":
  main()
