#!/usr/bin/env python3
"""Reproduce the benchmark section of README.md.

For every model under `models/`, every weight dtype (fp32 / fp16 / bf16) and
both the plain and the int8 export, this exports a `.bin`, runs one warm-up
round plus N measured rounds over a short and a ~10000 token prompt, and renders
the two tables that go between the BENCH markers in README.md.

Everything is resumable: results are written to `bench/results.json` after each
config, and a config that is already in there is skipped, so an interrupted run
is restarted by running the same command again.

Two practical things:

* Exports need torch. If you benchmark inside WSL -- which is where the numbers
  want to come from, see the README's Windows section -- point
  `--export-python` at an interpreter that has it, for example the Windows venv:

      python3 bench/bench.py run --export-python /mnt/d/repo/venv/Scripts/python.exe

* The exported model is deleted after each config unless `--keep-bins` is
  given, so peak disk usage is one model instead of all of them.

Typical session:

    python3 bench/bench.py run --dry-run         # what would be measured
    python3 bench/bench.py run                   # overnight
    python3 bench/bench.py render --update-readme
"""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import struct
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DTYPES = ["float32", "float16", "bfloat16"]
DTYPE_MACRO = {"float32": "FP32", "float16": "FP16", "bfloat16": "BF16"}
DTYPE_LABEL = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}
SHORT_PROMPT = "Once upon a time"
# Length of bench/prompt.txt in prefill tokens (including <bos>).
DEFAULT_LONG_TOKENS = 10000

# Benchmark arguments, mirroring what the README table was measured with.
BENCH_ARGS = [
  "--seqlen", "16384",
  "--threads", "10",
  "--temperature", "0",
  "--tokens", "100",
  "--no-mmap",
  "--text-only",
  "--chunk", "2048",
]

RE_PREFILL = re.compile(
  r"^prefilled (\d+) tokens in ([\d.]+) seconds \(([\d.]+) tok/s\)", re.M)
RE_GENERATE = re.compile(
  r"^generated (\d+) tokens in ([\d.]+) seconds \(([\d.]+) tok/s\)", re.M)
RE_PARAMS = re.compile(r"decoder params\s*:\s*(\d+)")

BENCH_BEGIN = "<!-- BENCH:BEGIN -->"
BENCH_END = "<!-- BENCH:END -->"


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def is_wsl():
  try:
    with open("/proc/version", "r", encoding="utf-8", errors="replace") as f:
      return "microsoft" in f.read().lower()
  except OSError:
    return False


def win_path(path):
  """Translate a WSL path into something a Windows binary understands."""
  if path.startswith("/mnt/") and len(path) > 6 and path[5].isalpha():
    return path[5].upper() + ":" + path[6:].replace("/", "\\")
  try:
    p = subprocess.run(["wslpath", "-w", path], capture_output=True, text=True)
    if p.returncode == 0:
      return p.stdout.strip()
  except OSError:
    pass
  return path


def human_bytes(n):
  for unit in ("B", "KB", "MB", "GB", "TB"):
    if n < 1024 or unit == "TB":
      return ("%.0f %s" if unit == "B" else "%.1f %s") % (n, unit)
    n /= 1024.0


SIGNALS = {4: "SIGILL", 6: "SIGABRT", 7: "SIGBUS", 8: "SIGFPE", 9: "SIGKILL",
           11: "SIGSEGV", 15: "SIGTERM"}


def describe_failure(returncode, out):
  """Name the exit status and keep the last thing gemma said about it.

  Without this a crash is reported as a bare "rc=-11", which says nothing
  about *why* a config died. gemma's own stderr ("error: memory allocation
  failed ...") is usually the whole answer.
  """
  if returncode < 0:
    sig = -returncode
    what = ("killed by %s" % SIGNALS[sig]) if sig in SIGNALS else (
      "killed by signal %d" % sig)
  else:
    what = "exit code %d" % returncode
  lines = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]
  if not lines:
    return what
  errors = [ln for ln in lines if ln.lower().startswith("error:")]
  return "%s: %s" % (what, (errors or lines)[-1][:200])


def bench_arg(name, default=0):
  """One integer out of BENCH_ARGS, e.g. bench_arg("--seqlen")."""
  try:
    return int(BENCH_ARGS[BENCH_ARGS.index(name) + 1])
  except (ValueError, IndexError):
    return default


def bin_arch(path):
  """Read the architecture out of the header export.py writes.

  Mirrors read_text_config() in gemma.c: "GEMA" + a 4-char dtype/quant tag,
  then big-endian fields. Returns None if the file is not a model, which is
  what a half-written export looks like.
  """
  try:
    with open(path, "rb") as f:
      head = f.read(31)
  except OSError:
    return None
  if len(head) < 31 or head[:4] != b"GEMA":
    return None
  nl, n_heads, nkv = head[8], head[9], head[10]
  (hdim, dim, mlp, q_scale, slide, img_tok) = struct.unpack(">6H", head[11:23])
  max_seq, nvocab = struct.unpack(">II", head[23:31])
  return {
    "tag": head[4:8].decode("ascii", "replace"),
    "nl": nl, "n_heads": n_heads, "nkv": nkv, "hdim": hdim, "dim": dim,
    "mlp": mlp, "q_scale": q_scale, "slide": slide, "img_tok": img_tok,
    "max_seq": max_seq, "nvocab": nvocab,
  }


def peak_memory(arch, weights, dtype, seqlen, chunk):
  """Bytes gemma will have to touch at peak for one config.

  Mirrors get_text_buffer_size() in gemma.c: the weights, the kv cache, the
  attention scores and the activation scratch. Everything but the weights is
  scaled by the dtype, because the scratch buffers use the weight type too
  (an fp32 config needs twice the memory of an fp16 one).
  """
  f = 4 if dtype == "float32" else 2
  q_sz = arch["n_heads"] * arch["hdim"]
  kv_sz = arch["nkv"] * arch["hdim"]
  kv = arch["nl"] * 2 * seqlen * kv_sz * f
  att = chunk * arch["n_heads"] * seqlen * f
  act = chunk * (2 * arch["dim"] + 2 * q_sz + 2 * kv_sz + 2 * arch["hdim"] +
                 2 * arch["mlp"]) * f + arch["nvocab"] * f
  return weights + kv + att + act


def ram_available():
  """Bytes the kernel says we can still get, or None where unknown (Windows)."""
  try:
    with open("/proc/meminfo", "r", encoding="utf-8") as f:
      for line in f:
        if line.startswith("MemAvailable:"):
          return int(line.split()[1]) * 1024
  except (OSError, ValueError, IndexError):
    pass
  return None


def describe_os():
  pretty = None
  try:
    with open("/etc/os-release", "r", encoding="utf-8", errors="replace") as f:
      for line in f:
        if line.startswith("PRETTY_NAME="):
          pretty = line.split("=", 1)[1].strip().strip('"')
  except OSError:
    pass
  if is_wsl():
    return "WSL2 / " + (pretty or "Linux")
  return pretty or ("%s %s" % (platform.system(), platform.release()))


def cpu_name():
  """The marketing name from /proc/cpuinfo, which is nicer than the arch."""
  try:
    with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as f:
      for line in f:
        if line.lower().startswith("model name"):
          return line.split(":", 1)[1].strip()
  except OSError:
    pass
  return platform.processor() or platform.machine()


def has_avx512bf16():
  """True/False on Linux, None where we can't tell (Windows/macOS)."""
  try:
    with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as f:
      flags = f.read().lower()
  except OSError:
    return None
  return "avx512bf16" in flags or "amx_bf16" in flags


def run(cmd, cwd=None, timeout=None):
  """Run a command, return (returncode, combined output)."""
  try:
    p = subprocess.run(
      cmd, cwd=cwd, timeout=timeout, capture_output=True, text=True,
      errors="replace")
    return p.returncode, (p.stdout or "") + (p.stderr or "")
  except subprocess.TimeoutExpired:
    return 124, "timeout after %ss" % timeout
  except OSError as exc:
    return 127, str(exc)


def load_results(path):
  if not os.path.isfile(path):
    return {"meta": {}, "models": {}, "runs": {}}
  with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
  data.setdefault("meta", {})
  data.setdefault("models", {})
  data.setdefault("runs", {})
  return data


def save_results(path, data):
  os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
  tmp = path + ".tmp"
  with open(tmp, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=1, sort_keys=True)
    f.write("\n")
  os.replace(tmp, path)


class CacheTrimmer(threading.Thread):
  """Drop the page cache of the model file while it is being read.

  A 17 GB model plus the 17 GB of page cache the kernel keeps on top of it is
  more than a WSL VM typically has, and the OOM killer is not gentle about it.
  `posix_fadvise` needs no privileges and keeps the resident set at roughly the
  process size. No-op where the call does not exist (Windows).
  """

  def __init__(self, path, interval=0.5):
    super().__init__(daemon=True)
    self.interval = interval
    self.stop = threading.Event()
    self.fd = None
    if hasattr(os, "posix_fadvise"):
      try:
        self.fd = os.open(path, os.O_RDONLY)
      except OSError:
        self.fd = None

  def run(self):
    if self.fd is None:
      return
    while not self.stop.wait(self.interval):
      try:
        os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_DONTNEED)  # type: ignore
      except OSError:
        return

  def close(self):
    self.stop.set()
    self.join(timeout=2)
    if self.fd is not None:
      try:
        os.close(self.fd)
      except OSError:
        pass


def peak_rss(pid):
  try:
    with open("/proc/%d/status" % pid, "r", encoding="utf-8") as f:
      for line in f:
        if line.startswith("VmHWM:"):
          return int(line.split()[1]) * 1024
  except OSError:
    pass
  return 0


# --------------------------------------------------------------------------- #
# building
# --------------------------------------------------------------------------- #

def build_binary(cc, dtype, out_path):
  """Compile gemma.c with one DTYPE, mirroring the Makefile's flags."""
  cmd = [
    cc, "-std=c11", "-O3", "-ffast-math", "-march=native", "-mtune=native",
    "-fopenmp", "-DDTYPE=" + DTYPE_MACRO[dtype],
    os.path.join(REPO, "gemma.c"), "-o", out_path,
  ]
  if os.name == "nt":
    cmd += ["-static", "-lshell32"]
  else:
    cmd += ["-lm"]
    if "clang" in os.path.basename(cc).lower():
      cmd += ["-fopenmp=libgomp"]
    elif platform.system() != "Darwin":
      cmd += ["-lgomp"]
  os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
  rc, out = run(cmd, cwd=REPO)
  if rc != 0:
    raise SystemExit("build failed for %s:\n%s" % (dtype, out))
  return out.strip()


# --------------------------------------------------------------------------- #
# exporting
# --------------------------------------------------------------------------- #

def check_export_python(python):
  rc, out = run([python, "-c", "import torch"], timeout=300)
  if rc != 0:
    raise SystemExit(
      "%s cannot import torch, so it can't export models.\n"
      "Point --export-python at an interpreter that can (e.g. the venv).\n%s"
      % (python, out.strip()))


def export_model(args, model_dir, dtype, quant, out_path):
  """Export one .bin with up to three attempts. Returns (ok, message)."""
  windows = os.name != "nt" and args.export_python.lower().endswith(".exe")
  py = args.export_python
  script = os.path.join(REPO, "export.py")
  mdir, outp = model_dir, out_path
  if windows:
    script, mdir, outp = win_path(script), win_path(mdir), win_path(out_path)

  msg = "no attempt"
  for attempt in range(1, 4):
    try:
      os.unlink(out_path)
    except OSError:
      pass
    cmd = [py, script, mdir, "-o", outp, "-d", dtype]
    if quant:
      cmd.append("-q")
    rc, out = run(cmd, cwd=REPO, timeout=args.export_timeout)
    if rc == 0 and os.path.isfile(out_path):
      return True, ""
    msg = out.strip().splitlines()[-1] if out.strip() else "rc=%d" % rc
    if attempt < 3:
      print("    export failed (attempt %d): %s" % (attempt, msg))
      time.sleep(10)
  return False, msg


def drop_page_cache(path):
  """Flush and evict the pages a fresh copy of the model left behind.

  `shutil.copyfile` leaves the whole model in the page cache, and those pages
  are dirty until writeback catches up, so they cannot be reclaimed. On a
  16 GB model that is 16 GB of unswappable memory next to the 16 GB gemma is
  about to allocate, which is how a run ends up taking the whole VM down
  instead of just failing. fsync() turns the pages clean, fadvise() drops
  them.
  """
  if not hasattr(os, "posix_fadvise"):
    return
  try:
    fd = os.open(path, os.O_RDONLY)
  except OSError:
    return
  try:
    os.fsync(fd)
  except OSError:
    pass
  try:
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)  # type: ignore
  except OSError:
    pass
  finally:
    try:
      os.close(fd)
    except OSError:
      pass


def fetch_model(args, model_dir, dtype, quant):
  """Export a model and make sure it sits on the filesystem we bench from."""
  work = os.path.join(args.work_dir, "model.bin")
  staged = os.path.join(args.export_dir, "model.bin")
  os.makedirs(args.work_dir, exist_ok=True)
  os.makedirs(args.export_dir, exist_ok=True)

  if args.skip_export:
    if not os.path.isfile(staged):
      return None, "no export at %s" % staged
  else:
    ok, msg = export_model(args, model_dir, dtype, quant, staged)
    if not ok:
      return None, "export failed: " + msg

  try:
    # Different filesystems (Windows export under WSL): copy, it is much faster
    # to read the model back from the Linux side.
    if os.stat(staged).st_dev != os.stat(args.work_dir).st_dev:
      shutil.copyfile(staged, work)
      if not args.keep_bins:
        os.unlink(staged)
    else:
      os.replace(staged, work)
    drop_page_cache(work)
  except OSError as exc:
    return None, "copy failed: %s" % exc
  return work, ""


# --------------------------------------------------------------------------- #
# benchmarking
# --------------------------------------------------------------------------- #

def run_once(args, binary, model_bin, prompt):
  """One measured run; returns a dict of measurements, or {'error': ...}."""
  cmd = [binary, model_bin, "--prompt", prompt] + BENCH_ARGS + ["--verbose"]
  t0 = time.monotonic()
  proc = subprocess.Popen(
    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    errors="replace")
  trimmer = CacheTrimmer(model_bin)
  trimmer.start()
  rss = 0
  box = {}

  def reader():
    box["out"] = proc.communicate()[0]

  th = threading.Thread(target=reader, daemon=True)
  th.start()
  while th.is_alive():
    rss = max(rss, peak_rss(proc.pid))
    if time.monotonic() - t0 > args.run_timeout:
      proc.kill()
    th.join(timeout=0.25)
  trimmer.close()

  out = box.get("out", "")
  if proc.returncode != 0:
    return {"error": describe_failure(proc.returncode, out)}
  p = RE_PREFILL.search(out)
  g = RE_GENERATE.search(out)
  if not p or not g:
    return {"error": "unparsable output"}
  m = RE_PARAMS.search(out)
  return {
    "prefill_tokens": int(p.group(1)),
    "prefill_s": float(p.group(2)),
    "prefill_tps": float(p.group(3)),
    "gen_s": float(g.group(2)),
    "gen_tps": float(g.group(3)),
    "peak_rss": rss,
    "params": int(m.group(1)) if m else 0,
  }


def cv(values):
  if len(values) < 2:
    return 0.0
  mean = statistics.mean(values)
  return 100.0 * statistics.pstdev(values) / mean if mean else 0.0


def bench_prompt(args, binary, model_bin, prompt, kind):
  """Warm-up + N rounds, plus M extra rounds when the spread is too big."""
  for i in range(2):
    warm = run_once(args, binary, model_bin, prompt)
    if "error" in warm:
      return None, warm["error"]
  
  samples = []
  for i in range(args.runs):
    rec = run_once(args, binary, model_bin, prompt)
    if "error" in rec:
      return None, rec["error"]
    samples.append(rec)
    print("    %-5s %d: prefill %6.2fs (%6.1f tok/s)   decode %6.2fs (%6.1f tok/s)"
          % (kind, i + 1, rec["prefill_s"], rec["prefill_tps"],
             rec["gen_s"], rec["gen_tps"]))
  spread = max(cv([s["prefill_tps"] for s in samples]),
               cv([s["gen_tps"] for s in samples]))
  if spread > args.noisy_cv and args.extra_runs > 0:
    print("    %-5s noisy (%.1f%% spread), %d more rounds"
          % (kind, spread, args.extra_runs))
    for i in range(args.extra_runs):
      rec = run_once(args, binary, model_bin, prompt)
      if "error" in rec:
        break
      samples.append(rec)
      print("    %-5s %d: decode %6.1f tok/s" % (kind, args.runs + i + 1,
                                                 rec["gen_tps"]))
  return samples, ""


def bench_config(args, binary, model_bin):
  short, err = bench_prompt(args, binary, model_bin, args.short_prompt, "short")
  if short is None:
    return {"status": "failed", "reason": err}
  lng, err = bench_prompt(args, binary, model_bin, args.long_prompt, "long")
  if lng is None:
    return {"status": "failed", "reason": err}
  params = next((s["params"] for s in lng + short if s.get("params")), 0)
  return {
    "status": "ok",
    "samples": {"short": short, "long": lng},
    "params": params,
    "size_bytes": os.path.getsize(model_bin),
    "peak_rss": max(s["peak_rss"] for s in lng + short),
  }


# --------------------------------------------------------------------------- #
# the sweep
# --------------------------------------------------------------------------- #

def model_entries(models_dir):
  out = []
  if not os.path.isdir(models_dir):
    return out
  for name in sorted(os.listdir(models_dir)):
    d = os.path.join(models_dir, name)
    cfg = os.path.join(d, "config.json")
    if not os.path.isfile(cfg):
      continue
    with open(cfg, "rb") as f:
      blob = json.loads(f.read().decode("utf-8"))
    out.append({
      "name": name,
      "dir": d,
      # same config.json => same architecture => the rows can be pooled
      "config_hash": hashlib.sha1(
        json.dumps(blob, sort_keys=True).encode()).hexdigest()[:12],
    })
  return out


def sources_bytes(model_dir):
  total = 0
  for fn in os.listdir(model_dir):
    if fn.endswith(".safetensors"):
      total += os.path.getsize(os.path.join(model_dir, fn))
  return total


def source_bpp(model_dir):
  """Bytes per parameter of the checkpoint on disk (bf16/f16 = 2, f32 = 4)."""
  try:
    with open(os.path.join(model_dir, "config.json"), "r",
              encoding="utf-8") as f:
      dt = json.load(f).get("torch_dtype", "bfloat16")
  except (OSError, ValueError):
    dt = "bfloat16"
  return 4 if "32" in dt else 2


def planned_size(model_dir, dtype, quant):
  src = sources_bytes(model_dir)
  bpp = source_bpp(model_dir)
  if quant:
    # int8 weights plus per-channel scales: ~0.53x the bf16 checkpoint
    return int(src * 0.53 * 2.0 / bpp)
  return int(src * (2 if dtype == "float32" else 1) * 2.0 / bpp)


def build_all(args, dtypes):
  bins = {}
  for d in dtypes:
    out = os.path.join(args.build_dir, "gemma_" + DTYPE_MACRO[d])
    bins[d] = out
    if args.skip_build:
      if not os.path.isfile(out):
        raise SystemExit("missing binary for %s: %s" % (d, out))
      continue
    print("  building %s ..." % out)
    build_binary(args.cc, d, out)
  return bins


def cmd_run(args):
  all_models = model_entries(args.models_dir)
  if not all_models:
    raise SystemExit("no models found under %s" % args.models_dir)
  models = all_models
  if args.only:
    rx = re.compile(args.only)
    models = [m for m in all_models if rx.search(m["name"])]
    if not models:
      raise SystemExit("--only matched no model")

  dtypes = [d.strip() for d in args.dtypes.split(",") if d.strip()]
  quants = {"both": [False, True], "no": [False], "yes": [True]}[args.quant]

  results = load_results(args.results)
  rc, cc_out = run([args.cc, "--version"], timeout=60)
  # Stamped on every run so that changing BENCH_ARGS or bench/prompt.txt
  # invalidates the numbers measured with the old settings instead of letting
  # them survive as "already measured".
  bench_args = " ".join(BENCH_ARGS)
  prompt_sha1 = hashlib.sha1(args.long_prompt.encode("utf-8")).hexdigest()[:12]
  results["meta"].update({
    "date": time.strftime("%Y-%m-%d"),
    "os_name": describe_os(),
    "cpu": cpu_name(),
    "host": "%s %s" % (platform.platform(), platform.machine()),
    "cores": os.cpu_count(),
    "cc": args.cc,
    "cc_version": cc_out.splitlines()[0] if rc == 0 and cc_out else args.cc,
    "avx512bf16": has_avx512bf16(),
    "args": bench_args,
    "prompt_sha1": prompt_sha1,
    "runs": args.runs,
  })

  # what a complete sweep looks like, used by render to decide whether the
  # README may be rewritten: every model on disk, not just the --only subset
  planned = ["%s|%s|%s" % (m["name"], d, "q" if q else "nq")
             for m in all_models for d in dtypes for q in quants]
  results["meta"]["planned"] = sorted(
    set(planned) | set(results["meta"].get("planned", [])))

  if args.dry_run:
    print("would measure %d configs:" % len(planned))
    for m in models:
      for d in dtypes:
        for q in quants:
          tag = "%s|%s|%s" % (m["name"], d, "q" if q else "nq")
          rec = results["runs"].get(tag)
          state = rec.get("status", "todo") if rec else "todo"
          if state == "ok" and (rec.get("args") != bench_args  # type: ignore
                                or rec.get("prompt_sha1") != prompt_sha1):  # type: ignore
            state = "stale"
          est = planned_size(m["dir"], d, q)
          note = " (over --max-model-gb)" if est > args.max_model_gb * 1024 ** 3 else ""
          print("  %-40s %-6s ~%s%s" % (tag, state, human_bytes(est), note))
    return

  print("building binaries ...")
  bins = build_all(args, dtypes)
  save_results(args.results, results)

  for m in models:
    for dtype in dtypes:
      for quant in quants:
        tag = "%s|%s|%s" % (m["name"], dtype, "q" if quant else "nq")
        prev = results["runs"].get(tag)
        # configs that produced numbers with the current settings are kept;
        # failed, skipped or stale ones are only re-tried with --force, except
        # that changing the benchmark args or the prompt always re-measures.
        if prev is not None and prev.get("status") == "ok":
          if (prev.get("args") == bench_args
              and prev.get("prompt_sha1") == prompt_sha1):
            print("[skip] %s (already measured)" % tag)
            continue
          print("[redo] %s (benchmark args or prompt changed)" % tag)
        elif prev is not None and not args.force:
          print("[skip] %s (%s, use --force to retry)"
                % (tag, prev.get("status", "?")))
          continue
        est = planned_size(m["dir"], dtype, quant)
        if est > args.max_model_gb * 1024 ** 3:
          reason = "needs ~%s, over --max-model-gb" % human_bytes(est)
          print("[skip] %s: %s" % (tag, reason))
          results["runs"][tag] = {"status": "skipped", "reason": reason}
          save_results(args.results, results)
          continue

        print("[run ] %s (~%s)" % (tag, human_bytes(est)))
        model_bin, msg = fetch_model(args, m["dir"], dtype, quant)
        if model_bin is None:
          print("  %s" % msg)
          results["runs"][tag] = {"status": "failed", "reason": msg}
          save_results(args.results, results)
          continue

        # running a config that cannot fit does not fail cleanly: the kernel
        # starts swapping and the whole VM can go down with it, losing the
        # work of every config before it. the header of the exported .bin has
        # the exact architecture, so the peak can be predicted here.
        arch = bin_arch(model_bin)
        if arch is not None:
          seqlen = bench_arg("--seqlen", 4096)
          peak = peak_memory(arch, os.path.getsize(model_bin), dtype, seqlen,
                             bench_arg("--chunk", 1024))
          if args.max_ram_gb:
            budget = args.max_ram_gb * 1024 ** 3
          else:
            avail = ram_available()
            # MemAvailable already excludes reclaimable page cache, and the
            # estimate above runs ~20% over what a config really touches, so
            # no extra fudge factor here; the swap file covers the rest
            budget = avail
          if budget and peak > budget:
            reason = "needs ~%s at --seqlen %d, over the %s memory budget" % (
              human_bytes(peak), seqlen, human_bytes(budget))
            print("[skip] %s: %s" % (tag, reason))
            results["runs"][tag] = {"status": "skipped", "reason": reason}
            save_results(args.results, results)
            if not args.keep_bins:
              try:
                os.unlink(model_bin)
              except OSError:
                pass
            continue

        try:
          rec = bench_config(args, bins[dtype], model_bin)
        finally:
          if not args.keep_bins:
            try:
              os.unlink(model_bin)
            except OSError:
              pass
        rec["args"] = bench_args
        rec["prompt_sha1"] = prompt_sha1
        results["runs"][tag] = rec
        results["models"][m["name"]] = {
          "config_hash": m["config_hash"],
          "params": rec.get("params", 0),
        }
        save_results(args.results, results)
        if rec["status"] == "ok":
          print("  done: prefill %.1f tok/s, decode %.1f / %.1f tok/s"
                % (median_of(rec, "long", "prefill_tps"),
                   median_of(rec, "short", "gen_tps"),
                   median_of(rec, "long", "gen_tps")))
        else:
          print("  failed: %s" % rec.get("reason", ""))

  print("\nresults saved to %s" % args.results)
  render(args)


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

def steady(samples):
  """Drop the first measured run (it is consistently a bit slow)."""
  return samples[1:] if len(samples) > 1 else samples


def median_of(rec, kind, key):
  v = [s[key] for s in steady(rec["samples"][kind])]
  return statistics.median(v) if v else 0.0


def gather(results):
  """family -> {(dtype, quant): [per-model sample sets]} plus metadata."""
  fams = {}
  for tag, rec in results["runs"].items():
    if rec.get("status") != "ok":
      continue
    name, dtype, quant = tag.split("|")
    info = results["models"].get(name, {})
    h = info.get("config_hash") or name
    fam = fams.setdefault(h, {"names": set(), "params": 0, "sets": {}})
    fam["names"].add(name)
    fam["params"] = max(fam["params"], rec.get("params") or 0)
    fam["sets"].setdefault((dtype, quant == "q"), []).append(rec["samples"])
  for fam in fams.values():
    fam["names"] = sorted(fam["names"])
    fam["label"] = family_label(fam["names"])
  return fams


def family_label(names):
  if len(names) == 1:
    return names[0]
  prefix = os.path.commonprefix(names).rstrip("-_")
  return prefix or names[0]


def pooled(fam, dtype, quant, kind, key):
  out = []
  for sets in fam["sets"].get((dtype, quant), []):
    out += [s[key] for s in steady(sets[kind])]
  return statistics.median(out) if out else None


def pooled_n(fam, dtype, quant):
  sets = fam["sets"].get((dtype, quant), [])
  if not sets:
    return 0
  return min(len(steady(s[k])) for s in sets for k in ("short", "long"))


def pooled_spread(fam, dtype, quant):
  vals = []
  for key in ("prefill_tps", "gen_tps"):
    for sets in fam["sets"].get((dtype, quant), []):
      for kind in ("short", "long"):
        vals.append(cv([s[key] for s in steady(sets[kind])]))
  return max(vals) if vals else 0.0


def fmt(x, digits=1):
  return "—" if x is None else ("%.*f" % (digits, x))


def fmt_params(params):
  if not params:
    return "—"
  if params >= 1e9:
    return "%.1fB" % (params / 1e9)
  return "%.0fM" % (params / 1e6)


def prompt_tokens(results):
  """Median number of prefilled tokens for the long prompt (incl. <bos>)."""
  meta = results["meta"]
  recs = [r for r in results["runs"].values() if r.get("status") == "ok"]
  sha = meta.get("prompt_sha1")
  if sha:
    current = [r for r in recs
               if r.get("args") == meta.get("args")
               and r.get("prompt_sha1") == sha]
    if current:
      recs = current
  vals = [s.get("prefill_tokens", 0)
          for r in recs for s in r["samples"].get("long", [])]
  vals = [v for v in vals if v]
  return int(statistics.median(vals)) if vals else DEFAULT_LONG_TOKENS


def ctx_label(n):
  """10000 -> "10k", 1002 -> "1k", 512 -> "512"."""
  if n >= 1000:
    return "%dk" % int(round(n / 1000.0))
  return str(n)


def md_table_head(cells, right):
  """Markdown table header plus its delimiter row.

  The delimiter row has to carry exactly one cell per header cell: GFM only
  recognises a table when the two match, and a short delimiter row makes the
  whole thing render as plain text. Deriving it from `cells` means it cannot
  drift out of sync again (the generated table had 10 header cells against 9
  delimiter cells, which is why it never rendered).
  """
  head = "| " + " | ".join(cells) + " |"
  delim = "|" + "|".join(
    "---:" if i >= len(cells) - right else "---"
    for i in range(len(cells))) + "|"
  return [head, delim]


def render_tables(results):
  fams = gather(results)
  order = sorted(fams, key=lambda h: (fams[h]["params"] or 0, fams[h]["label"]))
  ctx = ctx_label(prompt_tokens(results))

  main = md_table_head([
    "model", "params", "dtype", "quant", "prefill tok/s",
    "decode tok/s (short ctx)", "decode tok/s (%s ctx)" % ctx,
    "ms/token (%s ctx)" % ctx, "n", "±%"], right=6)
  ratio = md_table_head([
    "model", "q/nq (fp32)", "q/nq (fp16)", "q/nq (bf16)",
    "fp16/fp32 (nq)", "fp16/fp32 (q)"], right=5)

  def short_tps(fam, dtype, quant):
    return pooled(fam, dtype, quant, "short", "gen_tps")

  def r(a, b):
    return "—" if not a or not b else "%.2f×" % (a / b)

  for h in order:
    fam = fams[h]
    for dtype in DTYPES:
      for quant in (False, True):
        if not fam["sets"].get((dtype, quant)):
          continue
        pre = pooled(fam, dtype, quant, "long", "prefill_tps")
        ds = pooled(fam, dtype, quant, "short", "gen_tps")
        dl = pooled(fam, dtype, quant, "long", "gen_tps")
        main.append("| %s | %s | %s | %s | %s | %s | %s | %s | %d | %s |" % (
          fam["label"], fmt_params(fam["params"]), DTYPE_LABEL[dtype],
          "yes" if quant else "no", fmt(pre), fmt(ds), fmt(dl),
          fmt(1000.0 / dl) if dl else "—",
          pooled_n(fam, dtype, quant), fmt(pooled_spread(fam, dtype, quant))))
    ratio.append("| %s | %s | %s | %s | %s | %s |" % (
      fam["label"],
      r(short_tps(fam, "float32", True), short_tps(fam, "float32", False)),
      r(short_tps(fam, "float16", True), short_tps(fam, "float16", False)),
      r(short_tps(fam, "bfloat16", True), short_tps(fam, "bfloat16", False)),
      r(short_tps(fam, "float16", False), short_tps(fam, "float32", False)),
      r(short_tps(fam, "float16", True), short_tps(fam, "float32", True))))

  return main, ratio


def render_block(results):
  if not results["runs"]:
    raise SystemExit("no results yet — run the sweep first")
  main, ratio = render_tables(results)
  meta = results["meta"]
  n = prompt_tokens(results)
  ctx = ctx_label(n)

  out = [
    "%s, %s (%s threads), %s, `%s`."
    % (meta.get("os_name", "?"), meta.get("cpu", "?"),
       meta.get("cores", "?"), meta.get("cc_version", meta.get("cc", "?")),
       "-O3 -ffast-math -march=native -mtune=native -fopenmp"),
    "",
    "Every cell: `%s`" % meta.get("args", " ".join(BENCH_ARGS)),
    "",
    "* **prefill tok/s** - %d-token prompt, prompt ingestion only" % n,
    "* **decode tok/s** - 100 tokens after a 5-token prompt (\"short ctx\") "
    "or after",
    "  the %d-token prompt (\"%s ctx\")" % (n, ctx),
    "",
  ] + main + [""] + ratio + [""]
  block = "\n".join(out)
  check_tables(block)
  return block


def check_tables(block):
  """Fail loudly on a table whose rows disagree on the cell count.

  A single short row is enough to turn the whole table back into plain text,
  and that is invisible to whoever runs the script - it only shows up in the
  rendered markdown. Cheap to check, so check.
  """
  group = []
  for line in block.splitlines() + [""]:
    if line.lstrip().startswith("|"):
      group.append(line)
      continue
    if len(group) > 1:
      counts = sorted({ln.count("|") for ln in group})
      if len(counts) > 1:
        raise SystemExit(
          "malformed markdown table: rows have %s pipes each\n  %s"
          % (counts, group[0][:100]))
    group = []


def render(args, write_readme=True):
  results = load_results(args.results)
  block = render_block(results)

  os.makedirs(os.path.dirname(os.path.abspath(args.md)), exist_ok=True)
  with open(args.md, "w", encoding="utf-8") as f:
    f.write("\n" + block + "\n")
  print("wrote %s" % args.md)

  if not (write_readme and args.update_readme):
    return
  planned = set(results["meta"].get("planned", []))
  # only a run that actually produced numbers with the current settings counts
  # as done; skipped, failed or stale configs stay "missing" so the README is
  # not silently truncated
  meta = results["meta"]
  sha = meta.get("prompt_sha1")
  done = set(tag for tag, rec in results["runs"].items()
             if rec.get("status") == "ok"
             and (not sha
                  or (rec.get("args") == meta.get("args")
                      and rec.get("prompt_sha1") == sha)))
  if planned and not planned <= done and not args.force:
    print("README not touched: %d of %d planned configs are still missing "
          "(use --force)" % (len(planned - done), len(planned)))
    return
  update_readme(args.readme, block)


def update_readme(path, text):
  with open(path, "r", encoding="utf-8") as f:
    doc = f.read()
  i, j = doc.find(BENCH_BEGIN), doc.find(BENCH_END)
  if i < 0 or j < 0 or j < i:
    raise SystemExit("%s has no %s / %s markers" % (path, BENCH_BEGIN, BENCH_END))
  doc = doc[:i + len(BENCH_BEGIN)] + "\n" + text + "\n" + doc[j:]
  with open(path, "w", encoding="utf-8") as f:
    f.write(doc)
  print("updated %s" % path)


# --------------------------------------------------------------------------- #
# chunk sweep
# --------------------------------------------------------------------------- #

def cmd_chunk(args):
  models = {m["name"]: m for m in model_entries(args.models_dir)}
  if args.model not in models:
    raise SystemExit("unknown model %r (have: %s)"
                     % (args.model, ", ".join(sorted(models)) or "none"))
  binary = os.path.join(args.build_dir, "gemma_" + DTYPE_MACRO[args.dtype])
  if not args.skip_build:
    build_binary(args.cc, args.dtype, binary)
  elif not os.path.isfile(binary):
    raise SystemExit("missing binary: %s" % binary)

  model_bin, msg = fetch_model(args, models[args.model]["dir"], args.dtype,
                               args.quant == "yes")
  if model_bin is None:
    raise SystemExit(msg)

  rows = []
  try:
    for size in [int(s) for s in args.sizes.split(",")]:
      k = BENCH_ARGS.index("--chunk")
      old = BENCH_ARGS[k + 1]
      BENCH_ARGS[k + 1] = str(size)
      try:
        samples, err = bench_prompt(args, binary, model_bin, args.long_prompt,
                                    "chunk")
      finally:
        BENCH_ARGS[k + 1] = old
      rows.append((size, samples, err))
  finally:
    if not args.keep_bins:
      try:
        os.unlink(model_bin)
      except OSError:
        pass

  print("\n| chunk | prefill tok/s | prefill s |")
  print("|---|---:|---:|")
  for size, samples, err in rows:
    if not samples:
      print("| %d | — | %s |" % (size, err))
      continue
    tps = statistics.median([s["prefill_tps"] for s in steady(samples)])
    ntok = statistics.median([s.get("prefill_tokens", DEFAULT_LONG_TOKENS)
                              for s in steady(samples)])
    print("| %d | %.1f | %.2f |" % (size, tps, ntok / tps))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def add_common(p):
  p.add_argument("--models-dir", default=os.path.join(REPO, "models"))
  p.add_argument("--work-dir", default=None,
                 help="scratch dir holding the model being benchmarked "
                      "(default: a fast local dir — /mnt/d is slow to read)")
  p.add_argument("--export-dir",
                 default=os.path.join(REPO, "exported", ".bench"),
                 help="where export.py writes; must be reachable from "
                      "--export-python")
  p.add_argument("--export-python", default=sys.executable,
                 help="interpreter that can import torch (runs export.py)")
  p.add_argument("--build-dir", default=None,
                 help="where the per-dtype binaries are built "
                      "(default: <work-dir>/bin)")
  p.add_argument("--dtypes", default="float32,float16,bfloat16")
  p.add_argument("--quant", choices=["both", "no", "yes"], default="both")
  p.add_argument("--only", default=None, help="regex on the model dir name")
  p.add_argument("--max-model-gb", type=float, default=26.0,
                 help="skip configs whose .bin would be bigger than this")
  p.add_argument("--max-ram-gb", type=float, default=None,
                 help="skip configs whose predicted peak memory (weights plus "
                      "kv cache, attention and scratch buffers) is bigger than "
                      "this (default: MemAvailable)")
  p.add_argument("--cc", default=os.environ.get("CC", "gcc"))
  p.add_argument("--skip-build", action="store_true")
  p.add_argument("--skip-export", action="store_true",
                 help="reuse the .bin already sitting in --export-dir")
  p.add_argument("--keep-bins", action="store_true")
  p.add_argument("--dry-run", action="store_true")
  p.add_argument("--prompt-file",
                 default=os.path.join(REPO, "bench", "prompt.txt"))
  p.add_argument("--short-prompt", default=SHORT_PROMPT)
  p.add_argument("--runs", type=int, default=5)
  p.add_argument("--extra-runs", type=int, default=5)
  p.add_argument("--noisy-cv", type=float, default=3.0)
  p.add_argument("--run-timeout", type=int, default=3600)
  p.add_argument("--export-timeout", type=int, default=7200)
  p.add_argument("--results", default=os.path.join(REPO, "bench", "results.json"))
  p.add_argument("--md", default=os.path.join(REPO, "bench", "results.md"))
  p.add_argument("--readme", default=os.path.join(REPO, "README.md"))
  p.add_argument("--update-readme", action="store_true")
  p.add_argument("--force", action="store_true")
  p.add_argument("--verbose", action="store_true")


def main(argv=None):
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = ap.add_subparsers(dest="cmd")

  p_run = sub.add_parser("run", help="export + benchmark everything")
  add_common(p_run)

  p_render = sub.add_parser("render", help="re-render the tables from results")
  add_common(p_render)

  p_chunk = sub.add_parser("chunk", help="chunk-size sweep on one model")
  add_common(p_chunk)
  p_chunk.add_argument("--model", required=True)
  p_chunk.add_argument("--dtype", default="float32", choices=DTYPES)
  p_chunk.add_argument("--sizes", default="64,128,256,512,1024,2048")

  args = ap.parse_args(argv)
  if not args.cmd:
    ap.print_help()
    return 1

  if not args.work_dir:
    args.work_dir = (
      os.path.join(os.path.expanduser("~"), ".cache", "gemma-bench")
      if is_wsl() else os.path.join(REPO, "exported", ".bench"))
  if not args.build_dir:
    args.build_dir = os.path.join(args.work_dir, "bin")

  if args.cmd != "render":
    with open(args.prompt_file, "r", encoding="utf-8", errors="replace") as f:
      args.long_prompt = f.read()
    if len(args.long_prompt) < 30000:
      raise SystemExit("prompt file %s looks too short for a 10000-token "
                       "prompt (%d characters)"
                       % (args.prompt_file, len(args.long_prompt)))
    if not args.dry_run:
      check_export_python(args.export_python)
    if args.cmd == "run":
      cmd_run(args)
    else:
      cmd_chunk(args)
  else:
    render(args)
  return 0


if __name__ == "__main__":
  sys.exit(main())
