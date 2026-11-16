import re
import sys


def rule(width):
  return "/*" + "-" * (width - 4) + "*/"

def banner(number, title, width):
  return [
    rule(width),
    f"/* {number:02d} {title}{' ' * (width - 8 - len(title))}*/", rule(width)
  ]

def find_banners(lines):
  spans = []
  i = 0
  while i + 3 - 1 < len(lines):
    rule_re = r"^/\*-+\*/$"
    if re.match(rule_re, lines[i]) and re.match(rule_re, lines[i + 2]):
      m = re.match(r"^/\*\s*(?:\d+\s+)?(.*?)\s*\*/$", lines[i + 1])
      if m:
        spans.append((i, m.group(1).strip()))
        i += 3
        continue
    i += 1
  return spans

def main(argv):
  check = '--check' in argv[1:]
  args = [a for a in argv[1:] if not a.startswith('--')]
  if len(args) != 1:
    sys.stderr.write("invalid args\n")
    return 1

  path = args[0]
  with open(path, 'rb') as fp:
    raw = fp.read().decode('utf-8')
  crlf = "\r\n" in raw
  text = raw.replace("\r\n", "\n") if crlf else raw
  lines = text.split("\n")

  spans = find_banners(lines)
  if not spans:
    sys.stderr.write(f"toc: no section banners found in {path}\n")
    return 1

  try:
    begin = next(i for i, l in enumerate(lines) if "<toc>" in l)
    end = next(i for i, l in enumerate(lines) if "</toc>" in l and i > begin)
  except StopIteration:
    sys.stderr.write(f"toc: <toc> / </toc> markers missing in {path}\n")
    return 1

  # the block has one line per section, so its final height is known up front
  # and a single pass is enough to get the line numbers right
  delta = len(spans) - (end - begin - 1)

  out = list(lines)
  for k, (top, title) in enumerate(spans):
    out[top:top + 3] = banner(k + 1, title, 80)
  
  out[begin + 1:end] = [
    f" *   {k+1:02d} {top + 1 + delta:5d}  {title}"
    for k, (top, title) in enumerate(spans)
  ]

  for k, (top, title) in enumerate(spans):
    for line in out[top:top + 3]:
      if len(line) > 80:
        sys.stderr.write(
          f"toc: banner {k + 1:02d} is {len(line)} columns, over 80\n"
        )

  eol = "\r\n" if crlf else "\n"
  new = eol.join(out)
  if new == raw:
    if not check:
      print(f"toc: {path} is up to date")
    return 0
  if check:
    sys.stderr.write(f"toc: {path} is stale, run `make toc`\n")
    return 1
  with open(path, "wb") as fp:
    fp.write(new.encode("utf-8"))
  print(f"toc: {len(spans)} sections indexed in {path}")
  return 0


if __name__ == "__main__":
  sys.exit(main(sys.argv))
