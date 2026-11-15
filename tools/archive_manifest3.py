import os, hashlib

RUN = "/root/carbon/data/rl3/snipe_n1"
RL3 = "/root/carbon/data/rl3"
B1 = "/root/archive_batch1_manifest.txt"
OUT = "/root/archive_batch3_manifest.txt"

done = set()
for line in open(B1):
    line = line.rstrip("\n")
    if line:
        done.add(line.split("  ", 2)[2])

entries = []
for root, dirs, files in os.walk(RUN):
    for f in sorted(files):
        p = os.path.join(root, f)
        rel = "run_dir/" + os.path.relpath(p, RUN)
        if rel not in done:
            entries.append((rel, p))

for f in sorted(os.listdir(RL3)):
    p = os.path.join(RL3, f)
    if os.path.isfile(p) and f != "snipe_n1.log":
        entries.append(("rl3/" + f, p))

lines = []
total = 0
for rel, p in entries:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    sz = os.path.getsize(p)
    total += sz
    lines.append(f"{h.hexdigest()}  {sz}  {rel}")

with open(OUT, "w") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"files={len(lines)} total_bytes={total} manifest={OUT}")
