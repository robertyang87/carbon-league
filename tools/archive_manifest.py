import os, hashlib, sys

RUN = "/root/carbon/data/rl3/snipe_n1"
LOG = "/root/carbon/data/rl3/snipe_n1.log"
ATT = "/root/ppo_attrib"
OUT = "/root/archive_batch1_manifest.txt"

entries = []  # (relpath_in_archive, abspath)

big = [
    "state_last.pt", "ckpt_96.npz", "ckpt_last.npz",
    "state_pre_termU_it74.pt", "state_pre_pool320r1_it73.pt",
    "state_pre_kl012_it87.pt", "state_pre_kl010_it92.pt",
    "state_frozen_it85.pt", "state_previous.pt",
]
for f in big:
    p = os.path.join(RUN, f)
    if os.path.isfile(p):
        entries.append(("run_dir/" + f, p))

# small records in run dir root
for f in sorted(os.listdir(RUN)):
    p = os.path.join(RUN, f)
    if os.path.isfile(p) and (f.endswith(".txt") or f.endswith(".json") or f.endswith(".jsonl")):
        entries.append(("run_dir/" + f, p))

if os.path.isfile(LOG):
    entries.append(("run_dir/snipe_n1.log", LOG))

# ppo_attrib whole tree except value-diag and __pycache__
for root, dirs, files in os.walk(ATT):
    dirs[:] = [d for d in dirs if d != "__pycache__" and d != "value-diag"]
    for f in sorted(files):
        p = os.path.join(root, f)
        rel = os.path.relpath(p, ATT)
        entries.append(("ppo_attrib/" + rel, p))

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
