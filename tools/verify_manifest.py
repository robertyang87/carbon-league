import os, hashlib, sys
base = sys.argv[1]; man = sys.argv[2]
ok = bad = miss = 0; total = 0
for line in open(man):
    line = line.rstrip("\n")
    if not line: continue
    md5, sz, rel = line.split("  ", 2)
    p = os.path.join(base, rel)
    if not os.path.isfile(p):
        print("MISSING", rel); miss += 1; continue
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    if h.hexdigest() == md5 and os.path.getsize(p) == int(sz):
        ok += 1; total += int(sz)
    else:
        print("MISMATCH", rel, h.hexdigest(), os.path.getsize(p)); bad += 1
print(f"OK={ok} MISMATCH={bad} MISSING={miss} total_bytes={total}")
