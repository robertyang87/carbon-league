import json, os

for a in ("m", "c"):
    d = f"/root/carbon/data/rl3/ppou2_{a}"
    tp = os.path.join(d, "train.jsonl")
    warm = formal = 0
    first = None
    if os.path.exists(tp):
        for line in open(tp):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("warm"):
                warm += 1
            else:
                formal += 1
                if first is None:
                    first = r
    print(a, "warm=%d formal=%d" % (warm, formal), json.dumps(first)[:400] if first else "-")
