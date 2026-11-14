import json, os
for a in ("m", "c"):
    d = f"/root/carbon/data/rl3/ppou2_{a}"
    print(f"=== ARM {a} ===")
    wp = os.path.join(d, "warmup.jsonl")
    if os.path.exists(wp):
        lines = open(wp).read().strip().splitlines()
        print("warmup lines:", len(lines), "| last:", lines[-1][:300] if lines else "-")
    tp = os.path.join(d, "train.jsonl")
    if os.path.exists(tp):
        lines = open(tp).read().strip().splitlines()
        print("train lines:", len(lines))
        if lines:
            r = json.loads(lines[0])
            print("train[0] keys:", {k: r.get(k) for k in ("it", "kl", "kl_coef", "pl", "vl", "winrate") if k in r})
    ep = os.path.join(d, "effective_params.json")
    if os.path.exists(ep):
        e = json.load(open(ep))
        keep = {k: e[k] for k in e if any(s in k for s in ("value_head", "reward", "kl", "lam", "gamma", "rscale", "utility"))}
        print("effective:", keep)
    lg = os.path.join(d, "league.json")
    if os.path.exists(lg):
        L = json.load(open(lg))
        if isinstance(L, dict):
            items = L.get("members") or L.get("pool") or L
            print("league type dict keys:", list(L)[:8])
            print("league raw:", json.dumps(L)[:400])
        else:
            print("league members:", len(L), json.dumps(L)[:300])
    print("files:", sorted(os.listdir(d)))
