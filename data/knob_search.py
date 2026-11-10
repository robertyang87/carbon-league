"""knob_search -- automatic search of the MACRO knobs (cnet2_search cfg 'MACRO'; reports/plan_macro_training.md, stage A).

Pure-net games (search off, so a game is fast), a pool of opponents, the same seeds for every configuration, each
configuration scored against the baseline (MACRO off) game by game on the same (opponent, seed, seat).
Objective: the WORST opponent's mean paired margin difference -- a configuration that only helps against one opponent
loses. Successive halving: every survivor gets more seeds, the top fraction survives.

  python3 knob_search.py OUT_DIR US_FACTORY.py name=OPP.py [name=OPP.py ...] [options]
    US_FACTORY.py   defines make(**conf_overrides) -> agent (must run with SEARCH off; called once per game)
    OPP.py          a module with agent(obs, config) at module level, or with make() -> agent
  options: --configs N (120)  --stages "40:0.25,120:0.2,400:1" (games per opponent : fraction kept)
           --seed0 S (991000)  --workers W (cpu count)  --rng R (1)  --only-report
Resumable: finished games are read back from OUT_DIR/games.jsonl. Every game is flushed as it finishes.

Per game it also records (from eval_v53's ledger) deposits, the tree business
(tree income - plant and seize spend - 30 * planters recruited), the opponent's tree-steps and the knob counters."""
import json
import os
import random
import statistics as stt
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"      # one BLAS thread per game process (set before numpy loads; workers inherit it)

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..", "code")
sys.path.insert(0, CODE)

US_TMPL = '''import importlib.util, json
_s = importlib.util.spec_from_file_location("knob_us_factory", %(factory)r); _m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)
_MAC = json.loads(%(mac)r)
_inner = _m.make(MACRO=_MAC) if _MAC else _m.make()
def agent(obs, config=None):
    a = _inner(obs, config)
    agent.last_stats = getattr(_inner, "last_stats", None) or getattr(_inner, "stats", None)
    return a
'''
OPP_TMPL = '''import importlib.util
_s = importlib.util.spec_from_file_location("knob_opp_factory", %(path)r); _m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)
_inner = _m.make() if hasattr(_m, "make") else _m.agent
def agent(obs, config=None):
    return _inner(obs, config)
'''


def sample_config(rng):
    """one random point of the knob space (None = that knob is off)"""
    c = {"PL_T1": rng.choice([40, 50, 60, 80]), "PL_T2": rng.choice([220, 240, 260])}
    c["PL_CAP_E"] = rng.choice([None, None, 3, 4, 5])
    c["PL_CAP_M"] = rng.choice([None, 1, 2, 2, 3, 3, 4])
    c["PL_CAP_L"] = rng.choice([None, 0, 1, 1, 2, 3])
    c["PRICE_M"] = rng.choice([None, 45, 55, 65, 80, 100])
    c["PRICE_L"] = rng.choice([None, 25, 35, 45, 60])
    c["REC_STOP_P"] = rng.choice([999, 999, 250, 260, 270, 280])
    c["REACT_T"] = rng.choice([0.0, 0.0, 0.2, 0.34, 0.5])
    c["REACT_S"] = rng.choice([0.0, 0.0, 3.0, 6.0])
    c["CAP_TO_CAR"] = rng.choice([True, True, False])
    return c


def _play(task):
    import eval_v53 as E
    cid, opp, a_file, b_file, seed, side = task
    r = E.play_game((f"knob{cid}", a_file, b_file, seed, side))
    L = r["ledger_pol"]; lx = r["ledger_extra"]["pol"]; ox = r["ledger_extra"]["opp"]; st = r.get("a_stats") or {}
    return dict(cid=cid, opp=opp, seed=seed, side=side, margin=r["margin"], dep=L["deposits"],
                tb=L["tree_income"] - L["plant_spend"] - L["seize_spend"] - 30.0 * lx.get("pla_rec", 0),
                tinc=L["tree_income"], pla=lx.get("pla_rec", 0), planted=lx.get("planted", 0), seized=lx.get("seized", 0),
                ots=ox.get("treesteps", 0), ts=lx.get("treesteps", 0), cs=lx.get("colsteps", 0), err=r.get("a_errors", 0),
                mac=[st.get("mac_cap", 0), st.get("mac_stop", 0), st.get("mac_price", 0)])


def main():
    a = sys.argv[1:]
    opt = {"--configs": "120", "--stages": "40:0.25,120:0.2,400:1", "--seed0": "991000", "--workers": str(os.cpu_count() or 4), "--rng": "1"}
    only = "--only-report" in a
    if only:
        a.remove("--only-report")
    for k in list(opt):
        if k in a:
            i = a.index(k); opt[k] = a[i + 1]; del a[i:i + 2]
    out, factory = a[0], os.path.abspath(a[1])
    opps = dict(x.split("=", 1) for x in a[2:])
    os.makedirs(os.path.join(out, "agents"), exist_ok=True)
    rng = random.Random(int(opt["--rng"]))
    cfp = os.path.join(out, "configs.json")
    if os.path.exists(cfp):
        configs = json.load(open(cfp))
    else:
        configs = [None] + [sample_config(rng) for _ in range(int(opt["--configs"]))]     # id 0 = baseline (MACRO off)
        json.dump(configs, open(cfp, "w"), indent=0)
    for i, c in enumerate(configs):
        p = os.path.join(out, "agents", f"us_{i}.py")
        if not os.path.exists(p):
            open(p, "w").write(US_TMPL % dict(factory=factory, mac=json.dumps(c)))
    for name, path in opps.items() if not only else ():      # a report must not touch the agent files of a running search
        open(os.path.join(out, "agents", f"opp_{name}.py"), "w").write(OPP_TMPL % dict(path=os.path.abspath(path)))
    gp = os.path.join(out, "games.jsonl")
    done = {}
    if os.path.exists(gp):
        for line in open(gp):
            g = json.loads(line); done[(g["cid"], g["opp"], g["seed"])] = g
    seed0 = int(opt["--seed0"])
    alive = list(range(1, len(configs)))
    stages = [(int(x.split(":")[0]), float(x.split(":")[1])) for x in opt["--stages"].split(",")]

    def score(cid, n):
        """-> (worst-opponent mean paired margin diff, per-opponent {name: (mean, se, n)}, ledger diffs)"""
        per = {}; led = {k: [] for k in ("dep", "tb", "ots", "pla", "planted", "cs")}
        for name in opps:
            d = []
            for s in range(seed0, seed0 + n):
                g, b = done.get((cid, name, s)), done.get((0, name, s))
                if g and b:
                    d.append(g["margin"] - b["margin"])
                    for k in led:
                        led[k].append(g[k] - b[k])
            per[name] = (stt.mean(d), stt.stdev(d) / len(d) ** 0.5 if len(d) > 1 else 0.0, len(d)) if d else (float("-inf"), 0.0, 0)
        return min(v[0] for v in per.values()), per, {k: (stt.mean(v) if v else 0.0) for k, v in led.items()}

    def table(n, ids, top=12):
        rows = sorted(((score(c, n), c) for c in ids), key=lambda x: -x[0][0])
        base_ots = stt.mean([g["ots"] for (c, _, s), g in done.items() if c == 0 and s < seed0 + n] or [1])
        print(f"\n== {n} games per opponent, {len(ids)} configs; baseline opp tree-steps {base_ots:.0f}", flush=True)
        for (obj, per, led), c in rows[:top]:
            po = "  ".join(f"{k} {v[0]:+5.0f}±{v[1]:.0f}" for k, v in per.items())
            print(f"  #{c:3d} worst {obj:+5.0f} | {po} | dep {led['dep']:+5.0f} tree-biz {led['tb']:+5.0f} planters {led['pla']:+5.1f} planted {led['planted']:+5.1f} "
                  f"car-steps {led['cs']:+5.0f} opp-tree-steps {led['ots']:+5.0f} ({led['ots'] / max(base_ots, 1):+.0%}) | {json.dumps(configs[c])}", flush=True)
        return [c for _, c in rows]

    for n, keep in stages:
        if not only:
            tasks = [(cid, name, os.path.join(out, "agents", f"us_{cid}.py"), os.path.join(out, "agents", f"opp_{name}.py"), s, s % 2)
                     for cid in [0] + alive for name in opps for s in range(seed0, seed0 + n) if (cid, name, s) not in done]
            print(f"stage {n}: {len(alive)} configs, {len(tasks)} games to play", flush=True)
            with ProcessPoolExecutor(int(opt["--workers"])) as ex, open(gp, "a") as f:
                futs = [ex.submit(_play, t) for t in tasks]
                for k, fu in enumerate(as_completed(futs)):
                    g = fu.result(); done[(g["cid"], g["opp"], g["seed"])] = g
                    f.write(json.dumps(g) + "\n"); f.flush()
                    if (k + 1) % 200 == 0:
                        print(f"  {k + 1}/{len(tasks)}", flush=True)
        errs = sum(g["err"] for g in done.values())
        if errs:
            print(f"!! {errs} agent errors recorded -- results are not trustworthy until this is 0", flush=True)
        ranked = table(n, alive)
        alive = ranked[:max(1, int(round(len(ranked) * keep)))]
    json.dump({"final": alive, "configs": {str(c): configs[c] for c in alive}}, open(os.path.join(out, "final.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
