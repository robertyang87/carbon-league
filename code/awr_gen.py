"""awr_gen -- opening games for advantage-weighted regression (AWR) of our planters / recruits.

For each seed: one GREEDY game (our net, no sampling) and K SAMPLED games (our net sampled at temperature TAU for
steps < 100), all against the same opponent, all stopped at step 100. Reward of a game (our side minus theirs) at
step 100:  cash + cargo on board + remaining tree value, tree value = 4-neighbour carbon * (1 - 0.9625^min(50-age, 30)).
Advantage of a sampled game = its reward - the greedy game's reward on the same seed (paired baseline).

Writes a corpus directory readable by cnet2_build (index.json + replays/*.json.gz, our side named AWR_NAME) and
awr.jsonl (one line per game: file, seed, k, reward, advantage).
  usage: awr_gen.py OUT_DIR OUR_AGENT.py OPP_AGENT.py SEED0 NSEEDS K TAU [WORKERS]
Our agent module must accept env overrides AWR_TAU / AWR_SEED (see exp/awr_us.py)."""
import gzip
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
AWR_NAME = "awr_us"
STEPS = 100


def _tree_value(carbon, trees):
    v = 0.0
    for pos, age in trees.values():
        r, c = divmod(int(pos), 15)
        nb = sum(carbon[((r + dr) % 15) * 15 + (c + dc) % 15] for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)))
        v += nb * (1.0 - 0.9625 ** max(0, min(50 - int(age), 30)))
    return v


def play(task):
    out, us, opp, seed, k, tau = task
    os.environ["AWR_TAU"] = str(tau if k > 0 else 0.0)
    os.environ["AWR_SEED"] = str(seed * 100 + k)
    import eval_v53 as E
    from fastsim import FastCarbon
    side = seed % 2                      # our seat alternates with the seed
    sim = FastCarbon(E.CFG).reset(seed=seed)
    if float(os.environ.get("AWR_OPP_CASH", "0")):        # ruler opponent's extra start cash (rule M stand-in: 300)
        sim.cash[1 - (seed % 2)] += float(os.environ["AWR_OPP_CASH"])
    agents = [None, None]
    agents[side] = E.load_agent_fn(os.path.join(HERE, us))
    agents[1 - side] = E.load_agent_fn(os.path.join(HERE, opp))
    steps = []

    def snap(acts):
        full = [round(float(v), 2) for v in sim.carbon]
        row = []
        for p in (0, 1):
            o = sim.obs(p)
            ob = {"step": o["step"], "players": o["players"], "carbon": [round(float(v), 2) for v in o["carbon"]], "player": p}
            if p == 0:
                ob["full_carbon"] = full
            row.append({"observation": ob, "action": acts[p] if acts else None, "status": "ACTIVE"})
        return row
    steps.append(snap(None))
    for t in range(1, STEPS + 1):
        acts = [agents[p](sim.obs(p), E.CFG) for p in (0, 1)]
        sim.step(acts[0], acts[1])
        steps.append(snap(acts))
    st = sim.get_state()
    val = []
    for p in (0, 1):
        P = st["players"][p] if "players" in st else None
        cash = float(st["cash"][p])
        cargo = sum(float(w["carbon"]) for w in st["workers"].values() if w["player"] == p)
        trees = {tid: (t_["pos"], t_["age"]) for tid, t_ in st["trees"].items() if t_["player"] == p}
        val.append(cash + cargo + _tree_value(sim.carbon, trees))
    reward = val[side] - val[1 - side]
    key = f"awr_{seed}_{k}"
    fn = os.path.join(out, "replays", key + ".json.gz")
    with gzip.open(fn, "wt") as fh:
        json.dump({"steps": steps, "info": {"TeamNames": [AWR_NAME if p == side else "opp" for p in (0, 1)]}}, fh)
    return {"key": key, "file": "replays/" + key + ".json.gz", "seed": seed, "k": k, "side": side, "reward": round(reward, 1)}


def main():
    out, us, opp, s0, ns, K, tau = sys.argv[1:8]
    W = int(sys.argv[8]) if len(sys.argv) > 8 else 8
    s0, ns, K, tau = int(s0), int(ns), int(K), float(tau)
    os.makedirs(os.path.join(out, "replays"), exist_ok=True)
    tasks = [(out, us, opp, s, k, tau) for s in range(s0, s0 + ns) for k in range(K + 1)]
    res = []
    with ProcessPoolExecutor(W) as ex:
        for r in ex.map(play, tasks, chunksize=4):
            res.append(r)
    base = {r["seed"]: r["reward"] for r in res if r["k"] == 0}
    idx = []
    with open(os.path.join(out, "awr.jsonl"), "w") as fh:
        for r in res:
            r["adv"] = round(r["reward"] - base[r["seed"]], 1)
            fh.write(json.dumps(r) + "\n")
            order = ["a", "b"] if r["side"] == 0 else ["b", "a"]     # seat 0 plays strategy order[0]
            idx.append({"id": r["key"], "file": r["file"], "steps": 300, "order": order,
                        "strategies": {"a": {"name": AWR_NAME}, "b": {"name": "opp"}},
                        "scores": {"a": 1.0, "b": 0.0}})
    json.dump(idx, open(os.path.join(out, "index.json"), "w"))
    adv = [r["adv"] for r in res if r["k"] > 0]
    import statistics as S
    print(json.dumps({"games": len(res), "seeds": ns, "K": K, "tau": tau, "adv_mean": round(S.mean(adv), 1),
                      "adv_sd": round(S.pstdev(adv), 1), "greedy_reward_mean": round(S.mean(base.values()), 1)}), flush=True)


if __name__ == "__main__":
    main()
