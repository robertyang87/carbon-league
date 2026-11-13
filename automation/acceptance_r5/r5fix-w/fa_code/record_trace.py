"""record_trace: full-trace recorder for Carbon League local matches.

Why this exists (user directive, 2026-11-03): the eval scripts only saved
final scores, which makes post-game review (复盘) impossible -- a score
says WHO lost, never WHERE.  From now on every local eval game is recorded
step-by-step into traces/, and a game without a trace does not count.

The game loop here mirrors eval_v0_1.py exactly (same CFG built from
carbon.json defaults, same agent loading, same obs -> act -> step order),
so re-running a game with the same seed reproduces the recorded eval game
bit-for-bit (all local policies are deterministic).  Recording itself is
read-only: FastCarbon.get_state() never mutates the sim.

Trace format (JSON, gzipped), one file per game:
  traces/<policy>_vs_<opp>_seed<seed>_side<0|1>.json.gz
  {
    "meta": {format, policy, opp, tag, seed, policy_side, steps, size,
             bases {player: [[base_id, pos], ...]}, final_cash [c0, c1],
             legend},
    "frames": [                       # 301 frames; frame 0 = initial state
      {"step": 0, "cash": [80, 80], "carbon": [225 floats],
       "workers": {wid: [pos, cargo, "C"|"P"]},
       "trees":   {tid: [pos, owner_player, age]}},
      {"step": t, "acts": [{entity_id: action, ...},   # player 0's raw
                           {...}],                       # player 1's raw
       "cash": [...], "carbon": [...], "workers": {...}, "trees": {...}},
      ...                             # frame t = state AFTER step t
    ]
  }
Actions with value None are dropped (None == no action in this env).
Worker cargo is rounded to 2 dp, cell carbon is already rounded to 3 dp
by the environment; neither affects the final cash, which is exact.

Usage:
  python3 record_trace.py POLICY_FILE OPP SEED SIDE [--tag TAG]
  python3 record_trace.py --games16     # backfill diag_v1 + eval_v01
                                        # (16 juzi games), verify final
                                        # cash against eval_v0_1_results.jsonl
                                        # and write traces/INDEX.md
"""

import importlib.util
import json
import os
import sys
import gzip
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "env_src", "carbon_challenge-master"))

from fastsim import FastCarbon  # noqa: E402

_spec = json.load(open(os.path.join(
    HERE, "env_src", "carbon_challenge-master",
    "zerosum_env", "envs", "carbon", "carbon.json")))
CFG = {}
for _k, _v in _spec["configuration"].items():
    CFG[_k] = _v.get("default") if isinstance(_v, dict) else _v

TRACES_DIR = os.path.join(HERE, "traces")
RESULTS_JSONL = os.path.join(HERE, "eval_v0_1_results.jsonl")

_counter = [0]


def load_agent_fn(path, kind):
    """Identical to eval_v0_1.load_agent_fn: policies via make_agent(),
    opponents via their module-level agent()."""
    _counter[0] += 1
    name = f"rectrace_{kind}_{_counter[0]}_{os.getpid()}"
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    if kind == "policy":
        return m.make_agent()
    return m.agent


def _num(v):
    """Compact JSON number: ints stay ints, floats rounded to 3 dp."""
    v = float(v)
    if v == int(v):
        return int(v)
    return round(v, 3)


def snapshot(sim):
    st = sim.get_state()
    workers = {}
    for wid, w in st["workers"].items():
        workers[wid] = [w["pos"], _num(round(w["carbon"], 2)),
                        "C" if w["type"] == "COLLECTOR" else "P"]
    trees = {}
    for tid, t in st["trees"].items():
        trees[tid] = [t["pos"], t["player"], t["age"]]
    return {
        "step": st["step"],
        "cash": [st["cash"][0], st["cash"][1]],
        "carbon": [_num(v) for v in st["carbon"]],
        "workers": workers,
        "trees": trees,
    }, st


def record_game(policy_file, opp_kind, seed, policy_side, tag=None):
    """Play one game exactly like eval_v0_1.play_game, recording frames.

    Returns (trace_dict, final_cash_list)."""
    sim = FastCarbon(CFG).reset(seed=seed)
    policy = load_agent_fn(os.path.join(HERE, policy_file), "policy")
    opp = load_agent_fn(os.path.join(HERE, f"{opp_kind}.py"), opp_kind)
    agents = [None, None]
    agents[policy_side] = policy
    agents[1 - policy_side] = opp

    frame0, st0 = snapshot(sim)
    bases = {str(p): [[bid, pos] for bid, pos in st0["bases"][p]]
             for p in range(2)}
    frames = [frame0]
    for _ in range(300):
        acts = []
        for p in range(2):
            o = sim.obs(p)
            acts.append(agents[p](o, CFG))
        sim.step(acts[0], acts[1])
        frame, _ = snapshot(sim)
        frame["acts"] = [
            {k: v for k, v in (a or {}).items() if v is not None}
            for a in acts]
        frames.append(frame)

    final_cash = frames[-1]["cash"]
    trace = {
        "meta": {
            "format": "carbon-trace/1",
            "policy": os.path.splitext(os.path.basename(policy_file))[0],
            "opp": opp_kind,
            "tag": tag,
            "seed": seed,
            "policy_side": policy_side,
            "steps": 300,
            "size": int(CFG["size"]),
            "bases": bases,
            "final_cash": final_cash,
            "legend": {
                "workers": "wid -> [pos_index, cargo, C=COLLECTOR|P=PLANTER]",
                "trees": "tid -> [pos_index, owner_player, age]",
                "pos_index": "row * size + col, row 0 = top (obs index space)",
                "cash": "cash IS the score in this game",
            },
        },
        "frames": frames,
    }
    return trace, final_cash


def trace_filename(policy_file, opp_kind, seed, policy_side):
    pol = os.path.splitext(os.path.basename(policy_file))[0]
    return f"{pol}_vs_{opp_kind}_seed{seed}_side{policy_side}.json.gz"


def write_trace(trace, path):
    raw = json.dumps(trace, separators=(",", ":"),
                     ensure_ascii=False).encode("utf-8")
    with gzip.open(path, "wb", compresslevel=6) as f:
        f.write(raw)
    return len(raw), os.path.getsize(path)


def _job(args):
    policy_file, opp_kind, seed, side, tag = args
    trace, final_cash = record_game(policy_file, opp_kind, seed, side, tag)
    os.makedirs(TRACES_DIR, exist_ok=True)
    path = os.path.join(TRACES_DIR,
                        trace_filename(policy_file, opp_kind, seed, side))
    raw_len, gz_len = write_trace(trace, path)
    return {
        "policy": trace["meta"]["policy"], "opp": opp_kind, "tag": tag,
        "seed": seed, "side": side, "final_cash": final_cash,
        "file": os.path.basename(path), "raw_bytes": raw_len,
        "gz_bytes": gz_len,
    }


def load_expected():
    """(tag, seed, policy_side) -> (my_cash, opp_cash) from the eval log."""
    exp = {}
    with open(RESULTS_JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            exp[(r["tag"], r["seed"], r["policy_side"])] = (
                r["my_cash"], r["opp_cash"])
    return exp


def games16():
    jobs = []
    for policy_file, tag in (("policy_v1.py", "diag_v1"),
                             ("policy_v0_1.py", "eval_v01")):
        for seed in (1000, 1001, 1002, 1003):
            for side in (0, 1):
                jobs.append((policy_file, "reference_juzi", seed, side, tag))
    expected = load_expected()
    results = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_job, j): j for j in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            my = r["final_cash"][r["side"]]
            op = r["final_cash"][1 - r["side"]]
            emy, eop = expected[(r["tag"], r["seed"], r["side"])]
            r["my_cash"], r["opp_cash"] = my, op
            r["exp_my"], r["exp_opp"] = emy, eop
            r["diff_my"] = round(my - emy, 4)
            r["diff_opp"] = round(op - eop, 4)
            r["ok"] = abs(r["diff_my"]) <= 0.02 and abs(r["diff_opp"]) <= 0.02
            results.append(r)
            print(f"{r['tag']} seed{r['seed']} side{r['side']}: "
                  f"rerun {my:.2f}:{op:.2f} vs recorded {emy:.2f}:{eop:.2f} "
                  f"diff {r['diff_my']:+.2f}/{r['diff_opp']:+.2f} "
                  f"{'OK' if r['ok'] else 'MISMATCH'} "
                  f"({r['gz_bytes']/1024:.0f} KiB gz)", flush=True)
    results.sort(key=lambda r: (r["tag"], r["seed"], r["side"]))
    write_index(results)
    bad = [r for r in results if not r["ok"]]
    print(f"\n{len(results) - len(bad)}/{len(results)} games match the "
          f"recorded final scores within 0.02.")
    if bad:
        print("MISMATCHED:", [(r["tag"], r["seed"], r["side"]) for r in bad])
    return results


def write_index(results):
    lines = [
        "# 对局留痕索引（traces/INDEX）",
        "",
        "每一局本地对测都必须留痕（见上级目录 README.md 的「对测留痕规矩」）。",
        "回放：打开上级目录的 `replay_viewer.html` 选择局目即可。",
        "终分口径：我方（policy）: 对手，cash 即本游戏得分；"
        "「原记录」= eval_v0_1_results.jsonl，重跑为确定性复现，"
        "核对差 = 重跑 − 原记录。",
        "",
        "| 版本 | 对手 | seed | 先后手（我方） | 终分（我方 : 对手） | 原记录 | 核对差 | 文件 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        side_txt = "P0 先手" if r["side"] == 0 else "P1 后手"
        lines.append(
            f"| {r['policy']}（{r['tag']}） | juzi（reference_juzi） "
            f"| {r['seed']} | {side_txt} "
            f"| {r['my_cash']:.2f} : {r['opp_cash']:.2f} "
            f"| {r['exp_my']:.2f} : {r['exp_opp']:.2f} "
            f"| {r['diff_my']:+.2f} / {r['diff_opp']:+.2f} "
            f"| [{r['file']}]({r['file']}) |")
    lines.append("")
    with open(os.path.join(TRACES_DIR, "INDEX.md"), "w") as f:
        f.write("\n".join(lines))


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--games16":
        games16()
        return
    policy_file, opp_kind = sys.argv[1], sys.argv[2]
    seed, side = int(sys.argv[3]), int(sys.argv[4])
    tag = None
    if "--tag" in sys.argv:
        tag = sys.argv[sys.argv.index("--tag") + 1]
    r = _job((policy_file, opp_kind, seed, side, tag))
    print(json.dumps(r, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
