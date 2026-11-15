#!/usr/bin/env python3
"""selfplay_gen -- shardable bare-net self-play game generation for the KataGo-style generational loop.

Writes cnet2_build-compatible replay corpora: <out>/index.json + <out>/replays/*.json.gz,
in the platform steps convention (see NOTES below). One shard per invocation; merging
is just passing several shard dirs to cnet2_build / build_top.py (scan dedups by replay
file name, so shards must use disjoint seed ranges).

  python3 selfplay_gen.py --weights <npz> --seed0 991000 --seed1 991020 --out <dir> --workers 56

Player: cnet2_search.make_search_agent with the pack CONF verbatim (code/packs/
PACK_CONF_TEMPLATE.txt, 13 keys: SEARCH=False, ROLL="self", GUARD=True,
GUARD_NARROW=True, MODE="sweep", MIX=False, TEACHER=2, TEACHER_LATE=2,
LATE_STEP=240, BUDGET=0.7, HARD=0.85, ABORT_DL=0.95, START_DL=None). SEARCH off,
so this is the bare net (+ guard rules) exactly as the Echo3 pack / exam
wrappers play it. Two agent instances per game, one per side -- a single
instance cannot serve both sides: make_search_agent resets its Feat trackers
whenever the seat changes (cnet2_search.py, agent(): `st["f"].seat != me`).

NOTES -- record format, grounded in the repo (2026-11-13):
* The only file-based training-data format in the repo is the replay corpus
  consumed by cnet2_build.scan/build (index.json + replays/*.json.gz). The
  rl2_ppo_rollout "recs" are in-memory PPO trajectories tied to one adapter's
  packed tensors -- not a file format, not consumable by anything except the
  PPO trainer in-process -- so they were NOT used here.
* steps convention (verified against a real ladder replay
  952811c3...-1.json.gz): steps has 300 entries; entry t holds the
  observation with obs.step == t and the action chosen FROM obs t-1
  (entry 0: action {}, entry t>=1: first real action; last entry status
  DONE). cnet2_build pairs obs at t with the action at t+1, so this shift
  is load-bearing. Each entry is [{p0}, {p1}] with keys
  {action, reward, info, observation, status}. reward = the side's cash
  (terminal score on extinction, F01 rule); status ACTIVE / DONE.
* index entries mirror the hakimi/m3s2 corpus builder:
  {id, game, order, scores, winner, steps, file, created,
   strategies: {"a": {"id","name"}, "b": {"id","name"}}}.
  `order` maps player index -> strategy key ("a" plays side a_side).
  `scores` are keyed by "a"/"b" and hold the terminal scores (cash, or the
  F01 terminal score on extinction), so build's margin
  scores[s]-scores[1-s] is exact.
* Raw margin + U (2026-11-13 foundation 定版): every index entry ALSO
  carries "margin": {"a","b"} (raw terminal margin per strategy key,
  margin_b = -margin_a) and "u": {"a","b"} with
  U = 0.5*W + 0.5*sigmoid(M/1000), W = 1 / 0 / 0.5 for M > / = / < 0
  (CarbonFormer adoption: E[U] is the primary value target). Each replay
  file carries the same numbers per PLAYER side under a top-level
  "result": {"scores":[s0,s1], "margin":[m0,m1], "u":[u0,u1],
  "winner_side": 0|1|null, "a_side": a_side}. cnet2_build compatibility:
  scan() reads only {file, order, strategies, scores, steps} from the
  index and build_game only d["steps"] from the replay, so the extra
  fields are inert for today's builder -- the corpus is the durable
  record and U/margin are readable back from index.json / replays
  directly, with NO atanh reconstruction (the tanh saturation caveat now
  applies only to legacy corpora). BUT the built DATASET shards still
  carry only value=tanh(M/3000): dataset-level raw margin needs the
  pinpointed builder change (cnet2_build.py FIELDS + build_game's
  per-row append + trainer loader) -- deliberately NOT done here.
  Entries written by the pre-定版 generator have no margin/u keys;
  regenerate or backfill a corpus before mixing it with new entries.
* Teacher-name caveat: cnet2_build.scan only keeps sides whose name is in
  the build's teacher set. build_top.py overwrites F.TEACHERS at runtime
  from CNET2_TOP_TEACHERS ("name@row,..."). A self-play corpus is therefore
  buildable only when the build is invoked with the generating policy's name
  in that env, e.g. CNET2_TOP_TEACHERS="m3s2r1@2". The script defaults both
  sides' names to --name (default "m3s2r1"); the caller owns the name.
* Diversity: with the pack CONF the agent is fully greedy (DEFAULTS
  OPEN_T=0.0, OPEN_STEPS=50, OPEN_SEED=0) -- per-game diversity comes from
  the map seed only, exactly as in exams. --open-t/--open-steps/--open-seed
  override the opening sampler; generation diversity policy is a Stage-1
  decision, not this script's.
* Extinction: FastCarbon does not model it; the generator applies the
  original-env rule itself (same logic as cnet2_search._extinct /
  rl2_ppo_rollout._extinct) and pads the remaining entries to 300 with the
  final observation, action {}, status DONE. cnet2_build breaks per side at
  the first non-ACTIVE status, so padding is inert.
* SPGEN_STUB=1 is a LOCAL-ONLY validation path (empty actions, no net):
  it exercises recording/indexing only. Never used on fleet machines.
"""
import argparse
import datetime
import gzip
import json
import math
import os
import random
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("BC_W", "64")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_v53  # noqa: E402
from fastsim import FastCarbon  # noqa: E402
import cnet2_search as CS  # noqa: E402

CFG = eval_v53.CFG
CONF = {"SEARCH": False, "ROLL": "self", "GUARD": True, "GUARD_NARROW": True,
        "MODE": "sweep", "MIX": False, "TEACHER": 2, "TEACHER_LATE": 2,
        "LATE_STEP": 240, "BUDGET": 0.7, "HARD": 0.85, "ABORT_DL": 0.95,
        "START_DL": None}
MARK = "SPGEN-RESULT"
STUB = os.environ.get("SPGEN_STUB") == "1"


def _extinct(sim, p):
    """Original-env elimination (cnet2_search._extinct, sim-object form)."""
    if any(w.player == p for w in sim.workers.values()):
        return False
    if any(t.player == p for t in sim.trees.values()):
        return False
    if not sim.bases[p]:
        return True
    cheapest = min(sim.config.get("recCollectorCost", 30), sim.config.get("recPlanterCost", 30))
    return float(sim.cash[p]) < cheapest


def _jdefault(o):
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def _u_of_margin(m):
    """U = 0.5*W + 0.5*sigmoid(M/1000), W = 1 / 0.5 / 0 for M > / = / < 0."""
    w = 1.0 if m > 0 else (0.0 if m < 0 else 0.5)
    return 0.5 * w + 0.5 / (1.0 + math.exp(-float(m) / 1000.0))


class _Gen:
    """One worker process: two cached agent instances (one per side)."""

    def __init__(self, weights, conf, name_a, name_b):
        self.weights = weights
        self.conf = conf
        self.name_a = name_a
        self.name_b = name_b
        self.agents = None

    def _agents(self):
        if self.agents is None:
            if STUB:
                self.agents = [lambda obs, cfg: {}, lambda obs, cfg: {}]
            else:
                self.agents = [CS.make_search_agent(self.conf, weights=self.weights),
                               CS.make_search_agent(self.conf, weights=self.weights)]
        return self.agents

    def play(self, seed, outdir, record=True):
        """Play one self-play game; write replay + return the index entry.
        record=False: play only (warm-up), no file written.
        Returns (entry, steps_played, margin_a, error_or_None)."""
        a_side = seed % 2
        random.seed(eval_v53.rng_seed(seed, a_side))
        sim = FastCarbon(CFG).reset(seed=seed)
        agents = self._agents()
        repdir = os.path.join(outdir, "replays")
        os.makedirs(repdir, exist_ok=True)
        fname = f"selfplay-{seed}.json.gz"
        fpath = os.path.join(repdir, fname)
        if record and os.path.exists(fpath):
            return None, 0, 0.0, "exists"
        entries = []
        cash0 = [float(sim.cash[0]), float(sim.cash[1])]
        entries.append([{"observation": sim.obs(p), "action": {}, "reward": cash0[p],
                         "status": "ACTIVE", "info": {}} for p in (0, 1)])
        scores = None
        nsteps = 0
        err = None
        try:
            for step in range(1, 300):
                obs_pre = [sim.obs(0), sim.obs(1)]
                acts = [agents[p](obs_pre[p], CFG) for p in (0, 1)]
                pre_cash = [float(sim.cash[0]), float(sim.cash[1])]
                sim.step(acts[0], acts[1])
                nsteps = step
                ext = [_extinct(sim, 0), _extinct(sim, 1)]
                term = any(ext)
                if term:
                    pen = float(step - 300 - 1)
                    rew = [pen if ext[p] else pre_cash[p] for p in (0, 1)]
                else:
                    rew = [float(sim.cash[0]), float(sim.cash[1])]
                entries.append([{"observation": sim.obs(p), "action": acts[p],
                                 "reward": rew[p],
                                 "status": "DONE" if (term or step == 299) else "ACTIVE",
                                 "info": {}} for p in (0, 1)])
                if term:
                    scores = rew
                    break
            if scores is None:
                scores = [float(sim.cash[0]), float(sim.cash[1])]
        except Exception as e:  # noqa: BLE001 - record loudly, never silently drop a game
            err = repr(e)[:300]
            return None, nsteps, 0.0, err
        # pad to 300 entries (platform convention); inert for cnet2_build,
        # which breaks per side at the first non-ACTIVE status.
        while len(entries) < 300:
            last = entries[-1]
            entries.append([{"observation": last[p]["observation"], "action": {},
                             "reward": last[p]["reward"], "status": "DONE", "info": {}}
                            for p in (0, 1)])
        # raw margin + U per PLAYER side (foundation 定版, see docstring).
        mg_side = [float(scores[0] - scores[1]), float(scores[1] - scores[0])]
        u_side = [_u_of_margin(mg_side[0]), _u_of_margin(mg_side[1])]
        if record:
            with gzip.open(fpath, "wt", encoding="utf-8") as f:
                json.dump({"id": "selfplay", "steps": entries,
                           "result": {"scores": [float(scores[0]), float(scores[1])],
                                      "margin": mg_side, "u": u_side,
                                      "winner_side": 0 if scores[0] > scores[1] else
                                      (1 if scores[1] > scores[0] else None),
                                      "a_side": a_side}}, f, default=_jdefault)
        # strategy "a" plays side a_side; order maps player index -> key.
        order = ["a", "b"] if a_side == 0 else ["b", "a"]
        sa = scores[a_side]
        sb = scores[1 - a_side]
        entry = {"id": "selfplay", "game": seed, "order": order,
                 "scores": {"a": sa, "b": sb},
                 "margin": {"a": float(sa - sb), "b": float(sb - sa)},
                 "u": {"a": _u_of_margin(sa - sb), "b": _u_of_margin(sb - sa)},
                 "winner": "a" if sa > sb else ("b" if sb > sa else None),
                 "steps": nsteps, "file": f"replays/{fname}",
                 "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 "strategies": {"a": {"id": "", "name": self.name_a},
                                "b": {"id": "", "name": self.name_b}}}
        return entry, nsteps, sa - sb, None


_GEN = [None]
_CPU0 = [None]


def _init(weights, conf, name_a, name_b):
    _GEN[0] = _Gen(weights, conf, name_a, name_b)
    _CPU0[0] = os.times()


def _task(job):
    seed, outdir, record = job
    t0 = time.time()
    entry, nsteps, margin, err = _GEN[0].play(seed, outdir, record=record)
    cpu = os.times()
    return {"seed": seed, "entry": entry, "steps": nsteps, "margin": margin,
            "err": err, "wall": time.time() - t0,
            "cpu": (cpu.user + cpu.system) - (_CPU0[0].user + _CPU0[0].system)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--seed0", type=int, required=True)
    ap.add_argument("--seed1", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--name", default="m3s2r1")
    ap.add_argument("--name-a", default=None)
    ap.add_argument("--name-b", default=None)
    ap.add_argument("--max-games", type=int, default=20)
    ap.add_argument("--max-seconds", type=float, default=180.0)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--open-t", type=float, default=None)
    ap.add_argument("--open-steps", type=int, default=None)
    ap.add_argument("--open-seed", type=int, default=None)
    a = ap.parse_args()
    conf = dict(CONF)
    if a.open_t is not None:
        conf["OPEN_T"] = a.open_t
    if a.open_steps is not None:
        conf["OPEN_STEPS"] = a.open_steps
    if a.open_seed is not None:
        conf["OPEN_SEED"] = a.open_seed
    name_a = a.name_a or a.name
    name_b = a.name_b or a.name
    if not STUB and not os.path.exists(a.weights):
        print(f"{MARK} FAIL weights-missing {a.weights}", flush=True)
        return 1
    from concurrent.futures import ProcessPoolExecutor, as_completed
    os.makedirs(os.path.join(a.out, "replays"), exist_ok=True)
    idx_path = os.path.join(a.out, "index.json")
    done_entries = []
    if os.path.exists(idx_path):
        try:
            done_entries = json.load(open(idx_path))
        except Exception:
            done_entries = []
    have = {e["game"] for e in done_entries}
    seeds = [s for s in range(a.seed0, a.seed1) if s not in have]
    t_start = time.time()
    games = steps_tot = 0
    margins = []
    errors = []
    cpu_tot = 0.0
    budget = None
    with ProcessPoolExecutor(max_workers=a.workers,
                             initializer=_init, initargs=(a.weights, conf, name_a, name_b)) as ex:
        # warm-up: one game per worker, unrecorded seeds, untimed (agent build + numpy warm).
        if a.warmup and not STUB:
            wseeds = [a.seed0 - 1000 - i for i in range(a.workers)]
            list(ex.map(_task, [(s, a.out, False) for s in wseeds]))
        t0 = time.time()
        todo = list(seeds)
        while todo and games < a.max_games and (time.time() - t0) < a.max_seconds:
            chunk = todo[:max(1, a.workers)]
            todo = todo[len(chunk):]
            futs = [ex.submit(_task, (s, a.out, True)) for s in chunk]
            for f in as_completed(futs):
                r = f.result()
                cpu_tot += r["cpu"]
                if r["err"]:
                    errors.append({"seed": r["seed"], "err": r["err"]})
                    print(f"GAME-ERROR seed={r['seed']} {r['err']}", flush=True)
                    continue
                if r["entry"] is None:  # exists
                    continue
                done_entries.append(r["entry"])
                games += 1
                steps_tot += r["steps"]
                margins.append(r["margin"])
            if errors:
                break
    wall = time.time() - t0
    total_wall = time.time() - t_start
    if games >= a.max_games:
        budget = "games"
    elif (time.time() - t0) >= a.max_seconds:
        budget = "seconds"
    json.dump(done_entries, open(idx_path, "w"), ensure_ascii=False)
    import statistics as st
    res = {"games": games, "wall_s": round(wall, 1), "total_wall_s": round(total_wall, 1),
           "games_per_min": round(games / (wall / 60.0), 2) if wall > 0 else 0,
           "mean_steps": round(steps_tot / games, 1) if games else 0,
           "mean_margin": round(st.mean(margins), 1) if margins else None,
           "mean_u": round(st.mean([_u_of_margin(m) for m in margins]), 4) if margins else None,
           "workers": a.workers, "cpu_seconds": round(cpu_tot, 1),
           "eff_cores": round(cpu_tot / wall, 2) if wall > 0 else 0,
           "errors": errors, "budget_hit": budget, "stub": STUB,
           "weights": a.weights, "conf": {k: conf[k] for k in sorted(conf)},
           "out": a.out}
    print(f"{MARK} {json.dumps(res, ensure_ascii=False)}", flush=True)
    if errors:
        print(f"{MARK} FAIL game-errors={len(errors)}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
