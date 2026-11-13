#!/usr/bin/env python3
"""distill_capture -- re-play the corpus-batch-1 game matrix with a
recording wrapper on the teacher side(s), capturing the teacher's OWN
belief state at every decision point.

Why: bc_v53_corpus.npz stores only *visible* carbon, but the teacher's
actions are a function of its Belief estimate (base_v53.py L710-711:
bel.update(obs) then est = bel.estimate(); after L711 the belief is
read-only for the rest of act()).  Distillation inputs must therefore
be the belief grid the DP actually planned on.  This script replays
the identical matrix / seeding / loop order as corpus_batch1_run.py
and snapshots, after each teacher act() returns (== the post-update
belief state the decision was computed from):

  belief estimate grid (225 float32) + known mask (225 bool)
  tracker quantities a deployed policy can maintain itself:
    prev_act per corpus worker slot (int8, 5 = unknown)
    seq triple (prev_n, lost, since) with rl_common.SeqState semantics
    steps since last observed own deposit (capped 60)

Alignment is proven, not assumed: every replayed game's final cash
must equal corpus_batch1_results.jsonl exactly (all randomness flows
through the same global-random seeding; the wrapper draws none).
The check summary is written to distill_capture_check.json and any
mismatch aborts before the npz is written.

Output: bc_data/distill_capture.npz with one row per corpus sample,
keyed by (file, player, step) -- joinable 1:1 onto bc_v53_corpus.npz.

Usage: python3 distill_capture.py [--only TAG] [--seeds a,b,c]
"""

import importlib.util
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "env_src", "carbon_challenge-master"))

import record_trace  # noqa: E402

RESULTS_SRC = os.path.join(HERE, "corpus_batch1_results.jsonl")
OUT = os.path.join(HERE, "bc_data", "distill_capture.npz")
CHECK_OUT = os.path.join(HERE, "distill_capture_check.json")
SIZE = 15

MATRIX = [
    ("cb1_mirror", "base_v53.py", "base_v53.py",
     "base_v53", "base_v53", range(2000, 2020), "both"),
    ("cb1_ge", "base_v53.py", "reference_gloriousevolution_v53.py",
     "base_v53", "GloriousEvolution-v53", range(2020, 2030), "policy"),
    ("cb1_shadow", "base_v53.py", "shadow_alpha_v2.py",
     "base_v53", "shadow_alpha_v2", range(2030, 2040), "policy"),
    ("cb1_v05", "base_v53.py", "policy_v0_5.py",
     "base_v53", "policy_v0_5", range(2040, 2050), "policy"),
    ("cb1_fork", "fork_v0_1.py", "base_v53.py",
     "fork_v0_1", "base_v53", range(2050, 2060), "both"),
]

_counter = [0]


def load_module_fresh(path):
    _counter[0] += 1
    name = f"dcap_{_counter[0]}_{os.getpid()}_" + os.path.splitext(
        os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_agent_fresh(path):
    mod = load_module_fresh(path)
    if hasattr(mod, "make_agent"):
        return mod.make_agent()
    return mod.agent


def _suf(wid):
    try:
        return int(str(wid).rsplit("-", 1)[1])
    except Exception:
        return 0


def _delta_act(prev_pos, cur_pos):
    """Action index (extract_bc encoding: 0 stay, 1 UP, 2 RIGHT,
    3 DOWN, 4 LEFT) that moves prev_pos -> cur_pos on the torus."""
    if prev_pos == cur_pos:
        return 0
    pr, pc = divmod(int(prev_pos), SIZE)
    cr, cc = divmod(int(cur_pos), SIZE)
    dr = (cr - pr) % SIZE
    dc = (cc - pc) % SIZE
    if dr == SIZE - 1 and dc == 0:
        return 1
    if dr == 0 and dc == 1:
        return 2
    if dr == 1 and dc == 0:
        return 3
    if dr == 0 and dc == SIZE - 1:
        return 4
    return 5  # teleport / inconsistent -- counted, should not happen


class TeacherRecorder:
    """Wraps a fresh Bot from the teacher module.  Records, per call,
    the post-update belief estimate + tracker features aligned to the
    corpus sample for (player, step).  Tracker logic mirrors exactly
    what the deployed distilled policy maintains for itself."""

    def __init__(self, mod, player):
        self.mod = mod
        self.player = player
        self.bot = None
        self.prev_pos = {}      # wid -> pos (last obs)
        self.prev_cargo = {}    # wid -> cargo (last obs)
        self.seq_prev_n = None
        self.seq_prev_rec = 0
        self.seq_last_rec = None
        self.seq_last_step = None
        self.last_dep_step = None
        self.rows = []          # per step dict of arrays
        self.errors = 0

    def _tracker(self, obs):
        step = int(obs["step"])
        me = self.player
        pl = obs["players"][me]
        my_bases = {pos for _bid, pos in pl[1].items()}
        workers = pl[2]
        wids = sorted(workers.keys(), key=_suf)
        # prev_act per corpus slot (sorted by wid suffix, cap 10)
        prev_act = np.full(10, 5, dtype=np.int8)
        for j, wid in enumerate(wids[:10]):
            if wid in self.prev_pos:
                prev_act[j] = _delta_act(self.prev_pos[wid],
                                         workers[wid][0])
        # deposit detection: own collector that stood on an own base
        # with cargo last obs and has cargo 0 now
        for wid, w in workers.items():
            if w[2] == "COLLECTOR" and float(w[1]) == 0.0 \
                    and int(w[0]) in my_bases \
                    and self.prev_cargo.get(wid, 0.0) > 0:
                self.last_dep_step = step
        # seq triple (rl_common.SeqState.seq_for semantics)
        n_now = len(wids)
        if self.seq_prev_n is None or self.seq_last_step is None \
                or step != self.seq_last_step + 1:
            seq = (float(n_now), 0.0, 60.0)
        else:
            lost = max(0, self.seq_prev_n + self.seq_prev_rec - n_now)
            since = 60.0 if self.seq_last_rec is None \
                else float(step - self.seq_last_rec)
            seq = (float(self.seq_prev_n), float(lost), since)
        since_dep = 60.0 if self.last_dep_step is None \
            else float(min(step - self.last_dep_step, 60))
        return prev_act, np.asarray(seq, dtype=np.float32), since_dep, \
            wids, workers

    def __call__(self, obs, conf):
        step = int(obs["step"])
        if step == 0 or self.bot is None:
            self.bot = self.mod.Bot()
            self.prev_pos = {}
            self.prev_cargo = {}
            self.seq_prev_n = None
            self.seq_prev_rec = 0
            self.seq_last_rec = None
            self.seq_last_step = None
            self.last_dep_step = None
        prev_act, seq, since_dep, wids, workers = self._tracker(obs)
        try:
            acts = self.bot.act(obs, conf)
        except Exception:
            self.errors += 1
            acts = {}
        bel = self.bot.belief
        est = np.asarray(bel.estimate(), dtype=np.float32)
        known = np.asarray(bel.known, dtype=bool)
        self.rows.append({
            "step": step,
            "belief": est,
            "known": known,
            "prev_act": prev_act,
            "seq": seq,
            "since_dep": np.float32(since_dep),
        })
        # tracker update (SeqState.update semantics)
        n_rec = 0
        if acts:
            my_bids = {bid for bid, _pos in
                       obs["players"][self.player][1].items()}
            for k, v in acts.items():
                if k in my_bids and v in ("RECCOLLECTOR", "RECPLANTER"):
                    n_rec += 1
        self.seq_prev_n = len(wids)
        self.seq_prev_rec = n_rec
        self.seq_last_step = step
        if n_rec > 0:
            self.seq_last_rec = step
        self.prev_pos = {wid: int(w[0]) for wid, w in workers.items()}
        self.prev_cargo = {wid: float(w[1]) for wid, w in workers.items()}
        return acts


def play_capture(job):
    seed, side = job["seed"], job["side"]
    random.seed(7919 * seed + 104729 * side + 31)
    sim = record_trace.FastCarbon(record_trace.CFG).reset(seed=seed)
    agents = [None, None]
    recorders = {}
    pol_mod = load_module_fresh(os.path.join(HERE, job["policy_file"]))
    opp_mod = load_module_fresh(os.path.join(HERE, job["opp_file"]))

    def agent_of(mod, fname):
        if hasattr(mod, "make_agent"):
            return mod.make_agent()
        return mod.agent

    teachers = job["teachers"]
    if teachers in ("policy", "both"):
        rec = TeacherRecorder(pol_mod, side)
        recorders[side] = rec
        agents[side] = rec
    else:
        agents[side] = agent_of(pol_mod, job["policy_file"])
    if teachers in ("opp", "both"):
        rec = TeacherRecorder(opp_mod, 1 - side)
        recorders[1 - side] = rec
        agents[1 - side] = rec
    else:
        agents[1 - side] = agent_of(opp_mod, job["opp_file"])

    for _k in range(300):
        acts = []
        for p in range(2):
            o = sim.obs(p)
            acts.append(agents[p](o, record_trace.CFG))
        sim.step(acts[0], acts[1])
    _frame, _st = record_trace.snapshot(sim)
    final_cash = [float(_frame["cash"][0]), float(_frame["cash"][1])]

    pol_stem = os.path.splitext(job["policy_file"])[0]
    opp_stem = os.path.splitext(job["opp_file"])[0]
    fname = f"{pol_stem}_vs_{opp_stem}_seed{seed}_side{side}.json.gz"
    out = {"tag": job["tag"], "seed": seed, "side": side, "file": fname,
           "final_cash": final_cash, "players": {}}
    for p, rec in recorders.items():
        rows = sorted(rec.rows, key=lambda r: r["step"])
        out["players"][str(p)] = {
            "belief": np.stack([r["belief"] for r in rows]),
            "known": np.stack([r["known"] for r in rows]),
            "prev_act": np.stack([r["prev_act"] for r in rows]),
            "seq": np.stack([r["seq"] for r in rows]),
            "since_dep": np.asarray([r["since_dep"] for r in rows],
                                    dtype=np.float32),
            "steps": np.asarray([r["step"] for r in rows],
                                dtype=np.int16),
            "errors": rec.errors,
        }
    return out


def _job(job):
    t0 = time.time()
    last_err = None
    for attempt in (1, 2):
        try:
            r = play_capture(job)
            r["attempts"] = attempt
            r["ok"] = True
            r["sec"] = round(time.time() - t0, 1)
            return r
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
    return {"tag": job["tag"], "seed": job["seed"], "side": job["side"],
            "ok": False, "attempts": 2, "error": last_err,
            "sec": round(time.time() - t0, 1)}


def all_jobs(only=None, seeds=None):
    jobs = []
    for tag, pf, of, pn, on, seed_range, teachers in MATRIX:
        if only and tag != only:
            continue
        for seed in seed_range:
            if seeds and seed not in seeds:
                continue
            for side in (0, 1):
                jobs.append({"tag": tag, "policy_file": pf, "opp_file": of,
                             "policy_name": pn, "opp_name": on,
                             "seed": seed, "side": side,
                             "teachers": teachers})
    return jobs


def main():
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]
    seeds = None
    if "--seeds" in sys.argv:
        seeds = {int(x) for x in
                 sys.argv[sys.argv.index("--seeds") + 1].split(",")}

    expected = {}
    with open(RESULTS_SRC) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("ok"):
                expected[(r["tag"], r["seed"], r["side"])] = \
                    [float(x) for x in r["final_cash"]]

    jobs = all_jobs(only, seeds)
    print(f"capture jobs={len(jobs)}", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(_job, j): j for j in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            if r["ok"]:
                print(f"OK  {r['tag']} seed{r['seed']} side{r['side']} "
                      f"cash={r['final_cash']} ({r['sec']}s)", flush=True)
            else:
                print(f"FAIL {r['tag']} seed{r['seed']} side{r['side']} "
                      f"{r['error']}", flush=True)

    # alignment check against the recorded corpus games
    mismatches = []
    checked = 0
    for r in results:
        if not r["ok"]:
            mismatches.append((r["tag"], r["seed"], r["side"], "FAILED"))
            continue
        exp = expected.get((r["tag"], r["seed"], r["side"]))
        if exp is None:
            mismatches.append((r["tag"], r["seed"], r["side"],
                               "NO-EXPECTED"))
            continue
        checked += 1
        if [float(x) for x in r["final_cash"]] != exp:
            mismatches.append((r["tag"], r["seed"], r["side"],
                               f"cash {r['final_cash']} != {exp}"))
    check = {"jobs": len(results), "checked": checked,
             "mismatches": [list(map(str, m)) for m in mismatches]}
    with open(CHECK_OUT, "w") as f:
        json.dump(check, f, ensure_ascii=False, indent=1)
    print(f"alignment: checked={checked} mismatches={len(mismatches)}")
    if mismatches:
        print("MISMATCHES:", mismatches[:10])
        print("ABORT: capture does not reproduce the corpus games; "
              "npz NOT written.")
        return 1

    # aggregate rows keyed to corpus samples
    files = []
    rows = {"file_idx": [], "player": [], "step": [], "belief": [],
            "known": [], "prev_act": [], "seq": [], "since_dep": []}
    total_errors = 0
    for r in sorted((x for x in results if x["ok"]),
                    key=lambda x: x["file"]):
        fidx = len(files)
        files.append(r["file"])
        for pstr, pack in sorted(r["players"].items()):
            total_errors += int(pack["errors"])
            n = len(pack["steps"])
            rows["file_idx"].append(np.full(n, fidx, dtype=np.int32))
            rows["player"].append(np.full(n, int(pstr), dtype=np.int8))
            rows["step"].append(pack["steps"])
            rows["belief"].append(pack["belief"])
            rows["known"].append(pack["known"])
            rows["prev_act"].append(pack["prev_act"])
            rows["seq"].append(pack["seq"])
            rows["since_dep"].append(pack["since_dep"])
    packed = {k: np.concatenate(v) for k, v in rows.items()}
    packed["files"] = np.asarray(files)
    np.savez_compressed(OUT, **packed)
    print(f"wrote {OUT}: rows={len(packed['step'])} "
          f"teacher_act_errors={total_errors}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
