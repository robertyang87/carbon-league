#!/usr/bin/env python3
"""selfplay_gen -- shardable bare-net self-play game generation for the KataGo-style generational loop.

Writes cnet2_build-compatible replay corpora: <out>/index.json + <out>/replays/*.json.gz,
in the platform steps convention (see NOTES below). One shard per invocation; merging
is just passing several shard dirs to cnet2_build / build_top.py (scan dedups by replay
file name, so shards must use disjoint seed ranges).

  python3 selfplay_gen.py --weights <npz> --seed0 991000 --seed1 991020 --out <dir> --workers 56

* --max-games is a BATCH-BOUNDARY soft budget, not an exact hard cap:
  games are dispatched in chunks of --workers, the budget is only
  re-checked between chunks, so a single run may produce up to
  workers-1 games beyond --max-games (e.g. --max-games 1 --workers 3
  can produce 3 games). Size seed ranges, not --max-games, when an
  exact game count matters.

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

F11 data-contract fixes (REVIEW_FULL_20261113, 2026-11-13):
* full_carbon (SUPERVISION ONLY): every recorded observation also carries
  "full_carbon" -- the simulator's complete carbon map at that step, in the
  field cnet2_build already reads for its carbon TARGET
  (observation["full_carbon"]). It is a training label, never an actor
  input: the agent / cnet2_feat read only the visible "carbon" field, and
  the pack runtime never sees replay files. Before this, replays carried
  only the per-player visible map, so the builder's carbon target fell
  back to the visible map (seed 0 step 0: target all-zero while the true
  map has 103 non-zero cells) while the trainer scores the full map.
* Terminal typing: every index entry (and the replay "result") carries
  "terminal": true and "end_reason": "extinction" | "horizon". A game
  only gets an entry after it actually ended, so terminal=true is a
  producer attestation; cnet2_build accepts terminal-marked games below
  its min_steps default (counted separately) instead of filtering real
  short extinction games as if truncated. Truncated/corrupt games never
  receive an entry -- they surface as errors.
* Atomic commit + recovery: a replay is written to a temp file, verified
  (gzip re-open + JSON parse), then os.replace'd into place, and the
  index is rewritten atomically after every completed chunk. Startup
  reconciliation verifies every indexed replay (existence + recorded
  sha256 + parse); a missing/corrupt indexed replay is an ERROR (dropped
  and regenerated when its seed is in range, reported as a failure
  otherwise) -- never a silent skip. Orphan replay files (no index
  entry) are verified and adopted by reconstructing the entry from the
  replay's own result block (marked "adopted": true), or removed when
  corrupt -- never a permanent "exists" deadlock.
* Recovery identity (owner review pt.4 follow-up, 2026-11-13): "already
  generated" is a claim about a CONFIGURATION, not just a seed. Every
  index entry and every replay payload carries an "identity" block:
  {weights_md5, config_hash, config, names, code_md5{eval_v53, fastsim,
  cnet2_search, selfplay_gen}}. Startup reconciliation compares each
  committed entry's identity with the CURRENT run identity; ANY
  mismatch (weights changed by one byte, config changed, code changed)
  is an identity_mismatch -- the game is NOT adopted and NOT skipped as
  done. Policy: mismatched products are stale output of another
  configuration -- they are removed, regenerated when the seed is in
  this run's range, and a hard error when it is not. The run identity
  is also written to <out>/identity.json.
* Outcome class: every entry/result declares "outcome": "normal"
  (local self-play has no platform execution faults; the builder maps
  declared fault fields of other corpora to fault_one_side/unusable).
* Policy-error accounting: the search agent swallows its own failures
  into agent.stats counters (*_errors / errors; the root fallback
  returns an empty action instead of raising). play() snapshots those
  counters before and after the game; any increase marks the game as
  contaminated: it is rejected (no replay written) and reported as an
  error with the counter delta, instead of being written as a normal
  replay with err=None.
"""
import argparse
import datetime
import gzip
import hashlib
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
sys.path.insert(0, os.path.dirname(HERE))   # code/: eval_v53/fastsim/cnet2_search live beside producer/, not in it -- bare CLI must resolve them without PYTHONPATH
import eval_v53  # noqa: E402
import fastsim as _fastsim_mod  # noqa: E402
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


def _md5_file(path):
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _run_identity(weights, conf, name_a, name_b):
    """The identity a generated game belongs to: agent weights bytes,
    the full generation config (CONF + opening overrides + names) and
    the rules/code hashes. Deterministic -- no timestamps -- so two
    runs with the same identity produce interchangeable corpora."""
    cfg = {k: conf[k] for k in sorted(conf)}
    code = {}
    for mod in (eval_v53, _fastsim_mod, CS):
        code[os.path.basename(mod.__file__)] = _md5_file(mod.__file__)
    code[os.path.basename(__file__)] = _md5_file(__file__)
    return {"weights_md5": (_md5_file(weights)
                            if weights and os.path.exists(weights) else "absent"),
            "config_hash": hashlib.sha256(
                json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest(),
            "config": cfg, "names": [name_a, name_b], "code_md5": code}


def _obs_full(sim, p):
    """sim.obs(p) plus the SUPERVISION-ONLY full carbon map (see module
    docstring): the complete simulator carbon at this step, stored under
    the key cnet2_build reads for its carbon target. The agent never
    receives this dict variant beyond its usual fields -- Feat/search
    read only the visible "carbon" list from the same dict."""
    o = sim.obs(p)
    o["full_carbon"] = [float(v) for v in sim.carbon]
    return o


def _stats_dicts(ag):
    """Every DISTINCT stats dict an agent exposes, in priority order:

    1. the ``stats`` attribute -- production make_search_agent attaches
       its counter dict to the returned function (agent.stats = stats);
    2. ``__globals__["stats"]`` -- a bare agent function used without
       the factory tail that attaches the attribute (e.g. exec'd from
       the make_search_agent AST, as in the final-acceptance in-process
       probe): its counters live in its global namespace, and reading
       only the attribute silently misses every swallowed error;
    3. closure cells holding an error-counter dict (the factory-local
       stats a returned agent closes over).

    Deduped by identity: the attribute and the closure cell are normally
    the SAME dict object for a production agent, and it must be counted
    exactly once -- this is the only counting site, so play() and the
    CLI/worker layer can never double-report the same increment."""
    out, seen = [], set()

    def _add(d):
        if isinstance(d, dict) and id(d) not in seen:
            seen.add(id(d))
            out.append(d)

    _add(getattr(ag, "stats", None))
    g = getattr(ag, "__globals__", None)
    if isinstance(g, dict):
        _add(g.get("stats"))
    for cell in getattr(ag, "__closure__", None) or ():
        try:
            v = cell.cell_contents
        except ValueError:
            continue
        if isinstance(v, dict) and any(
                k == "errors" or k.endswith("_errors") for k in v):
            _add(v)
    return out


def _agent_error_counters(agents):
    """Sum the swallowed-error counters the search agent keeps on its
    stats (keys "errors" and "*_errors", e.g. root_errors), wherever the
    agent exposes them (see _stats_dicts). Stub agents have no stats
    -> {} (no-op)."""
    tot = {}
    for ag in agents:
        for st in _stats_dicts(ag):
            for k, v in st.items():
                if (k == "errors" or k.endswith("_errors")) and isinstance(v, (int, float)):
                    tot[k] = tot.get(k, 0) + int(v)
    return tot


def _counter_delta(before, after):
    return {k: after.get(k, 0) - before.get(k, 0)
            for k in set(before) | set(after) if after.get(k, 0) - before.get(k, 0) > 0}


class _Gen:
    """One worker process: two cached agent instances (one per side)."""

    def __init__(self, weights, conf, name_a, name_b, identity=None):
        self.weights = weights
        self.conf = conf
        self.name_a = name_a
        self.name_b = name_b
        self.identity = identity
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
        entries.append([{"observation": _obs_full(sim, p), "action": {}, "reward": cash0[p],
                         "status": "ACTIVE", "info": {}} for p in (0, 1)])
        scores = None
        nsteps = 0
        err = None
        err_before = _agent_error_counters(agents)
        ended_extinct = False
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
                entries.append([{"observation": _obs_full(sim, p), "action": acts[p],
                                 "reward": rew[p],
                                 "status": "DONE" if (term or step == 299) else "ACTIVE",
                                 "info": {}} for p in (0, 1)])
                if term:
                    scores = rew
                    ended_extinct = True
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
        # F11 + final-acceptance residual 4: swallowed policy errors
        # contaminate the game. The search root fallback turns exceptions
        # into empty actions + a stats counter instead of raising, so the
        # outer try alone would write this game as a normal replay with
        # err=None (round-5 repro: root_errors=2, replay written). The
        # increment check lives HERE, in play() itself -- not only in the
        # CLI/worker layer -- so an in-process caller gets the same
        # rejection; _stats_dicts finds the counters wherever the agent
        # exposes them (attribute / __globals__ / closure).
        delta = _counter_delta(err_before, _agent_error_counters(agents))
        if delta:
            return None, nsteps, 0.0, f"policy_errors: {delta}"
        end_reason = "extinction" if ended_extinct else "horizon"
        # raw margin + U per PLAYER side (foundation 定版, see docstring).
        mg_side = [float(scores[0] - scores[1]), float(scores[1] - scores[0])]
        u_side = [_u_of_margin(mg_side[0]), _u_of_margin(mg_side[1])]
        sha = None
        nbytes = None
        if record:
            payload = {"id": "selfplay", "steps": entries,
                       "identity": self.identity,
                       "result": {"scores": [float(scores[0]), float(scores[1])],
                                  "margin": mg_side, "u": u_side,
                                  "winner_side": 0 if scores[0] > scores[1] else
                                  (1 if scores[1] > scores[0] else None),
                                  "a_side": a_side, "outcome": "normal",
                                  "identity": self.identity,
                                  "terminal": True, "end_reason": end_reason}}
            # F11: atomic commit -- temp file in the same dir, verified by
            # a full re-read, then os.replace. A crash can leave a temp
            # file (cleaned at startup) but never a torn final replay.
            tmp = os.path.join(repdir, f".{fname}.tmp-{os.getpid()}")
            with gzip.open(tmp, "wt", encoding="utf-8") as f:
                json.dump(payload, f, default=_jdefault)
            with gzip.open(tmp, "rt", encoding="utf-8") as f:
                chk = json.load(f)
            if len(chk.get("steps", [])) != 300 or "result" not in chk:
                os.remove(tmp)
                return None, nsteps, 0.0, "replay_verify_failed"
            os.replace(tmp, fpath)
            with open(fpath, "rb") as f:
                raw = f.read()
            sha = hashlib.sha256(raw).hexdigest()
            nbytes = len(raw)
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
                 "terminal": True, "end_reason": end_reason,
                 "outcome": "normal", "identity": self.identity,
                 "sha256": sha, "bytes": nbytes,
                 "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 "strategies": {"a": {"id": "", "name": self.name_a},
                                "b": {"id": "", "name": self.name_b}}}
        return entry, nsteps, sa - sb, None


_GEN = [None]
_CPU0 = [None]


def _init(weights, conf, name_a, name_b, identity=None):
    _GEN[0] = _Gen(weights, conf, name_a, name_b, identity=identity)
    _CPU0[0] = os.times()


def _task(job):
    seed, outdir, record = job
    t0 = time.time()
    entry, nsteps, margin, err = _GEN[0].play(seed, outdir, record=record)
    cpu = os.times()
    return {"seed": seed, "entry": entry, "steps": nsteps, "margin": margin,
            "err": err, "wall": time.time() - t0,
            "cpu": (cpu.user + cpu.system) - (_CPU0[0].user + _CPU0[0].system)}


def _verify_replay_file(fpath, entry=None):
    """Integrity + identity check for one committed replay. Returns
    (ok, reason). Checks existence, the recorded sha256 when the index
    entry carries one, and that the gzip parses with steps + result."""
    if not os.path.exists(fpath):
        return False, "missing_replay"
    try:
        with open(fpath, "rb") as f:
            raw = f.read()
        if entry is not None and entry.get("sha256"):
            if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                return False, "sha256_mismatch"
        d = json.loads(gzip.decompress(raw).decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        return False, f"unparseable: {repr(e)[:120]}"
    if len(d.get("steps", [])) != 300 or not isinstance(d.get("result"), dict):
        return False, "bad_replay_shape"
    return True, ""


def _entry_from_replay(outdir, fname, name_a, name_b):
    """Reconstruct an index entry for an orphan replay from the replay's
    own result block. nsteps/end_reason come from the first DONE entry
    (padding is DONE, real entries before it are ACTIVE). Names fall back
    to this run's --name values; adopted entries are marked as such."""
    fpath = os.path.join(outdir, "replays", fname)
    with gzip.open(fpath, "rt", encoding="utf-8") as f:
        d = json.load(f)
    res = d["result"]
    steps = d["steps"]
    first_done = next((i for i, e in enumerate(steps)
                       if any(s.get("status") == "DONE" for s in e)), 299)
    nsteps = first_done if first_done >= 1 else 299
    end_reason = res.get("end_reason") or ("extinction" if nsteps < 299 else "horizon")
    a_side = int(res.get("a_side", 0))
    order = ["a", "b"] if a_side == 0 else ["b", "a"]
    sa, sb = float(res["scores"][a_side]), float(res["scores"][1 - a_side])
    with open(fpath, "rb") as f:
        raw = f.read()
    seed = int(fname[len("selfplay-"):-len(".json.gz")])
    return {"id": "selfplay", "game": seed, "order": order,
            "scores": {"a": sa, "b": sb},
            "margin": {"a": float(sa - sb), "b": float(sb - sa)},
            "u": {"a": _u_of_margin(sa - sb), "b": _u_of_margin(sb - sa)},
            "winner": "a" if sa > sb else ("b" if sb > sa else None),
            "steps": nsteps, "file": f"replays/{fname}",
            "terminal": True, "end_reason": end_reason,
            "outcome": res.get("outcome", "normal"),
            "identity": d.get("identity") or res.get("identity"),
            "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
            "adopted": True,
            "created": datetime.datetime.fromtimestamp(
                os.path.getmtime(fpath), datetime.timezone.utc).isoformat(),
            "strategies": {"a": {"id": "", "name": name_a},
                           "b": {"id": "", "name": name_b}}}


def _reconcile(outdir, name_a, name_b, identity=None):
    """Startup reconciliation (F11 + identity). Returns
    (entries, dropped, adopted, mismatched):
    entries -- indexed games whose replay verified intact AND whose
               recorded identity equals the current run identity;
    dropped -- [{seed, err}] indexed entries whose replay is missing or
               corrupt (an ERROR condition, never a silent skip: the
               caller regenerates in-range seeds and fails on the rest);
    adopted -- count of orphan replays re-registered into the index;
    mismatched -- [{seed, err}] committed games (indexed or orphan)
               whose recorded identity differs from the current run
               identity (identity=None disables the check, legacy
               behaviour). A mismatched game is NEVER treated as done:
               its file is removed so an in-range seed regenerates under
               the current identity, and an out-of-range seed fails.
    Corrupt orphans and stale temp files are removed."""
    repdir = os.path.join(outdir, "replays")
    os.makedirs(repdir, exist_ok=True)
    for fn in os.listdir(repdir):
        if ".tmp-" in fn:
            try:
                os.remove(os.path.join(repdir, fn))
            except OSError:
                pass
    idx_path = os.path.join(outdir, "index.json")
    raw_entries = []
    if os.path.exists(idx_path):
        try:
            raw_entries = json.load(open(idx_path))
        except Exception:
            raw_entries = []  # corrupt index: rebuild from replays below
    entries, dropped, mismatched = [], [], []
    mismatched_seeds = set()
    indexed_files = set()
    for e in raw_entries:
        fpath = os.path.join(outdir, e.get("file", ""))
        ok, reason = _verify_replay_file(fpath, e)
        if not ok:
            dropped.append({"seed": e.get("game"), "err": f"indexed_replay_{reason}"})
            continue
        if identity is not None and e.get("identity") != identity:
            # Stale product of another configuration: not done, never
            # adopted below (the orphan pass removes the file).
            mismatched.append({"seed": e.get("game"),
                               "err": "identity_mismatch: indexed entry"})
            mismatched_seeds.add(e.get("game"))
            continue
        entries.append(e)
        indexed_files.add(os.path.basename(e["file"]))
    adopted = 0
    for fn in sorted(os.listdir(repdir)):
        if not (fn.startswith("selfplay-") and fn.endswith(".json.gz")):
            continue
        if fn in indexed_files:
            continue
        fpath = os.path.join(repdir, fn)
        ok, _reason = _verify_replay_file(fpath)
        ent = None
        if ok:
            try:
                ent = _entry_from_replay(outdir, fn, name_a, name_b)
            except Exception:  # noqa: BLE001
                ent = None
        if ent is not None and identity is not None and ent.get("identity") != identity:
            seed = ent.get("game")
            try:
                os.remove(fpath)  # stale-identity orphan: clear, seed regenerates
            except OSError:
                pass
            if seed not in mismatched_seeds:
                mismatched.append({"seed": seed,
                                   "err": "identity_mismatch: orphan replay"})
                mismatched_seeds.add(seed)
            continue
        if ent is not None:
            entries.append(ent)
            adopted += 1
            continue
        try:
            os.remove(fpath)  # corrupt orphan: clear it, seed regenerates
        except OSError:
            pass
    entries.sort(key=lambda e: e.get("game", 0))
    return entries, dropped, adopted, mismatched


def _write_index(outdir, entries):
    """Atomic index rewrite (temp + os.replace): a crash leaves either
    the previous complete index or the new one, never a torn file."""
    idx_path = os.path.join(outdir, "index.json")
    tmp = idx_path + f".tmp-{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False)
    os.replace(tmp, idx_path)


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
    ap.add_argument("--max-games", type=int, default=20,
                    help="soft budget enforced at batch boundaries only: games are dispatched in chunks of --workers, so a run may overshoot by up to workers-1 games; NOT an exact hard cap")
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
    identity = _run_identity(a.weights, conf, name_a, name_b)
    if not STUB and not os.path.exists(a.weights):
        print(f"{MARK} FAIL weights-missing {a.weights}", flush=True)
        return 1
    from concurrent.futures import ProcessPoolExecutor, as_completed
    os.makedirs(os.path.join(a.out, "replays"), exist_ok=True)
    # F11: verify what a previous (possibly crashed) run actually
    # committed before scheduling anything. Indexed-but-missing/corrupt
    # replays are dropped loudly (regenerated below when in range);
    # orphan replays are adopted, not deadlocked on "exists".
    done_entries, dropped, adopted, mismatched = _reconcile(
        a.out, name_a, name_b, identity=identity)
    if dropped:
        for d_ in dropped:
            print(f"RECONCILE-DROP seed={d_['seed']} {d_['err']}", flush=True)
    if mismatched:
        for d_ in mismatched:
            print(f"RECONCILE-MISMATCH seed={d_['seed']} {d_['err']}", flush=True)
    if adopted:
        print(f"RECONCILE-ADOPT orphans={adopted}", flush=True)
    _write_index(a.out, done_entries)
    have = {e["game"] for e in done_entries}
    seeds = [s for s in range(a.seed0, a.seed1) if s not in have]
    t_start = time.time()
    games = steps_tot = 0
    margins = []
    errors = []
    cpu_tot = 0.0
    budget = None
    with ProcessPoolExecutor(max_workers=a.workers,
                             initializer=_init,
                             initargs=(a.weights, conf, name_a, name_b, identity)) as ex:
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
                if r["entry"] is None:
                    # "exists": the replay was committed by a crashed run
                    # after reconciliation (a race), or is a leftover.
                    # Adopt it if it verifies; never deadlock, never
                    # silently skip a corrupt file.
                    fpath = os.path.join(a.out, "replays", f"selfplay-{r['seed']}.json.gz")
                    ok, reason = _verify_replay_file(fpath)
                    ent = None
                    if ok:
                        try:
                            ent = _entry_from_replay(
                                a.out, f"selfplay-{r['seed']}.json.gz", name_a, name_b)
                        except Exception:  # noqa: BLE001
                            ent = None
                    if ent is not None and ent.get("identity") != identity:
                        # Committed under another identity (race with a
                        # crashed run of a different configuration): not
                        # adoptable; clear it and fail loudly.
                        try:
                            os.remove(fpath)
                        except OSError:
                            pass
                        errors.append({"seed": r["seed"], "err": "identity_mismatch: exists race"})
                        print(f"GAME-ERROR seed={r['seed']} identity_mismatch: exists race", flush=True)
                    elif ent is not None:
                        done_entries.append(ent)
                        games += 1
                        adopted += 1
                    else:
                        try:
                            os.remove(fpath)
                        except OSError:
                            pass
                        errors.append({"seed": r["seed"], "err": f"exists_but_{reason}"})
                        print(f"GAME-ERROR seed={r['seed']} exists_but_{reason}", flush=True)
                    continue
                done_entries.append(r["entry"])
                games += 1
                steps_tot += r["steps"]
                margins.append(r["margin"])
            # F11: incremental index transaction -- a crash after this
            # chunk loses nothing that was already committed.
            done_entries.sort(key=lambda e: e.get("game", 0))
            _write_index(a.out, done_entries)
            if errors:
                break
    wall = time.time() - t0
    total_wall = time.time() - t_start
    if games >= a.max_games:
        budget = "games"
    elif (time.time() - t0) >= a.max_seconds:
        budget = "seconds"
    _write_index(a.out, done_entries)
    # F11: an indexed replay that was missing/corrupt and whose seed is
    # OUTSIDE this run's range could not be repaired here; it stays an
    # error (in-range drops were regenerated above and are only failures
    # if their regeneration also failed, which already landed in errors).
    have = {e["game"] for e in done_entries}
    for d_ in dropped + mismatched:
        if d_["seed"] not in have and not (a.seed0 <= (d_["seed"] or -1) < a.seed1):
            errors.append({"seed": d_["seed"], "err": d_["err"] + " (out of repair range)"})
    # The run identity is part of the corpus record (entries carry it
    # individually too); written atomically like the index.
    id_tmp = os.path.join(a.out, f"identity.json.tmp-{os.getpid()}")
    with open(id_tmp, "w", encoding="utf-8") as f:
        json.dump({"identity": identity}, f, ensure_ascii=False)
    os.replace(id_tmp, os.path.join(a.out, "identity.json"))
    import statistics as st
    res = {"games": games, "wall_s": round(wall, 1), "total_wall_s": round(total_wall, 1),
           "games_per_min": round(games / (wall / 60.0), 2) if wall > 0 else 0,
           "mean_steps": round(steps_tot / games, 1) if games else 0,
           "mean_margin": round(st.mean(margins), 1) if margins else None,
           "mean_u": round(st.mean([_u_of_margin(m) for m in margins]), 4) if margins else None,
           "workers": a.workers, "cpu_seconds": round(cpu_tot, 1),
           "eff_cores": round(cpu_tot / wall, 2) if wall > 0 else 0,
           "errors": errors, "budget_hit": budget, "stub": STUB,
           "adopted": adopted, "identity_mismatches": len(mismatched),
           "identity": identity,
           "repaired_drops": sum(1 for d_ in dropped if d_["seed"] in have),
           "weights": a.weights, "conf": {k: conf[k] for k in sorted(conf)},
           "out": a.out}
    print(f"{MARK} {json.dumps(res, ensure_ascii=False)}", flush=True)
    if errors:
        print(f"{MARK} FAIL game-errors={len(errors)}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
