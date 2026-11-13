"""cnet2_build -- behaviour-cloning dataset for cnet2 from top-bot replays.

Scans replay dirs (index.json + replays/*.json.gz), dedups games by file name
(id-game), keeps the sides played by the teacher strategies (cnet2_feat.TEACHERS),
replays each game's observations in order through cnet2_feat.Feat (history +
belief), pairs the observation at t with the action at t+1, and computes the
autoregressive per-unit features under teacher forcing.

Output dir: one .npy per field (memmap-able) + meta.json.
Usage: python cnet2_build.py OUT_DIR REPLAY_DIR [REPLAY_DIR ...]

C02 contract: a per-game load/process failure fails the whole build (exit 1)
and publishes nothing, unless CNET2_ALLOW_PARTIAL=1, which publishes a
dataset whose meta.json declares failed_games (original game id + reason),
partial=true and valid_games (games contributing >= 1 row). A legitimate
zero-row game (the step window excludes every row) is NOT a failure, but it
is not a valid game either. main() raises SystemExit(1) on failure so
wrappers such as build_top.py propagate the failure through their own
exit code instead of silently reporting success.

N03 contract (REVIEW_FIXES_20261113_R2): an indexed replay that PASSES the
eligibility filters (steps / teacher sides / ours-corpus rules) but whose
file is missing is a failure like any other -- it is recorded in
failed_games with reason="missing_replay" and follows the same default
exit-1 / CNET2_ALLOW_PARTIAL rules. Games excluded by those filters are
legitimate filtering, not failures; scan counts them separately
(filtered_games, by reason) and meta.json declares those counts.

P3 publish discipline: merged arrays and meta.json are written to a
temporary directory inside the output dir first and moved into place with
os.replace (meta.json last, as the commit marker) only after every array
is complete. A mid-merge failure therefore cannot leave half-overwritten
arrays behind a stale meta; the previously published product is untouched.

F05 generation publishing (REVIEW_FULL_20261113): the temp-dir merge only
protected the merge phase; the per-field os.replace sequence could still
leave a MIXED product (new maps=2 rows beside old y=6 rows behind an old
meta) for a concurrent reader or after an injected mid-publish failure,
and a stale derived file (w.npy from a weighted build) survived an
unweighted rebuild in the same directory and was loaded by existence.
Datasets are now published as immutable GENERATIONS:
  out/generations/gen-<id>/   one complete product: <field>.npy files,
                              meta.json and manifest.json (field list,
                              per-file sha256, row count, cfg_fp, build
                              declaration), written in a staging dir and
                              renamed into place in one operation;
  out/current.json            the pointer, switched LAST with a single
                              os.replace after the generation is complete
                              and the legacy flat view has been refreshed.
Consumers should resolve the generation BEFORE opening any array --
resolve_dataset(out) / `python cnet2_build.py --resolve OUT_DIR` --
which pins one generation directory and (by default) verifies every
manifest hash; files not declared in the manifest never enter the
product. The flat <field>.npy + meta.json view in `out` itself is kept
as a legacy mirror for existing consumers (replica_chain.sh deletes its
DS dir before every build and consumes serially; set_uw.py adds its
derived w.npy/uw.npy there, outside the generation manifest), refreshed
from the new generation before the pointer switch. On a C02 abort
nothing is published: the pointer and the previous generation stay
intact, the flat meta.json is removed so the aborted run cannot be
mistaken for a fresh product, and BUILD_FAILED.json records the abort.

Derived-layer contract (owner review pt.4, 2026-11-13): post-processing
(set_uw.py and successors) must NEVER write into a published generation
directory -- generations are immutable once sealed. The sanctioned path
is register_derived(out, files, params): derived files (w.npy / uw.npy)
live in the dataset ROOT (the flat view), and register_derived records a
self-describing sidecar (out/derived.json, format "cnet2-derived/1"):
per file {field, dtype, rows, shape, sha256, base_generation, params},
where base_generation is the generation the derived rows align with.
The flat meta.json carries the same declaration under "derived_layer",
and every build publish refreshes that declaration for pre-existing
derived files. load_generation() only ever reads the generation
manifest's own fields -- derived-layer files are never part of a
generation product, whether they sit in the root or (illegally) inside
a generation dir. assert_not_in_generation(path) is the write guard.

Exact-label contract (owner review pt.7 + follow-up, 2026-11-13): when a
source index entry carries explicit raw "margin"/"u" dicts (producer
corpora; dagger/awr GameResult corpora use the same keys), the dataset
gains four columns -- margin_raw / u_raw (float32), outcome_type
(int8), label_status (int8) -- aligned per row with the game each row
came from. outcome_type is the RESULT class: 0 normal, 1
fault_one_side, 2 unusable, -1 not recorded. The class comes from the
entry's declared fields ("outcome", "unusable", "fault_sides" player
indices); no declaration means normal. u_raw is computed BY CLASS from
the raw margin: normal -> 0.5*WDL + 0.5*sigmoid(M/1000);
fault_one_side -> the WDL value alone (1 / 0.5 / 0 by margin sign);
unusable -> NaN (margin_raw is NaN too -- an unusable game has no
usable result, and its labels are NEVER zero-filled as if true).
label_status is per row: 0 exact (raw labels, normal or fault class),
1 approx (no raw labels; only the legacy tanh value exists), 2
unusable. Games without raw dicts contribute approx rows with NaN raw
columns -- nothing is fabricated -- and a corpus with no raw-aware game
at all gets NO raw columns (absent means absent). The builder
cross-checks the source's raw margin against the index scores margin
and (for normal games) the source u against the formula; a mismatch
fails that game under the C02 contract.

F11 terminal contract (REVIEW_FULL_20261113): an index entry may carry
"terminal": true (producer attestation that the game actually ENDED,
with "end_reason": "extinction"|"horizon"). Terminal-marked games are
exempt from the min_steps filter -- a real two-step extinction game is
complete data, not a truncated fragment -- and are counted separately
in meta (terminal_games / terminal_short_games). Entries WITHOUT the
marker keep the historical min_steps behaviour, so truncated or legacy
short games are still filtered as min_steps, never silently promoted.
"""

import gc
import gzip
import hashlib
import json
import os
import shutil
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cnet2_feat as F  # noqa: E402

FIELDS = {  # name: (dtype, shape-per-sample)
    "maps": (np.uint8, (F.P, F.N)), "scal": (np.float16, (F.NS,)),
    "tok_pos": (np.int16, (F.T,)), "tok_kind": (np.int8, (F.T,)), "tok_mask": (np.bool_, (F.T,)),
    "tok_cargo": (np.float16, (F.T,)), "hist_cell": (np.int16, (F.T, F.H)),
    "hist_cargo": (np.float16, (F.T, F.H)), "hist_move": (np.int8, (F.T, F.H)),
    "uf": (np.float16, (F.OWN, F.UF)), "mf": (np.float16, (F.OWN, 5, F.MF)),
    "legal": (np.bool_, (F.OWN, 5)), "y": (np.int8, (F.OWN,)), "carbon": (np.uint8, (F.N,)),
    "value": (np.float16, ()), "teacher": (np.int8, ()), "gid": (np.int32, ()), "step": (np.int16, ()),
}


STEP_MIN = int(os.environ.get("CNET2_STEP_MIN", "0"))   # keep only rows at step >= this (endgame-only teachers)
STEP_MAX = int(os.environ.get("CNET2_STEP_MAX", "9999"))  # ... and <= this (opening-only teachers)
UW = os.environ.get("CNET2_UW") == "1"   # per-unit decision weights (dataset v6+)
UW_DELIV, UW_SEIZED, UW_SEIZED_K, UW_MINLOAD = 1.5, 0.3, 4, 50.0
UW_RECRUIT, UW_REC_T0, UW_REC_T1 = 2.0, 40, 100
UW_PLANT = os.environ.get("CNET2_UW_PLANT") == "1"   # dataset v10+: planter trips
UW_PLANT_OK, UW_SEIZE_OK, UW_PLANT_FAIL, UW_PLANT_FAIL_K = 2.0, 1.5, 0.7, 3   # planting is worth more than seizing
UW_TREELIFE = os.environ.get("CNET2_UW_TREELIFE") == "1"     # v18+: where a planted tree survives matters
UW_LIFE_LONG, UW_LIFE_SHORT, LIFE_LONG, LIFE_SHORT = 1.5, 0.8, 40, 15
ALLOW_PARTIAL = os.environ.get("CNET2_ALLOW_PARTIAL") == "1"   # C02: publish a DECLARED partial dataset despite per-game failures (default: any failure fails the build)
if UW:
    FIELDS["uw"] = (np.float16, (F.OWN,))

# Exact-label columns (owner pt.7): NOT part of FIELDS -- cnet2_train.load
# requires every FIELDS file to exist, and legacy datasets have no raw
# columns. They join a generation's field set only when at least one
# contributing game is label-aware (see main).
RAW_FIELDS = {"margin_raw": (np.float32, ()), "u_raw": (np.float32, ()),
              "outcome_type": (np.int8, ()), "label_status": (np.int8, ())}
OUTCOME_CODES = {"normal": 0, "fault_one_side": 1, "unusable": 2}
OUTCOME_MISSING = -1
LABEL_EXACT, LABEL_APPROX, LABEL_UNUSABLE = 0, 1, 2
_RAW_FILL = {"margin_raw": np.nan, "u_raw": np.nan,
             "outcome_type": np.int8(OUTCOME_MISSING),
             "label_status": np.int8(LABEL_APPROX)}
DERIVED_FILES = ("w.npy", "uw.npy")
DERIVED_SIDECAR = "derived.json"
DERIVED_FORMAT = "cnet2-derived/1"


def _u_of_margin(m):
    """Normal-game utility: U = 0.5*WDL + 0.5*sigmoid(M/1000),
    WDL = 1 / 0.5 / 0 for M > / = / < 0 (platform rule, CarbonFormer
    adoption). Same formula as producer selfplay_gen._u_of_margin."""
    import math
    w = 1.0 if m > 0 else (0.0 if m < 0 else 0.5)
    return 0.5 * w + 0.5 / (1.0 + math.exp(-float(m) / 1000.0))


def _wdl_value(m):
    """WDL value alone -- the utility of a single-side execution-fault
    game (platform rule: fault games are scored on WDL only)."""
    return 1.0 if m > 0 else (0.0 if m < 0 else 0.5)


def _label_inputs(e):
    """Normalized declared fields that decide the exact-label class.
    The class alone is too coarse for cache/generation identity: the
    producer's explicit outcome string, the unusable flag and the exact
    fault_sides list are all label provenance and are fingerprinted
    verbatim (after JSON-safe normalization)."""
    fs = e.get("fault_sides")
    if isinstance(fs, (list, tuple)):
        fs = [int(x) if isinstance(x, (int, float)) and float(x).is_integer()
              else str(x) for x in fs]
    elif fs is not None:
        fs = str(fs)
    return {"outcome": e.get("outcome") if isinstance(e.get("outcome"), str) else None,
            "unusable": e.get("unusable") is True,
            "fault_sides": fs}


def _outcome_class(e):
    """Result class of an index entry from its declared fault fields.
    Fault EVIDENCE wins over the "outcome" string: a producer stamps
    outcome="normal" at generation time, and platform fault records
    (fault_sides / unusable) are attached to the entry afterwards --
    the later, more specific record must not be overridden by the
    earlier generic stamp. Order: "unusable": true / fault_sides
    (player indices; 2+ distinct = unusable, 1 = fault_one_side) >
    explicit "outcome" > normal."""
    if e.get("unusable") is True:
        return "unusable"
    fs = e.get("fault_sides")
    if isinstance(fs, (list, tuple)):
        try:
            n = len({int(x) for x in fs if int(x) in (0, 1)})
        except Exception:
            n = 0
        if n >= 2:
            return "unusable"
        if n == 1:
            return "fault_one_side"
    oc = e.get("outcome")
    if oc in OUTCOME_CODES:
        return oc
    return "normal"


def unit_weights(st, p, slots):
    """slots: list over samples of (t, own_uid list). Returns [n, OWN] multipliers:
    loaded trips (max cargo >= UW_MINLOAD) delivered safely -> UW_DELIV on that unit's decisions during the trip;
    trips ending in the unit's death with cargo -> UW_SEIZED on its last UW_SEIZED_K decisions only (earlier steps x1);
    base slots (10, 11) at steps UW_REC_T0..UW_REC_T1 -> UW_RECRUIT."""
    T = len(st)
    cargo = [{} for _ in range(T)]; pos = [{} for _ in range(T)]; bases = [set() for _ in range(T)]
    kind = {}; own_trees = [set() for _ in range(T)]; own_tree_id = [{} for _ in range(T)]; own_tree_ids = [set() for _ in range(T)]
    for t in range(T):
        pl = st[t][0]["observation"]["players"][p]
        bases[t] = set(int(b) for b in pl[1].values())
        own_trees[t] = set(int(v[0]) for v in (pl[3] or {}).values()) if isinstance(pl[3], dict) else set()
        own_tree_id[t] = {int(v[0]): k for k, v in (pl[3] or {}).items()} if isinstance(pl[3], dict) else {}
        own_tree_ids[t] = set(own_tree_id[t].values())
        for u, w in pl[2].items():
            cargo[t][u] = float(w[1]); pos[t][u] = int(w[0]); kind[u] = w[2]
    mult = {}
    uids = set(u for c in cargo for u in c)
    for u in uids:
        t = 0
        while t < T:
            if cargo[t].get(u, 0) > 0:
                t0 = t; mx = 0.0
                while t < T and cargo[t].get(u, 0) > 0:
                    mx = max(mx, cargo[t][u]); t += 1
                t1 = t - 1                     # last step holding cargo
                if mx >= UW_MINLOAD and t < T:
                    if u not in cargo[t]:      # died while loaded -> seized
                        for k in range(max(t0, t1 - UW_SEIZED_K + 1), t1 + 1):
                            mult[(k, u)] = UW_SEIZED
                    elif pos[t].get(u) in bases[t]:
                        for k in range(t0, t1 + 1):
                            mult[(k, u)] = UW_DELIV
            else:
                t += 1
    if UW_PLANT:
        # planter trip = birth..last step alive. Success = the cell it stood on becomes OURS the next step
        # (a new tree planted there, or an enemy tree there seized). Success: whole trip x UW_PLANT_OK.
        # Failure: only the last UW_PLANT_FAIL_K decisions x UW_PLANT_FAIL (some deaths buy something: blocking, trades).
        # planters are NOT consumed by planting: one planter can plant/seize several times. Each segment
        # (birth or previous success .. success) is up-weighted; the tail after the last success up to death keeps
        # weight 1 except its last UW_PLANT_FAIL_K decisions (x UW_PLANT_FAIL) when the planter died there.
        for u in [u for u in kind if kind[u] == "PLANTER"]:
            ts = [t for t in range(T) if u in pos[t]]
            if not ts:
                continue
            t0, t1 = ts[0], ts[-1]
            seg = t0
            for t in ts:
                if t + 1 < T:
                    c = pos[t][u]
                    if c in own_trees[t + 1] and c not in own_trees[t]:     # planted here or seized the enemy tree here
                        tid = own_tree_id[t + 1].get(c, "")
                        planted = tid.startswith(f"player-{p}-")   # seized trees keep the enemy id
                        fac = UW_PLANT_OK if planted else UW_SEIZE_OK
                        if UW_TREELIFE and tid:
                            life = 0
                            while t + 1 + life < T and tid in own_tree_ids[t + 1 + life]:
                                life += 1
                            if t + 1 + life < T:            # (trees alive at the end are not judged)
                                fac *= UW_LIFE_LONG if life >= LIFE_LONG else (UW_LIFE_SHORT if life < LIFE_SHORT else 1.0)
                        for k in range(seg, t + 1):
                            mult[(k, u)] = mult.get((k, u), 1.0) * fac
                        seg = t + 1
            if t1 + 1 < T and u not in pos[t1 + 1]:                         # died
                for k in range(max(seg, t1 - UW_PLANT_FAIL_K + 1), t1 + 1):
                    mult[(k, u)] = mult.get((k, u), 1.0) * UW_PLANT_FAIL
    out = np.ones((len(slots), F.OWN), np.float32)
    for i, (t, own) in enumerate(slots):
        for s, u in enumerate(own):
            if u is None:
                continue
            if s >= 10:
                if UW_REC_T0 <= t <= UW_REC_T1:
                    out[i, s] = UW_RECRUIT
            else:
                out[i, s] = mult.get((t, u), 1.0)
    return out


def teacher_id(name):
    return F.TEACHERS.index(name) if name in F.TEACHERS else F.TEACHERS.index("other")


def build_game(task):
    try:
        return _build_game(task)
    except Exception as e:
        # C02: a load OR processing failure is a recorded FAILURE (game id +
        # reason), never a silent zero-row result -- a legitimate zero-row
        # game returns err=None from _build_game. F05: a failed task must
        # not leave a stale shard behind for the manifest to certify as a
        # current product.
        try:
            os.remove(task[5])
        except OSError:
            pass
        return task[2], 0, repr(e)[:200]


def _build_game(task):
    path, sides, gi, tids, margins, out = task[:6]
    # Exact-label payload (owner pt.7): None, or {"mg": [per-player raw
    # margins] | None, "u": [per-player source u], "outcome": code}.
    # "mg" is None only for declared-unusable games (labels are NaN by
    # definition; no margin is needed or trusted).
    raw = task[6] if len(task) > 6 else None
    d = json.load(gzip.open(path))
    st = d["steps"]
    rows = {k: [] for k in FIELDS}
    if raw is not None:
        for k in RAW_FIELDS:
            rows[k] = []
    for p, tid, mg in zip(sides, tids, margins):
        rmg = ru = None
        if raw is not None and raw["mg"] is not None:
            rmg = float(raw["mg"][p])
            ru = float(raw["u"][p])
            # Alignment cross-check: the declared raw margin must equal
            # the index-scores margin this side's value label is built
            # from, and (normal games) the declared u must equal the
            # formula value. A mismatch means the corpus's raw record
            # disagrees with its own scores -- fail the game (C02).
            if abs(rmg - float(mg)) > 1e-4:
                raise ValueError(
                    f"raw margin mismatch side {p}: declared {rmg} vs scores margin {float(mg)}")
            if raw["outcome"] == OUTCOME_CODES["normal"] and \
                    abs(ru - _u_of_margin(rmg)) > 1e-6:
                raise ValueError(
                    f"raw u mismatch side {p}: declared {ru} vs formula {_u_of_margin(rmg)}")
        f = F.Feat(p)
        slots = []
        for t in range(len(st) - 1):
            o0 = st[t][0]["observation"]
            obs = {"step": o0["step"], "players": o0["players"],
                   "carbon": st[t][p]["observation"]["carbon"], "player": p}
            if st[t][p].get("status") not in (None, "ACTIVE"):
                break
            x = f.observe(obs)
            act = st[t + 1][p].get("action") or {}
            y = f.labels(act)
            pf = F.Prefix(x, f.aux)
            uf = np.zeros((F.OWN, F.UF), np.float32); mf = np.zeros((F.OWN, 5, F.MF), np.float32)
            legal = np.zeros((F.OWN, 5), bool)
            for s in range(F.OWN):
                if not x["tok_mask"][s]:
                    continue
                uf[s], mf[s], legal[s] = pf.feats(s)
                if not legal[s, y[s]]:
                    y[s] = 0
                pf.advance(s, int(y[s]))
            if t < STEP_MIN or t > STEP_MAX:   # features still observe every step (history); rows only in the window
                continue
            full = np.asarray(o0.get("full_carbon", obs["carbon"]), np.float64)
            order = F.CANON if p else F.GRID
            r = rows
            r["maps"].append(x["maps"]); r["scal"].append(x["scal"])
            r["tok_pos"].append(x["tok_pos"]); r["tok_kind"].append(x["tok_kind"])
            r["tok_mask"].append(x["tok_mask"]); r["tok_cargo"].append(x["tok_cargo"])
            r["hist_cell"].append(x["hist_cell"]); r["hist_cargo"].append(x["hist_cargo"])
            r["hist_move"].append(x["hist_move"])
            r["uf"].append(uf); r["mf"].append(mf); r["legal"].append(legal); r["y"].append(y)
            r["carbon"].append(F.q8(full[order] / 100.0))
            r["value"].append(np.tanh(mg / 3000.0)); r["teacher"].append(tid)
            r["gid"].append(gi); r["step"].append(t)
            if raw is not None:
                if raw["outcome"] == OUTCOME_CODES["unusable"]:
                    r["margin_raw"].append(np.nan); r["u_raw"].append(np.nan)
                    r["label_status"].append(LABEL_UNUSABLE)
                else:
                    r["margin_raw"].append(rmg)
                    r["u_raw"].append(_u_of_margin(rmg) if raw["outcome"] == OUTCOME_CODES["normal"]
                                      else _wdl_value(rmg))
                    r["label_status"].append(LABEL_EXACT)
                r["outcome_type"].append(raw["outcome"])
            if UW:
                slots.append((t, list(f.own_uid)))
        if UW and slots:
            r["uw"].extend(unit_weights(st, p, slots))
    n = len(rows["y"])
    if n == 0:
        # F05: a zero-row outcome (e.g. the step window now excludes every
        # row of an early-ending replay) must not leave an earlier shard on
        # disk: the next run would otherwise resurrect it as a valid cached
        # product under the new config.
        try:
            os.remove(out)
        except OSError:
            pass
        return gi, 0, None
    arr = {k: np.asarray(v).astype((RAW_FIELDS[k] if k in RAW_FIELDS else FIELDS[k])[0])
           for k, v in rows.items()}
    np.savez(out, **arr)
    return gi, n, None


OUR_NAMES = ("claude-opus55-cloud-",)
STRONG = 2400.0
OURS_MIN_OPP = 2350.0


def ratings(dirs):
    r = {}
    for rd in dirs:
        fp = os.path.join(rd, "leaderboard.json")
        if not os.path.exists(fp):
            continue
        lb = json.load(open(fp))
        for e in (lb["strategies"] if isinstance(lb, dict) else lb):
            if e.get("rating") is not None:
                r[e["name"]] = max(r.get(e["name"], 0.0), float(e["rating"]))
    return r


def _scan(dirs, min_steps=200, weighted=False):
    """Core scanner. Returns (games, missing, filtered):
    games    -- eligible games whose replay file exists;
    missing  -- eligible games (passed steps/teacher/ours filters) whose
                replay file does NOT exist (N03: failures, not filtering);
    filtered -- {reason: count} for index rows legitimately excluded
                (duplicate / min_steps / no teacher side / ours-corpus
                rules). Eligibility is decided from the index entry alone,
    BEFORE the existence check, so a missing file can only be classified
    as missing when the game would otherwise have been built."""
    games, seen = [], set()
    missing = []
    missing_by_key = {}
    filtered = {}

    def _filtered(reason):
        filtered[reason] = filtered.get(reason, 0) + 1

    R = ratings(dirs) if weighted else {}
    for rd in dirs:
        ours = "ours_corpus" in rd
        idx = json.load(open(os.path.join(rd, "index.json")))
        for g in idx:
            key = os.path.basename(g["file"])
            if key in seen:
                _filtered("duplicate")
                continue
            # F11: a producer-attested terminal game (it actually ended,
            # e.g. a two-step extinction) is complete data and exempt
            # from min_steps; unmarked short games stay filtered.
            terminal = g.get("terminal") is True
            if g.get("steps", 0) < min_steps and not terminal:
                _filtered("min_steps")
                continue
            fp = os.path.join(rd, g["file"])
            order = g["order"]
            names = [g["strategies"][order[s]]["name"] for s in (0, 1)]
            scores = [g["scores"][order[s]] for s in (0, 1)]
            # Exact-label provenance (owner pt.7): explicit per-strategy-key
            # raw margin/u dicts are the ONLY source of exact labels; the
            # result class comes from the declared fault fields.
            raw_mg = raw_u = None
            if isinstance(g.get("margin"), dict) and isinstance(g.get("u"), dict):
                try:
                    raw_mg = [float(g["margin"][order[s]]) for s in (0, 1)]
                    raw_u = [float(g["u"][order[s]]) for s in (0, 1)]
                except Exception:
                    raw_mg = raw_u = None
            oclass = _outcome_class(g)
            if ours:
                us = [s for s in (0, 1) if names[s].startswith(OUR_NAMES)]
                if len(us) != 1:
                    _filtered("ours_shape")
                    continue
                o = 1 - us[0]
                if not (scores[o] > scores[us[0]] and (names[o] in F.TEACHERS[:-1] or R.get(names[o], 0) >= OURS_MIN_OPP)):
                    _filtered("ours_not_qualified")
                    continue
                sides = [o]
            else:
                sides = [s for s in (0, 1) if names[s] in F.TEACHERS[:-1]]
            if not sides:
                _filtered("no_teacher_side")
                continue
            if not os.path.exists(fp):
                if key in missing_by_key:
                    _filtered("duplicate")
                    continue
                # N03: this game SHOULD have been built (it passed every
                # eligibility filter) but its replay is gone. Record it as
                # a failure -- the old code continued here silently, so the
                # game never reached the errs contract and the build
                # published partial=false with no trace of the lost id.
                m = {"key": key, "file": fp, "names": names,
                     "scores": scores, "sides": sides,
                     "val": zlib.crc32(key.encode()) % 10 == 0,
                     "src": os.path.basename(rd.rstrip("/")),
                     "terminal": terminal, "end_reason": g.get("end_reason"),
                     "steps": g.get("steps", 0),
                     "raw_margin": raw_mg, "raw_u": raw_u, "outcome": oclass,
                     "label_inputs": _label_inputs(g),
                     "reason": "missing_replay",
                     "error": f"missing_replay: indexed replay file not found: {fp}"}
                missing.append(m)
                missing_by_key[key] = m
                continue
            if key in missing_by_key:
                # REVIEW_FULL S7: an earlier shard's copy was missing but
                # THIS copy exists -- the good copy wins regardless of
                # shard order (previously seen-before-existence made the
                # outcome depend on which copy came first).
                missing.remove(missing_by_key.pop(key))
            seen.add(key)
            g = {"file": fp, "key": key, "names": names, "scores": scores, "sides": sides,
                 "val": zlib.crc32(key.encode()) % 10 == 0, "src": os.path.basename(rd.rstrip("/")),
                 "terminal": terminal, "end_reason": g.get("end_reason"),
                 "raw_margin": raw_mg, "raw_u": raw_u, "outcome": oclass,
                 "label_inputs": _label_inputs(g),
                 "steps": g.get("steps", 0),
                 "short_terminal": bool(terminal and g.get("steps", 0) < min_steps)}
            if weighted:
                strong = min(R.get(names[0], 0), R.get(names[1], 0)) >= STRONG
                g["wts"] = [(2.0 if scores[s] > scores[1 - s] else 1.0) * (1.5 if strong else 1.0) for s in sides]
                g["ours_loss"] = ours
            games.append(g)
    return games, missing, filtered


def scan(dirs, min_steps=200, weighted=False):
    """weighted: per-side sample weight = (2 if side won) * (1.5 if both rated >= STRONG);
    dirs whose name contains 'ours_corpus' contribute only games our bot LOST, teacher = winning opponent
    (rated >= OURS_MIN_OPP or a listed teacher).
    Back-compatible wrapper over _scan: returns only the buildable games.
    Callers that publish data must use _scan (as main does) so missing
    eligible replays are accounted as failures, not silently dropped."""
    games, _missing, _filtered = _scan(dirs, min_steps=min_steps, weighted=weighted)
    return games


def _replay_content_digest(path):
    """SHA256 of the replay's decompressed content. The gzip/stat identity
    is NOT a content identity: a replay can be rewritten with different
    carbon values under an unchanged index and row count, and a cache or
    generation keyed on (size, mtime) -- or on the index alone -- would
    then rebuild a shard and still republish the previous generation.
    Hashing the decompressed stream also keeps a byte-identical replay
    from minting a new generation merely for new gzip framing."""
    h = hashlib.sha256()
    try:
        with gzip.open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def _game_fingerprint(g, tids, mg):
    """The single per-game content fingerprint used by BOTH the shard
    cache key and the generation key. It merges the three input classes
    that determine a game's published rows: (1) the index-derived shard
    inputs (kept sides, teacher ids, score margins, sample weights);
    (2) the exact-label inputs via _game_label_digest (scores, declared
    margin/u, outcome class and its declared fault provenance) -- the
    shard cache previously omitted these, so adding fault_sides=[0]
    changed the generation id while the stale shard underneath kept
    publishing normal labels; and (3) the replay content digest itself,
    so a carbon-only replay edit cannot hide behind an unchanged index.
    One fingerprint, one definition: the cache can never certify a
    shard that the generation key would not also distinguish."""
    detail = {"key": g["key"], "names": g["names"], "scores": g["scores"],
              "sides": g["sides"], "tids": list(tids),
              "mg": [float(x) for x in mg],
              "wts": g.get("wts"), "ours_loss": bool(g.get("ours_loss", False)),
              "label_digest": _game_label_digest(g),
              "replay_sha256": (_replay_content_digest(g["file"])
                                if g.get("file") and os.path.exists(g["file"]) else None)}
    return hashlib.sha256(json.dumps(detail, sort_keys=True).encode()).hexdigest()[:16]


def _game_content_key(g, tids, mg):
    """Backward-compatible alias: the shard cache key IS the unified
    per-game fingerprint (see _game_fingerprint)."""
    return _game_fingerprint(g, tids, mg)


def _game_label_digest(g):
    """Per-game digest of every INDEX-derived input that feeds this game's
    value/label columns, for the F05 generation key in main(). The
    generation id used to fingerprint only [key, row count] per game, so
    editing a game's scores/margin/u (or its fault class) in index.json
    rebuilt the shard -- the shard cache key covers scores/margins/wts --
    yet left the generation id unchanged: the sealed previous generation
    was reused wholesale and the build silently published the OLD value
    and raw-label columns for the edited game.
    Coverage: scores (value = tanh(margin/3000) with margins the per-side
    score differences), the declared raw margin/u pair (the margin_raw /
    u_raw columns), and the outcome class (it selects the u rule --
    formula vs WDL-only -- and NaN labelling for unusable games; the
    class is a deterministic function of the index fault fields and the
    label code consults only the class). Names/sides/wts ride along as
    in _game_content_key: they fix the teacher column and the weighted
    w column. A shard sha256 is NOT a substitute for this digest: the
    shard only partially reflects these inputs (a fault_one_side game's
    declared u is never written into it, and raw margins are
    cross-checked against scores only to 1e-4), so the digest is taken
    from the normalized index values themselves."""
    detail = {"names": g["names"], "scores": [float(x) for x in g["scores"]],
              "sides": g["sides"],
              "raw_margin": (None if g.get("raw_margin") is None
                             else [float(x) for x in g["raw_margin"]]),
              "raw_u": (None if g.get("raw_u") is None
                        else [float(x) for x in g["raw_u"]]),
              "outcome": g.get("outcome", "normal"),
              "label_inputs": g.get("label_inputs"),
              "wts": g.get("wts")}
    return hashlib.sha256(json.dumps(detail, sort_keys=True).encode()).hexdigest()[:16]


def _cfg_fingerprint():
    """R08: fingerprint of everything that changes shard CONTENT -- the step window,
    the unit-weight switches, the teacher map, the field schema / feature shapes and
    the builder+feat source itself. Cached shards are only reused under an identical
    fingerprint (plus an unchanged source file, see main); before this, the cache key
    was just the replay name, so a config change silently reused stale shards while
    meta.json claimed the new config."""
    import hashlib
    detail = {
        "step_min": STEP_MIN, "step_max": STEP_MAX,
        "uw": UW, "uw_plant": UW_PLANT, "uw_treelife": UW_TREELIFE,
        "weighted": os.environ.get("CNET2_WEIGHTED") == "1",
        "teachers": list(F.TEACHERS),
        "fields": {k: [str(dt), list(sh)] for k, (dt, sh) in FIELDS.items()},
        "raw_fields_v1": {k: [str(dt), list(sh)] for k, (dt, sh) in RAW_FIELDS.items()},
        "outcome_codes": OUTCOME_CODES, "label_codes": [LABEL_EXACT, LABEL_APPROX, LABEL_UNUSABLE],
        "feat_sig": [F.P, F.N, F.NS, F.T, F.H, F.OWN, F.UF, F.MF],
    }
    for mod in (__file__, getattr(F, "__file__", None)):
        try:
            detail["src_" + os.path.basename(mod)] = hashlib.sha256(open(mod, "rb").read()).hexdigest()[:16]
        except Exception:
            pass
    return hashlib.sha256(json.dumps(detail, sort_keys=True).encode()).hexdigest()[:16]


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_generation_path(path):
    parts = os.path.abspath(path).split(os.sep)
    return any(seg == "generations" and parts[i + 1].startswith("gen-")
               for i, seg in enumerate(parts[:-1]))


def assert_not_in_generation(path, what="write"):
    """Write guard (owner pt.4): NOTHING may write into a published
    generation directory -- generations are immutable once sealed, so a
    post-processing tool pointed at one is a configuration error, not a
    variant. Post-processing writes go to the dataset root via
    register_derived()."""
    if _is_generation_path(path):
        raise ValueError(
            f"derived-layer guard: refusing to {what} inside an immutable "
            f"generation directory: {path}")


def _patch_flat_meta_derived(out, side):
    """Mirror the derived-layer declaration into the flat meta.json so a
    flat-view consumer can find each derived file's base generation and
    hash without opening the sidecar. side=None strips the declaration.
    Only the FLAT meta is patched -- the generation's own meta/manifest
    stay pristine."""
    meta_path = os.path.join(out, "meta.json")
    if not os.path.exists(meta_path):
        return
    try:
        meta = json.load(open(meta_path))
    except Exception:
        return
    if side is None:
        if "derived_layer" not in meta:
            return
        meta.pop("derived_layer", None)
    else:
        meta["derived_layer"] = side
    tmp = meta_path + f".tmp-{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(meta, fh)
    os.replace(tmp, meta_path)


def register_derived(out, files, params=None, expected_base=None):
    """Sanctioned post-processing entry (owner pt.4): register derived
    files (e.g. set_uw's w.npy / uw.npy) that live in the dataset ROOT.

    `expected_base` (P1-B): a caller that read its derivation inputs
    from a pinned generation passes that generation id here; if the
    dataset pointer has moved since, registration is REFUSED before
    anything is written -- the derived rows were computed from another
    generation's features and must never be bound to the new one.
    Callers that pass nothing keep the historical behaviour (bind to
    whatever generation is current at registration time).

    Sidecar format "cnet2-derived/1" (out/derived.json, mirrored into
    flat meta.json under "derived_layer"):
      {"format": "cnet2-derived/1",
       "files": {<file name>: {"field": <name without .npy>,
                              "dtype": str, "rows": int, "shape": [..],
                              "sha256": <hex of the file bytes>,
                              "base_generation": <generation id the
                                  rows align with>,
                              "params": <generation parameters dict>}}}
    A consumer (planned: cnet2_train.load) validates: the file's
    base_generation == the dataset's current generation (else the
    derived rows belong to another build and must be regenerated),
    rows == the generation's row count, and the sha256 matches before
    use. Files inside a generation directory are REFUSED here -- a
    derived file can never become part of a generation product.
    Returns the sidecar dict."""
    out_abs = os.path.abspath(out)
    assert_not_in_generation(out_abs, "register a derived layer for")
    gen_dir, man = load_generation(out_abs, verify=False)
    base_gen = man["generation"]
    if expected_base is not None and base_gen != expected_base:
        raise ValueError(
            f"derived registration refused: dataset generation is {base_gen}, "
            f"caller pinned {expected_base} -- the pointer moved since the "
            f"derivation inputs were read; rerun the derivation")
    total_rows = int(man["total"])
    side_path = os.path.join(out_abs, DERIVED_SIDECAR)
    entries = {}
    if os.path.exists(side_path):
        try:
            prev = json.load(open(side_path))
            if isinstance(prev, dict):
                entries = prev.get("files", {}) or {}
        except Exception:
            entries = {}
    for fp in files:
        ap = os.path.abspath(fp)
        assert_not_in_generation(ap, "write derived file")
        if os.path.dirname(ap) != out_abs:
            raise ValueError(
                "derived files must live in the dataset root (flat view), "
                f"not in a generation directory: {ap}")
        if not os.path.exists(ap):
            raise FileNotFoundError(ap)
        arr = np.load(ap, mmap_mode="r")
        if int(arr.shape[0]) != total_rows:
            raise ValueError(
                f"derived file {os.path.basename(ap)} has {int(arr.shape[0])} rows, "
                f"generation {base_gen} has {total_rows}: row order/count mismatch")
        name = os.path.basename(ap)
        entries[name] = {
            "field": name[:-4] if name.endswith(".npy") else name,
            "dtype": str(arr.dtype), "rows": int(arr.shape[0]),
            "shape": [int(x) for x in arr.shape],
            "sha256": _sha256_file(ap), "base_generation": base_gen,
            "params": params or {}}
    side = {"format": DERIVED_FORMAT, "files": entries}
    tmp = side_path + f".tmp-{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(side, fh)
    os.replace(tmp, side_path)
    _patch_flat_meta_derived(out_abs, side)
    return side


def _refresh_derived_declaration(out, gen_id, prev_gen, pre_existing, gen_fields):
    """Publish-time derived bookkeeping: derived files that already sit
    in the flat root (written by post-processing between builds) are
    re-declared against their base generation -- the previous
    generation if the file's bytes predate this publish (params carried
    over from the old sidecar when the hash matches), else the new one.
    Generation-owned fields (a weighted build's w) are never derived."""
    side_path = os.path.join(out, DERIVED_SIDECAR)
    prev_entries = {}
    if os.path.exists(side_path):
        try:
            prev_entries = json.load(open(side_path)).get("files", {}) or {}
        except Exception:
            prev_entries = {}
    entries = {}
    for name in DERIVED_FILES:
        if name[:-4] in gen_fields:
            continue
        fp = os.path.join(out, name)
        if not os.path.exists(fp):
            continue
        sha = _sha256_file(fp)
        arr = np.load(fp, mmap_mode="r")
        ent = {"field": name[:-4], "dtype": str(arr.dtype),
               "rows": int(arr.shape[0]), "shape": [int(x) for x in arr.shape],
               "sha256": sha, "base_generation": gen_id, "params": {}}
        prev_ent = prev_entries.get(name)
        if name in pre_existing and pre_existing[name] == sha:
            ent["base_generation"] = (prev_ent or {}).get("base_generation") or prev_gen or gen_id
            ent["params"] = (prev_ent or {}).get("params", {})
        entries[name] = ent
    if entries:
        side = {"format": DERIVED_FORMAT, "files": entries}
        tmp = side_path + f".tmp-{os.getpid()}"
        with open(tmp, "w") as fh:
            json.dump(side, fh)
        os.replace(tmp, side_path)
        _patch_flat_meta_derived(out, side)
    else:
        try:
            os.remove(side_path)
        except OSError:
            pass
        _patch_flat_meta_derived(out, None)


def load_generation(out, verify=True):
    # F05: resolve the dataset generation the pointer certifies.
    # Returns (generation_dir, manifest). With verify=True (default)
    # every file the manifest declares must exist with a matching
    # sha256 -- a consumer that resolves BEFORE opening any array can
    # therefore never observe a mixed-generation product, and
    # undeclared files (e.g. a stale w.npy from an earlier weighted
    # build, or derived-layer files) are not part of the product and
    # are ignored here, never read. Raises
    # FileNotFoundError/ValueError on any inconsistency.
    ptr_path = os.path.join(out, "current.json")
    if not os.path.exists(ptr_path):
        raise FileNotFoundError(f"no dataset generation pointer: {ptr_path}")
    ptr = json.load(open(ptr_path))
    gen_dir = os.path.join(out, ptr["dir"])
    man_path = os.path.join(gen_dir, "manifest.json")
    if not os.path.exists(man_path):
        raise FileNotFoundError(f"generation manifest missing: {man_path}")
    man = json.load(open(man_path))
    if man.get("generation") != ptr.get("generation"):
        raise ValueError("pointer/manifest generation mismatch")
    if verify:
        for name, info in man.get("fields", {}).items():
            p = os.path.join(gen_dir, info["file"])
            if not os.path.exists(p):
                raise FileNotFoundError(f"generation file missing: {p}")
            if _sha256_file(p) != info["sha256"]:
                raise ValueError(f"generation file hash mismatch: {p}")
    return gen_dir, man


def resolve_dataset(out, verify=True):
    # Path of the current certified generation directory (see
    # load_generation). CLI: python cnet2_build.py --resolve OUT_DIR.
    return load_generation(out, verify=verify)[0]


def main(out, dirs):
    assert_not_in_generation(out, "build a dataset into")
    os.makedirs(os.path.join(out, "shards"), exist_ok=True)
    os.makedirs(os.path.join(out, "generations"), exist_ok=True)
    # P3: drop leftovers of a crashed publish; the temp dir is never a
    # published product (only os.replace'd files in `out` itself are).
    for stale in os.listdir(out):
        if stale.startswith(".tmp_publish_"):
            shutil.rmtree(os.path.join(out, stale), ignore_errors=True)
    for stale in os.listdir(os.path.join(out, "generations")):
        if stale.startswith(".staging-"):
            shutil.rmtree(os.path.join(out, "generations", stale), ignore_errors=True)
    weighted = os.environ.get("CNET2_WEIGHTED") == "1"
    games, missing, filtered = _scan(dirs, weighted=weighted)
    # N03: missing-but-eligible games join the games list as declared
    # failures (no task is built for them; their gi is pre-seeded into
    # errs below), exactly like a bad-gzip game: default abort, or a
    # declared partial dataset under CNET2_ALLOW_PARTIAL=1.
    missing_err = {m["key"]: m["error"] for m in missing}
    for m in missing:
        games.append({"file": m["file"], "key": m["key"], "names": m["names"],
                      "scores": m["scores"], "sides": m["sides"],
                      "val": m["val"], "src": m["src"], "missing_replay": True,
                      "terminal": m.get("terminal", False),
                      "end_reason": m.get("end_reason"), "steps": m.get("steps", 0),
                      "raw_margin": m.get("raw_margin"), "raw_u": m.get("raw_u"),
                      "outcome": m.get("outcome", "normal"),
                      "label_inputs": m.get("label_inputs")})
    fields = dict(FIELDS)
    if weighted:
        fields["w"] = (np.float16, ())
    tasks = []
    for gi, g in enumerate(games):
        g["gi"] = gi
        tids = [teacher_id(g["names"][s]) for s in g["sides"]]
        mg = [g["scores"][s] - g["scores"][1 - s] for s in g["sides"]]
        g["fingerprint"] = _game_fingerprint(g, tids, mg)
        suffix = ("_s" + "".join(map(str, g["sides"]))) if weighted else ""
        g["shard"] = os.path.join(out, "shards", g["key"].replace(".json.gz", "") + suffix + ".npz")
        if g.get("missing_replay"):
            continue
        # Exact-label payload: full (raw margins declared), unusable-only
        # (declared unusable -- labels are NaN, no margin trusted), or
        # None (approx game: only the legacy tanh value exists).
        if g.get("raw_margin") is not None:
            g["raw_payload"] = {"mg": g["raw_margin"], "u": g["raw_u"],
                                "outcome": OUTCOME_CODES[g.get("outcome", "normal")]}
        elif g.get("outcome") == "unusable":
            g["raw_payload"] = {"mg": None, "u": None,
                                "outcome": OUTCOME_CODES["unusable"]}
        else:
            g["raw_payload"] = None
        tasks.append((g["file"], g["sides"], gi, tids, mg, g["shard"], g["raw_payload"]))
    cfg_fp = _cfg_fingerprint()
    man_path = os.path.join(out, "shards", "_cfp.json")
    man = {}
    if os.path.exists(man_path):
        try:
            man = json.load(open(man_path))
        except Exception:
            man = {}
    fp_ok = man.get("cfg_fp") == cfg_fp
    if man and not fp_ok:
        print(f"R08: build-config fingerprint changed ({man.get('cfg_fp')} -> {cfg_fp}); "
              "cached shards invalidated, rebuilding all", flush=True)
    # F05: the manifest is the cache's certificate. Each entry certifies ONE
    # game's outcome of THIS configuration -- its row count (a certified 0 is
    # a legitimate outcome and needs no shard file) -- under both the build
    # config fingerprint and the per-game content key. Pre-F05 entries
    # ([size, mtime] lists) never match the dict shape, so old caches rebuild
    # once. A zero-row or failed task is never certified against a leftover
    # shard file: build_game deletes the file, and failed tasks are omitted
    # from the new manifest entirely.
    old_man = man.get("shards", {}) if fp_ok else {}
    ns = {}
    new_man = {}
    ckeys = {}
    todo = []
    for t in tasks:
        gi = t[2]
        ck = games[gi].get("fingerprint") or _game_content_key(games[gi], t[3], t[4])
        ckeys[gi] = ck
        base = os.path.basename(t[5])
        ent = old_man.get(base)
        reused = False
        if isinstance(ent, dict) and ent.get("ok") and ent.get("ckey") == ck:
            try:
                st_ = os.stat(t[0])
                src_ok = ent.get("src") == [st_.st_size, st_.st_mtime_ns]
            except Exception:
                src_ok = False
            n = ent.get("n")
            if src_ok and isinstance(n, int):
                if n == 0:
                    ns[gi] = 0
                    new_man[base] = ent
                    reused = True
                elif os.path.exists(t[5]):
                    try:
                        if int(np.load(t[5])["y"].shape[0]) == n:
                            ns[gi] = n
                            new_man[base] = ent
                            reused = True
                    except Exception:
                        pass
        if not reused:
            todo.append(t)
    tasks = todo
    errs = {}
    with ProcessPoolExecutor(max_workers=int(os.environ.get("BUILD_WORKERS", "8"))) as ex:
        for gi, n, err in ex.map(build_game, tasks, chunksize=1):
            ns[gi] = n
            if err:
                errs[gi] = err
                print("err", gi, err, flush=True)
    # N03: pre-seed the missing-but-eligible games into the same errs
    # contract, so the failed/partial/abort logic below covers them.
    for g in games:
        if g.get("missing_replay"):
            errs[g["gi"]] = missing_err[g["key"]]
            ns[g["gi"]] = 0
            print("err", g["gi"], missing_err[g["key"]], flush=True)
    for t in tasks:
        gi = t[2]
        base = os.path.basename(t[5])
        if gi in errs:
            try:
                os.remove(t[5])
            except OSError:
                pass
            continue
        try:
            st_ = os.stat(t[0])
            new_man[base] = {"src": [st_.st_size, st_.st_mtime_ns],
                             "ckey": ckeys[gi], "n": int(ns.get(gi, 0)), "ok": True}
        except Exception:
            pass
    json.dump({"cfg_fp": cfg_fp, "shards": new_man}, open(man_path, "w"))
    # C02: errs used to be print-only -- the build then published the partial
    # dataset as if complete (failed games sat in meta.games with n=0 and
    # the games/teacher_sides summaries still counted them) and exited 0,
    # so downstream return-code + count gates passed it into training.
    # Now: any per-game failure aborts the publish with a non-zero exit by
    # default; CNET2_ALLOW_PARTIAL=1 instead publishes a dataset whose meta
    # explicitly declares the failed game ids, reasons and valid game count.
    failed = [{"gi": gi, "key": games[gi]["key"], "error": errs[gi],
               **({"reason": "missing_replay"} if games[gi].get("missing_replay") else {})}
              for gi in sorted(errs)]
    # C02: "valid" = a game that contributed >= 1 row. A legitimate zero-row
    # game (step window excludes every row; err None) is NOT a failure -- the
    # build still succeeds -- but it contributes no training rows, so it does
    # not count as valid either: a launch chain must see valid == index.
    valid = sum(1 for g in games if ns.get(g["gi"], 0) > 0)
    if failed and not ALLOW_PARTIAL:
        print(f"C02-FAIL build aborted: {len(failed)} of {len(games)} games failed to load/process; "
              "no dataset published (set CNET2_ALLOW_PARTIAL=1 to publish a declared partial dataset)",
              flush=True)
        for f_ in failed:
            print(f"C02-FAIL game {f_['key']} (gi {f_['gi']}): {f_['error']}", flush=True)
        # F05: publish NOTHING on abort. The generation pointer keeps
        # certifying the previous complete generation (the certified
        # product of record); shards and the shard manifest are kept so
        # a retry after fixing the bad replay reuses them. The flat
        # meta.json is removed so the aborted run cannot be mistaken for
        # a fresh flat product, and BUILD_FAILED.json records the abort.
        try:
            os.remove(os.path.join(out, "meta.json"))
        except OSError:
            pass
        with open(os.path.join(out, "BUILD_FAILED.json"), "w") as fh:
            json.dump({"cfg_fp": cfg_fp, "time": time.time(),
                       "failed_games": [f_["key"] for f_ in failed]}, fh)
        sys.exit(1)
    total = sum(ns.values())
    for g in games:
        g["n"] = ns.get(g["gi"], 0)
        if g["gi"] in errs:
            g["failed"] = errs[g["gi"]]
    terminal_games = sum(1 for g in games if g.get("terminal"))
    terminal_short = sum(1 for g in games if g.get("short_terminal"))
    # Exact-label coverage (owner pt.7): the raw columns join this
    # generation's field set iff at least one CONTRIBUTING game is
    # label-aware; approx rows are counted, never hidden.
    raw_games = sum(1 for g in games
                    if g.get("raw_payload") is not None and ns.get(g["gi"], 0) > 0)
    if raw_games:
        fields.update(RAW_FIELDS)
    exact_rows = sum(ns.get(g["gi"], 0) for g in games
                     if g.get("raw_payload") is not None
                     and g["raw_payload"]["outcome"] != OUTCOME_CODES["unusable"])
    unusable_rows = sum(ns.get(g["gi"], 0) for g in games
                        if g.get("raw_payload") is not None
                        and g["raw_payload"]["outcome"] == OUTCOME_CODES["unusable"])
    label_cov = {"raw_games": raw_games, "games": len(games),
                 "exact_rows": exact_rows,
                 "approx_rows": total - exact_rows - unusable_rows,
                 "unusable_rows": unusable_rows, "rows": total}
    meta = {"games": games, "total": total, "teachers": F.TEACHERS, "cfg_fp": cfg_fp,
            "failed_games": failed, "partial": bool(failed),
            "valid_games": valid, "filtered_games": filtered,
            "labels": label_cov,
            "terminal_games": terminal_games, "terminal_short_games": terminal_short}
    # F05 (REVIEW_FULL_20261113): publish as one immutable GENERATION.
    # The R2 temp-dir merge protected only the merge phase; the per-field
    # os.replace sequence into `out` could still expose (or, on an
    # injected mid-publish failure, permanently leave) a mixed product.
    # Now the merged arrays are written into a staging dir under
    # out/generations/, sealed with manifest.json (field list, per-file
    # sha256, row count, cfg fingerprint, declaration), and renamed into
    # place as gen-<id> in one operation. The generation id is derived
    # from the full build content, so an identical rebuild reuses the
    # existing generation instead of rewriting it. The per-game component
    # of the id is [key, rows, unified fingerprint] -- the fingerprint
    # (see _game_fingerprint) puts the game's index/shard inputs, its
    # exact-label inputs (scores/margin/u, outcome class and declared
    # fault provenance) AND its replay content digest inside the id, so
    # a label-only index edit or a carbon-only replay edit mints a NEW
    # generation instead of reusing the sealed one. Only after the
    # generation is complete is the legacy flat view in `out` refreshed
    # from it (arrays first, meta.json last), and only then is the
    # pointer out/current.json switched with a single os.replace -- the
    # one atomic commit of the whole publish. A failure anywhere before
    # the pointer switch leaves the pointer certifying the previous
    # complete generation; resolve_dataset() consumers never see a mix.
    gen_key = {"cfg_fp": cfg_fp, "weighted": weighted, "total": total,
               "valid": valid, "partial": bool(failed),
               "failed": [f_["key"] for f_ in failed],
               "filtered": filtered,
               "labels": label_cov,
               "games": [[g["key"], int(ns.get(g["gi"], 0)),
                          g.get("fingerprint") or _game_fingerprint(
                              g,
                              [teacher_id(g["names"][s]) for s in g["sides"]],
                              [g["scores"][s] - g["scores"][1 - s] for s in g["sides"]])]
                         for g in games]}
    gen_id = "gen-" + hashlib.sha256(
        json.dumps(gen_key, sort_keys=True).encode()).hexdigest()[:16]
    gen_root = os.path.join(out, "generations")
    gen_dir = os.path.join(gen_root, gen_id)
    w_copy = None

    def _gen_complete(path):
        try:
            man = json.load(open(os.path.join(path, "manifest.json")))
            return all(os.path.exists(os.path.join(path, info["file"]))
                       for info in man.get("fields", {}).values())
        except Exception:
            return False

    if _gen_complete(gen_dir):
        man = json.load(open(os.path.join(gen_dir, "manifest.json")))
        if weighted:
            w_copy = np.load(os.path.join(gen_dir, "w.npy")).astype(np.float32)
    else:
        staging = os.path.join(gen_root, f".staging-{os.getpid()}")
        shutil.rmtree(staging, ignore_errors=True)
        os.makedirs(staging)
        try:
            mm = {k: np.lib.format.open_memmap(os.path.join(staging, k + ".npy"), mode="w+", dtype=dt,
                                               shape=(total,) + sh) for k, (dt, sh) in fields.items()}
            o = 0
            for gi in range(len(games)):
                if not ns.get(gi):
                    continue
                D = np.load(games[gi]["shard"])
                n = len(D["y"])
                if games[gi].get("raw_payload") is not None and "margin_raw" in D:
                    # Row-alignment assertion: every finite raw margin in
                    # the shard must be one of this game's kept sides'
                    # declared margins (full check, not a sample).
                    _rp = games[gi]["raw_payload"]
                    if _rp["outcome"] != OUTCOME_CODES["unusable"]:
                        _vals = np.asarray(D["margin_raw"], np.float64)
                        _fin = _vals[np.isfinite(_vals)]
                        _exp = {float(_rp["mg"][s_]) for s_ in games[gi]["sides"]}
                        if len(_fin) != len(_vals) or not all(
                                any(abs(v - e) <= 1e-4 for e in _exp) for v in _fin):
                            raise ValueError(
                                f"raw label alignment failed for {games[gi]['key']}")
                for k in fields:
                    if k == "w":
                        continue  # weighted w is computed below from wts, not stored in shards
                    if k in RAW_FIELDS and k not in D:
                        mm[k][o:o + n] = _RAW_FILL[k]
                    else:
                        mm[k][o:o + n] = D[k] if k != "gid" else gi
                if weighted:
                    # rows are side-major (each side's steps ascending): a step reset starts the next side
                    st_ = D["step"].astype(np.int64)
                    seg = np.concatenate([[0], np.cumsum(np.diff(st_) < 0)])
                    mm["w"][o:o + n] = np.asarray(games[gi]["wts"], np.float32)[seg]
                o += n
            if weighted:
                w_copy = np.asarray(mm["w"], np.float32).copy()
            for k in mm:
                mm[k].flush()
            mm = {}
            gc.collect()  # release the memmaps so the rename cannot race a writer
            with open(os.path.join(staging, "meta.json"), "w") as fh:
                json.dump(meta, fh)
            man = {"generation": gen_id, "created": time.time(),
                   "cfg_fp": cfg_fp, "weighted": weighted, "total": total,
                   "valid_games": valid, "partial": bool(failed),
                   "failed_games": [f_["key"] for f_ in failed],
                   "filtered_games": filtered,
                   "terminal_games": terminal_games,
                   "terminal_short_games": terminal_short,
                   "labels": label_cov,
                   "fields": {k: {"file": k + ".npy", "rows": total,
                                  "sha256": _sha256_file(os.path.join(staging, k + ".npy"))}
                              for k in fields}}
            with open(os.path.join(staging, "manifest.json"), "w") as fh:
                json.dump(man, fh)
            if _gen_complete(gen_dir):
                shutil.rmtree(staging, ignore_errors=True)  # identical gen landed meanwhile
            else:
                shutil.rmtree(gen_dir, ignore_errors=True)
                os.replace(staging, gen_dir)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    # Derived-layer snapshot: which derived files (bytes) predate this
    # publish, and which generation was current before it -- the
    # declaration written after the pointer switch binds pre-existing
    # derived files to the generation they were derived from.
    _pre_derived = {}
    for _dn in DERIVED_FILES:
        _dp = os.path.join(out, _dn)
        if os.path.exists(_dp):
            _pre_derived[_dn] = _sha256_file(_dp)
    _prev_gen = None
    try:
        _prev_gen = json.load(open(os.path.join(out, "current.json")))["generation"]
    except Exception:
        _prev_gen = None
    # Legacy flat view: refresh from the sealed generation (arrays first,
    # meta.json last as the flat commit marker), via a temp dir so each
    # individual file still lands by os.replace. Derived files written
    # later by other tools (set_uw's w.npy/uw.npy) are NOT touched here
    # except when the build itself is weighted, in which case w.npy is a
    # build field of THIS generation and is refreshed like any other.
    tmp_out = os.path.join(out, f".tmp_publish_{os.getpid()}")
    os.makedirs(tmp_out, exist_ok=True)
    try:
        for k in fields:
            shutil.copyfile(os.path.join(gen_dir, k + ".npy"), os.path.join(tmp_out, k + ".npy"))
        shutil.copyfile(os.path.join(gen_dir, "meta.json"), os.path.join(tmp_out, "meta.json"))
        # Residual 3: stamp the FLAT copy of the meta with the generation
        # it mirrors and that generation's row count (the generation's own
        # meta.json stays pristine). A flat-view consumer can then verify
        # the mirror belongs to a single build instead of trusting file
        # presence: after a torn publish the stamp names the NEW
        # generation while current.json still certifies the old one, and
        # the field row counts disagree with the stamp.
        _fm = json.load(open(os.path.join(tmp_out, "meta.json")))
        _fm["flat_generation"] = gen_id
        _fm["flat_total"] = total
        with open(os.path.join(tmp_out, "meta.json"), "w") as fh:
            json.dump(_fm, fh)
        for k in fields:
            os.replace(os.path.join(tmp_out, k + ".npy"), os.path.join(out, k + ".npy"))
        os.replace(os.path.join(tmp_out, "meta.json"), os.path.join(out, "meta.json"))
    except Exception:
        # Residual 3: a mid-publish failure leaves the pointer certifying
        # the previous generation (resolve_dataset consumers never see a
        # mix), but the flat mirror itself may be torn (new arrays + old
        # meta). Mark it explicitly so no flat-view consumer mistakes it
        # for a fresh product; loaders additionally resolve the pointer
        # and cross-check row counts (see cnet2_train.load).
        try:
            with open(os.path.join(out, "BUILD_FAILED.json"), "w") as fh:
                json.dump({"cfg_fp": cfg_fp, "time": time.time(),
                           "failed_stage": "publish", "generation": gen_id,
                           "error": repr(sys.exc_info()[1])[:300]}, fh)
        except Exception:
            pass
        raise
    finally:
        shutil.rmtree(tmp_out, ignore_errors=True)
    if not weighted:
        # F05 unweighted-rebuild case: a flat w.npy that is byte-identical
        # to the PREVIOUS generation's build-owned w field is a stale
        # build product, not a set_uw derivative -- remove it so an
        # existence-based consumer (BC_USE_W=1) cannot silently load
        # weights this generation never produced. A w.npy whose bytes
        # differ (rewritten by set_uw after the previous build) is kept.
        try:
            _pdir, _pman = load_generation(out, verify=False)
            if _pman.get("weighted") and "w" in _pman.get("fields", {}):
                _flat_w = os.path.join(out, "w.npy")
                if os.path.exists(_flat_w) and \
                        _sha256_file(_flat_w) == _pman["fields"]["w"]["sha256"]:
                    os.remove(_flat_w)
        except Exception:
            pass
    # The pointer switch is the single atomic commit of the publish.
    ptr_tmp = os.path.join(out, f".current-{os.getpid()}.tmp")
    with open(ptr_tmp, "w") as fh:
        json.dump({"generation": gen_id, "dir": f"generations/{gen_id}",
                   "cfg_fp": cfg_fp, "total": total, "created": time.time()}, fh)
    try:
        os.replace(ptr_tmp, os.path.join(out, "current.json"))
    except Exception:
        # Residual 3: same marking discipline as the flat refresh -- a
        # failed commit must not leave an unmarked, half-published root.
        try:
            with open(os.path.join(out, "BUILD_FAILED.json"), "w") as fh:
                json.dump({"cfg_fp": cfg_fp, "time": time.time(),
                           "failed_stage": "pointer", "generation": gen_id,
                           "error": repr(sys.exc_info()[1])[:300]}, fh)
        except Exception:
            pass
        raise
    try:
        os.remove(os.path.join(out, "BUILD_FAILED.json"))
    except OSError:
        pass
    # Derived-layer declaration for whatever post-processing files sit
    # in the flat root now (never generation fields, never generation
    # content): sidecar + flat meta mirror, refreshed every publish.
    _refresh_derived_declaration(out, gen_id, _prev_gen, _pre_derived, set(fields))
    # Prune generations older than the previous one (best effort): keep
    # the current generation and the one it replaced for rollback reads.
    try:
        gens = sorted((os.path.getmtime(os.path.join(gen_root, d)), d)
                      for d in os.listdir(gen_root)
                      if d.startswith("gen-") and os.path.isdir(os.path.join(gen_root, d)))
        for _mt, d in gens[:-2]:
            if d != gen_id:
                shutil.rmtree(os.path.join(gen_root, d), ignore_errors=True)
    except Exception:
        pass
    from collections import Counter
    c = Counter(F.TEACHERS[t] for g in games if ns.get(g["gi"], 0) > 0
                for t in [teacher_id(g["names"][s]) for s in g["sides"]])
    rep = {"games": valid, "samples": total,
           "val_games": sum(g["val"] for g in games if ns.get(g["gi"], 0) > 0),
           "teacher_sides": c, "cfg_fp": cfg_fp}
    if failed:
        rep["failed_games"] = [f_["key"] for f_ in failed]
        rep["partial"] = True
    if filtered:
        rep["filtered_games"] = filtered
    if terminal_games:
        rep["terminal_games"] = terminal_games
        rep["terminal_short_games"] = terminal_short
    if raw_games:
        rep["labels"] = label_cov
    rep["generation"] = gen_id
    if weighted:
        w = w_copy
        rep["ours_loss_games"] = sum(1 for g in games if g.get("ours_loss"))
        rep["ours_loss_samples"] = int(sum(g["n"] for g in games if g.get("ours_loss")))
        rep["weight_hist"] = {str(v): int((w == v).sum()) for v in np.unique(w)}
        rep["weight_mean"] = round(float(w.mean()), 3)
    print(json.dumps(rep))


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--resolve":
        print(resolve_dataset(sys.argv[2], verify="--no-verify" not in sys.argv))
    else:
        main(sys.argv[1], sys.argv[2:])
