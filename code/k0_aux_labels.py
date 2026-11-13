"""k0_aux_labels -- K0-branch auxiliary labels (death / deposit / worth-delta
prediction at 10 / 25 / 50 steps) with the three data gates the plain
aux_labels.py does not enforce. Same six output arrays, same shapes and
dtypes as aux_labels (drop-in targets), but computed against the CERTIFIED
generation of a dataset and sealed with a lineage sidecar.

Gate 1 -- first-terminal truncation. aux_labels.py takes `last = T - 1`
(the replay FILE length) as the game end, so a game that ended at frame 2
and was padded to 300 frames (the producer does exactly this, see
selfplay_gen.py: entries are padded with DONE copies of the terminal
frame) yields fully "valid" 10/25/50-step windows at root step 1 and a
worth-delta label read out of the padding. Here every window is bounded
by the game's FIRST TERMINAL frame:
  * the first frame whose seat entries carry a non-ACTIVE status, or
  * the first frame where a side is extinct under the original-env rule
    (cnet2_search._extinct, observation form: no workers AND no trees
    AND (no bases OR cash below the cheapest recruit cost)) -- replays
    in the bare aux_labels format carry no status fields, so the state
    signal is the only terminal evidence there;
  * if neither fires, the last recorded frame (cause "end_of_record").
A window [t, t+k] is valid iff t + k <= first_terminal. Padding frames
never count as steps and are never read for labels.

Gate 2 -- physical death vs terminal clearing, per UID. The replay
formats carry no death-event stream; a death is inferred from a unit
vanishing between consecutive frames, and the two kinds are told apart
by WHERE the vanishing happens:
  * vanishing at frame t < first_terminal: the game is verifiably live
    in frame t -> PHYSICAL death (the only kind the die labels use);
  * vanishing at t == first_terminal in a game that ends by extinction:
    the terminal transition IS the elimination/settlement event, and
    every simultaneous vanishing there (the eliminated side's remaining
    units above all) is TERMINAL CLEARING, never a physical death;
  * vanishing at t == first_terminal of a game that runs its full
    course (status/horizon end, no side extinct): that frame is the
    resolution of the last real step -> physical (empirically sparse,
    ~1 per game-seat in platform replays, i.e. combat, not a sweep);
  * vanishing at t > first_terminal: padding-region artifact -> clearing.
Known conservative cost: a genuine last-step kill inside an EXTINCTION
game is attributed to the settlement. The gate prefers that bias over
teaching the head that standing units die predictably at game end.

Gate 3 -- lineage binding. Labels are computed from the generation the
dataset pointer certifies (cnet2_build.load_generation with verify=True:
gid / step / tok_* / meta are read from the pinned generation directory,
never from the flat mirror), written to the dataset ROOT like the
derived layer (never into the immutable generation), and sealed with
out/k0_aux.json, format "cnet2-k0-aux/1", mirroring the cnet2-derived/1
entry shape. The sidecar binds: base_generation, row count, a row-key
fingerprint (sha256 over the pinned gid.npy + step.npy file hashes), the
sha256 of every label file, and per-game lineage {gi, key, file,
replay_sha256 (file bytes), sides, first_terminal, terminal_cause,
physical/clearing counts}. verify_aux_labels(ds) / `--verify` re-pins
the CURRENT generation and rejects on any mismatch: a moved pointer, a
tampered label row (hash), a swapped replay (hash), or a lineage set
that no longer matches the certified meta.

Usage:
  python3 k0_aux_labels.py DS [WORKERS]     write labels + sidecar
  python3 k0_aux_labels.py --verify DS      check an existing bundle
"""
import gzip
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import aux_labels as AL  # noqa: E402  (_worth + horizon constants; semantics fixed here)
import cnet2_build as B  # noqa: E402  (load_generation / _sha256_file: the certification contract)

KD = AL.KD
KP = AL.KP
AUX_FILES = ("aux_die", "aux_die_m", "aux_dep", "aux_dep_m", "aux_dw", "aux_dw_m")
SIDECAR = "k0_aux.json"
FORMAT = "cnet2-k0-aux/1"
DEFAULT_RECRUIT_COST = 30.0


def _frame_obs(entry):
    e = entry[0] if isinstance(entry, (list, tuple)) else entry
    return e.get("observation") if isinstance(e, dict) else None


def _side_extinct(P, cheapest):
    """Observation form of cnet2_search._extinct for one player record
    P = [cash, bases, units, trees]."""
    if len(P[2]) > 0 or len(P[3]) > 0:
        return False
    if len(P[1]) == 0:
        return True
    return float(P[0]) < cheapest


def first_terminal(steps, cheapest=DEFAULT_RECRUIT_COST):
    """-> (index, cause): the first frame at which the game is over.
    cause is "status" (a seat entry first reports non-ACTIVE),
    "extinction" (a side first meets the _extinct rule), or
    "end_of_record" (neither fired; index is the last frame)."""
    for t, entry in enumerate(steps):
        entries = entry if isinstance(entry, (list, tuple)) else [entry]
        if any(isinstance(e, dict) and e.get("status") not in (None, "ACTIVE")
               for e in entries):
            return t, "status"
        obs = _frame_obs(entry)
        if obs is not None:
            players = obs.get("players")
            if players:
                for P in players:
                    if _side_extinct(P, cheapest):
                        return t, "extinction"
    return len(steps) - 1, "end_of_record"


def _classify(vanish_t, ft, extinct_at_ft):
    """Gate 2: classify one unit vanishing at frame vanish_t."""
    if vanish_t < ft:
        return "physical"
    if vanish_t > ft:
        return "clearing"
    return "clearing" if extinct_at_ft else "physical"


def game_tables_k0(path, p):
    """Per-step tables of side p with gate 1/2 semantics.
    -> pos2uid, phys_death[uid], dep_events, worth_margin, has_full,
       ft, cause, n_physical, n_clearing"""
    raw = json.load(gzip.open(path))
    st = raw["steps"]
    cfg = raw.get("configuration") or {}
    cheapest = min(float(cfg.get("recCollectorCost", DEFAULT_RECRUIT_COST)),
                   float(cfg.get("recPlanterCost", DEFAULT_RECRUIT_COST)))
    ft, cause = first_terminal(st, cheapest)
    obs_ft = _frame_obs(st[ft])
    extinct_at_ft = bool(obs_ft and obs_ft.get("players")) and any(
        _side_extinct(P, cheapest) for P in obs_ft["players"])
    T = len(st)
    last = T - 1
    pos2uid = []
    death = {}
    dep = {}
    wm = np.zeros(T)
    prev = None
    has_full = True
    n_phys = 0
    n_clear = 0
    for t in range(T):
        o = st[t][0]["observation"]
        P = o["players"][p]
        W = P[2]
        full = o.get("full_carbon")
        if full is None:
            has_full = False
        wm[t] = AL._worth(o, full, p, last - t) - AL._worth(o, full, 1 - p, last - t)
        pos2uid.append({int(w[0]): (u, w[2]) for u, w in W.items()})
        bases = {int(b) for b in P[1].values()}
        if prev is not None:
            for u in prev:
                if u not in W and u not in death:
                    if _classify(t, ft, extinct_at_ft) == "physical":
                        death[u] = t
                        n_phys += 1
                    else:
                        n_clear += 1
            if t <= ft:
                for u, w in W.items():
                    if u in prev and w[2] == "COLLECTOR" and prev[u][1] > 5 and w[1] < 0.5 \
                            and int(w[0]) in bases:
                        dep.setdefault(u, []).append((t, float(prev[u][1])))
        prev = W
    return pos2uid, death, dep, wm, has_full, ft, cause, n_phys, n_clear


def one(task):
    path, p, steps, tok_pos, tok_mask, tok_kind = task
    import cnet2_feat as F
    n = len(steps)
    if not path.endswith(".json.gz"):              # no trajectory: masks stay False (as aux_labels)
        z = lambda sh, dt: np.zeros(sh, dt)
        return (z((n, 10, 3), np.uint8), z((n, 10, 3), bool), z((n, 10, 2), np.float16),
                z((n, 10, 2), bool), z((n, 3), np.float16), z((n, 3), bool), 0, None, None)
    pos2uid, death, dep, wm, has_full, ft, cause, n_phys, n_clear = game_tables_k0(path, p)
    die = np.zeros((n, 10, 3), np.uint8)
    die_m = np.zeros((n, 10, 3), bool)
    dp = np.zeros((n, 10, 2), np.float16)
    dp_m = np.zeros((n, 10, 2), bool)
    dw = np.zeros((n, 3), np.float16)
    dw_m = np.zeros((n, 3), bool)
    miss = 0
    for i, t in enumerate(steps):
        t = int(t)
        if t > ft:
            continue                               # root itself sits in the padding: nothing is real
        for j, k in enumerate(KD):
            if t + k <= ft:
                dw[i, j] = np.clip((wm[t + k] - wm[t]) / 1000.0, -3, 3)
                dw_m[i, j] = True
        for s in range(10):
            if not tok_mask[i, s]:
                continue
            cp = int(tok_pos[i, s])
            pos = int(F.CANON[cp]) if p else cp
            hit = pos2uid[t].get(pos)
            if hit is None:
                miss += 1
                continue
            u, kind = hit
            for j, k in enumerate(KD):
                if t + k <= ft:
                    die_m[i, s, j] = True
                    die[i, s, j] = 1 if (u in death and death[u] <= t + k) else 0
            if kind == "COLLECTOR":
                for j, k in enumerate(KP):
                    if t + k <= ft:
                        dp_m[i, s, j] = True
                        dp[i, s, j] = min(sum(a for (td, a) in dep.get(u, ()) if t < td <= t + k) / 100.0, 60.0)
    stats = {"ft": int(ft), "cause": cause, "physical": int(n_phys), "clearing": int(n_clear)}
    return die, die_m, dp, dp_m, dw, dw_m, miss, has_full, stats


def _row_key(gen_dir, man):
    fields = man.get("fields", {})
    parts = []
    for name in ("gid", "step"):
        info = fields.get(name)
        if info is None:
            raise ValueError(f"generation manifest lacks field {name!r}: cannot bind row order")
        parts.append(B._sha256_file(os.path.join(gen_dir, info["file"])))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def label_dataset(ds, workers=None):
    """Compute the six aux arrays for the dataset's CERTIFIED generation
    and write them + the lineage sidecar into the dataset root."""
    ds = os.path.abspath(ds)
    gen_dir, man = B.load_generation(ds, verify=True)
    gen_id = man["generation"]
    total = int(man["total"])
    meta = json.load(open(os.path.join(gen_dir, "meta.json")))
    gid = np.load(os.path.join(gen_dir, "gid.npy"))
    step = np.load(os.path.join(gen_dir, "step.npy"))
    tok_pos = np.load(os.path.join(gen_dir, "tok_pos.npy"), mmap_mode="r")
    tok_mask = np.load(os.path.join(gen_dir, "tok_mask.npy"), mmap_mode="r")
    tok_kind = np.load(os.path.join(gen_dir, "tok_kind.npy"), mmap_mode="r")
    if len(gid) != total:
        raise ValueError(f"generation {gen_id}: gid has {len(gid)} rows, manifest total {total}")
    games = {g["gi"]: g for g in meta["games"]}
    N = len(gid)
    tasks = []
    spans = []
    i = 0
    while i < N:                                   # rows of one (game, side): contiguous, steps increasing
        g = int(gid[i])
        j = i + 1
        while j < N and gid[j] == g and step[j] > step[j - 1]:
            j += 1
        k = sum(1 for (a, b, gg) in spans if gg == g)
        sides = games[g]["sides"]
        if k >= len(sides):
            raise SystemExit(f"k0_aux_labels: game {g} has more row blocks than teacher sides")
        tasks.append((games[g]["file"], sides[k], np.asarray(step[i:j]),
                      np.asarray(tok_pos[i:j, :10]), np.asarray(tok_mask[i:j, :10]),
                      np.asarray(tok_kind[i:j, :10])))
        spans.append((i, j, g))
        i = j
    out = {"aux_die": np.zeros((N, 10, 3), np.uint8), "aux_die_m": np.zeros((N, 10, 3), bool),
           "aux_dep": np.zeros((N, 10, 2), np.float16), "aux_dep_m": np.zeros((N, 10, 2), bool),
           "aux_dw": np.zeros((N, 3), np.float16), "aux_dw_m": np.zeros((N, 3), bool)}
    miss = 0
    nofull = 0
    skipped = 0
    per_game = {}
    W = workers or (os.cpu_count() or 4)
    with ProcessPoolExecutor(W) as ex:
        for (a, b, g), r in zip(spans, ex.map(one, tasks, chunksize=4)):
            for nm, arr in zip(AUX_FILES, r[:6]):
                out[nm][a:b] = arr
            miss += r[6]
            if r[7] is None:
                skipped += 1
            elif not r[7]:
                nofull += 1
            if r[8] is not None:
                acc = per_game.setdefault(g, {"physical": 0, "clearing": 0})
                acc["physical"] += r[8]["physical"]
                acc["clearing"] += r[8]["clearing"]
                acc["ft"] = r[8]["ft"]
                acc["cause"] = r[8]["cause"]
    files_meta = {}
    for nm, arr in out.items():
        fp = os.path.join(ds, nm + ".npy")
        tmp = fp + f".tmp-{os.getpid()}"
        with open(tmp, "wb") as fh:
            np.save(fh, arr)
        os.replace(tmp, fp)
        files_meta[nm + ".npy"] = {
            "field": nm, "dtype": str(arr.dtype), "rows": int(arr.shape[0]),
            "shape": [int(x) for x in arr.shape], "sha256": B._sha256_file(fp),
            "base_generation": gen_id, "params": {"tool": "k0_aux_labels"}}
    lineage = []
    for g in sorted(games):
        gm = games[g]
        entry = {"gi": int(g), "key": gm.get("key") or os.path.basename(gm["file"]),
                 "file": gm["file"], "sides": [int(s) for s in gm["sides"]],
                 "meta_terminal": gm.get("terminal"), "meta_end_reason": gm.get("end_reason")}
        if gm["file"].endswith(".json.gz") and os.path.exists(gm["file"]):
            entry["replay_sha256"] = B._sha256_file(gm["file"])
        acc = per_game.get(g)
        if acc is not None:
            entry["first_terminal"] = acc["ft"]
            entry["terminal_cause"] = acc["cause"]
            entry["physical_deaths"] = acc["physical"]
            entry["clearing_deaths"] = acc["clearing"]
        lineage.append(entry)
    side = {"format": FORMAT, "tool": "k0_aux_labels", "base_generation": gen_id,
            "rows": int(N), "row_key": {"fields": ["gid", "step"],
                                        "sha256": _row_key(gen_dir, man)},
            "horizons": {"die": list(KD), "dep": list(KP), "dw": list(KD)},
            "files": files_meta, "games": lineage,
            "summary": {"slots_unmatched": int(miss), "games_without_full_carbon": int(nofull),
                        "blocks_not_replay": int(skipped),
                        "physical_deaths": sum(v["physical"] for v in per_game.values()),
                        "clearing_deaths": sum(v["clearing"] for v in per_game.values())}}
    sp = os.path.join(ds, SIDECAR)
    tmp = sp + f".tmp-{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(side, fh)
    os.replace(tmp, sp)
    return side


def verify_aux_labels(ds):
    """Re-pin the dataset's CURRENT certified generation and check the
    k0 aux bundle against it. Returns {"ok": bool, "errors": [...], ...};
    every failed binding is named, nothing is silently skipped."""
    ds = os.path.abspath(ds)
    errors = []
    sp = os.path.join(ds, SIDECAR)
    if not os.path.exists(sp):
        return {"ok": False, "errors": [f"sidecar missing: {sp}"]}
    try:
        side = json.load(open(sp))
    except Exception as e:
        return {"ok": False, "errors": [f"sidecar unreadable: {e!r}"]}
    if side.get("format") != FORMAT:
        errors.append(f"sidecar format {side.get('format')!r} != {FORMAT!r}")
    try:
        gen_dir, man = B.load_generation(ds, verify=True)
    except Exception as e:
        return {"ok": False, "errors": errors + [f"generation does not certify: {e!r}"]}
    gen_id = man["generation"]
    if side.get("base_generation") != gen_id:
        errors.append(f"base_generation {side.get('base_generation')!r} != current "
                      f"certified generation {gen_id!r} (pointer moved or bundle re-bound)")
    if side.get("rows") != int(man["total"]):
        errors.append(f"rows {side.get('rows')} != generation total {man['total']}")
    try:
        rk = _row_key(gen_dir, man)
        if (side.get("row_key") or {}).get("sha256") != rk:
            errors.append("row_key fingerprint mismatch: generation row order "
                          "(gid/step) differs from the labelled rows")
    except Exception as e:
        errors.append(f"row_key cannot be recomputed: {e!r}")
    for name, ent in sorted((side.get("files") or {}).items()):
        fp = os.path.join(ds, name)
        if not os.path.exists(fp):
            errors.append(f"label file missing: {name}")
            continue
        if B._sha256_file(fp) != ent.get("sha256"):
            errors.append(f"label file hash mismatch: {name} (contents changed since labelling)")
            continue
        arr = np.load(fp, mmap_mode="r")
        if [int(x) for x in arr.shape] != ent.get("shape") or str(arr.dtype) != ent.get("dtype"):
            errors.append(f"label file shape/dtype mismatch: {name}")
    try:
        meta = json.load(open(os.path.join(gen_dir, "meta.json")))
        games = {g["gi"]: g for g in meta["games"]}
    except Exception as e:
        games = {}
        errors.append(f"generation meta unreadable: {e!r}")
    lin = {e.get("gi"): e for e in side.get("games") or []}
    if set(lin) != set(games):
        errors.append(f"lineage game set {sorted(lin)} != generation meta games {sorted(games)}")
    for gi, ent in sorted(lin.items()):
        gm = games.get(gi)
        if gm is None:
            continue
        if ent.get("file") != gm["file"]:
            errors.append(f"game {gi}: lineage file {ent.get('file')!r} != meta file {gm['file']!r}")
            continue
        want = ent.get("replay_sha256")
        if want is not None:
            if not os.path.exists(gm["file"]):
                errors.append(f"game {gi}: replay file missing: {gm['file']}")
            elif B._sha256_file(gm["file"]) != want:
                errors.append(f"game {gi}: replay hash mismatch (replay replaced since labelling)")
    return {"ok": not errors, "errors": errors, "generation": gen_id,
            "rows": int(man["total"]), "files": sorted((side.get("files") or {}).keys())}


def main():
    args = sys.argv[1:]
    if args and args[0] == "--verify":
        res = verify_aux_labels(args[1])
        print(json.dumps(res, indent=1))
        raise SystemExit(0 if res["ok"] else 1)
    ds = args[0]
    W = int(args[1]) if len(args) > 1 else None
    side = label_dataset(ds, W)
    print(json.dumps({"generation": side["base_generation"], "rows": side["rows"],
                      "sidecar": SIDECAR, "summary": side["summary"],
                      "games": [{"gi": e["gi"], "first_terminal": e.get("first_terminal"),
                                 "terminal_cause": e.get("terminal_cause"),
                                 "physical_deaths": e.get("physical_deaths"),
                                 "clearing_deaths": e.get("clearing_deaths")}
                                for e in side["games"]]}))


if __name__ == "__main__":
    main()
