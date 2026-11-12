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
"""

import gzip
import json
import os
import sys
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
    path, sides, gi, tids, margins, out = task
    d = json.load(gzip.open(path))
    st = d["steps"]
    rows = {k: [] for k in FIELDS}
    for p, tid, mg in zip(sides, tids, margins):
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
    arr = {k: np.asarray(v).astype(FIELDS[k][0]) for k, v in rows.items()}
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


def scan(dirs, min_steps=200, weighted=False):
    """weighted: per-side sample weight = (2 if side won) * (1.5 if both rated >= STRONG);
    dirs whose name contains 'ours_corpus' contribute only games our bot LOST, teacher = winning opponent
    (rated >= OURS_MIN_OPP or a listed teacher)."""
    games, seen = [], set()
    R = ratings(dirs) if weighted else {}
    for rd in dirs:
        ours = "ours_corpus" in rd
        idx = json.load(open(os.path.join(rd, "index.json")))
        for g in idx:
            key = os.path.basename(g["file"])
            if key in seen or g.get("steps", 0) < min_steps:
                continue
            fp = os.path.join(rd, g["file"])
            if not os.path.exists(fp):
                continue
            order = g["order"]
            names = [g["strategies"][order[s]]["name"] for s in (0, 1)]
            scores = [g["scores"][order[s]] for s in (0, 1)]
            if ours:
                us = [s for s in (0, 1) if names[s].startswith(OUR_NAMES)]
                if len(us) != 1:
                    continue
                o = 1 - us[0]
                if not (scores[o] > scores[us[0]] and (names[o] in F.TEACHERS[:-1] or R.get(names[o], 0) >= OURS_MIN_OPP)):
                    continue
                sides = [o]
            else:
                sides = [s for s in (0, 1) if names[s] in F.TEACHERS[:-1]]
            if not sides:
                continue
            seen.add(key)
            g = {"file": fp, "key": key, "names": names, "scores": scores, "sides": sides,
                 "val": zlib.crc32(key.encode()) % 10 == 0, "src": os.path.basename(rd.rstrip("/"))}
            if weighted:
                strong = min(R.get(names[0], 0), R.get(names[1], 0)) >= STRONG
                g["wts"] = [(2.0 if scores[s] > scores[1 - s] else 1.0) * (1.5 if strong else 1.0) for s in sides]
                g["ours_loss"] = ours
            games.append(g)
    return games


def _game_content_key(g, tids, mg):
    """F05/R08: per-game content key -- every index-derived input that changes
    shard CONTENT: the kept sides, their teacher ids, the margins computed
    from index scores, and (weighted mode) the per-side sample weights. The
    replay file itself is covered by its (size, mtime_ns) stat in the
    manifest. Before this, the cache key was only the replay stat, so editing
    index.json scores silently reused shards whose value labels were computed
    from the old margins."""
    import hashlib
    detail = {"key": g["key"], "names": g["names"], "scores": g["scores"],
              "sides": g["sides"], "tids": list(tids),
              "mg": [float(x) for x in mg],
              "wts": g.get("wts"), "ours_loss": bool(g.get("ours_loss", False))}
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
        "feat_sig": [F.P, F.N, F.NS, F.T, F.H, F.OWN, F.UF, F.MF],
    }
    for mod in (__file__, getattr(F, "__file__", None)):
        try:
            detail["src_" + os.path.basename(mod)] = hashlib.sha256(open(mod, "rb").read()).hexdigest()[:16]
        except Exception:
            pass
    return hashlib.sha256(json.dumps(detail, sort_keys=True).encode()).hexdigest()[:16]


def main(out, dirs):
    os.makedirs(os.path.join(out, "shards"), exist_ok=True)
    weighted = os.environ.get("CNET2_WEIGHTED") == "1"
    games = scan(dirs, weighted=weighted)
    fields = dict(FIELDS)
    if weighted:
        fields["w"] = (np.float16, ())
    tasks = []
    for gi, g in enumerate(games):
        g["gi"] = gi
        tids = [teacher_id(g["names"][s]) for s in g["sides"]]
        mg = [g["scores"][s] - g["scores"][1 - s] for s in g["sides"]]
        suffix = ("_s" + "".join(map(str, g["sides"]))) if weighted else ""
        g["shard"] = os.path.join(out, "shards", g["key"].replace(".json.gz", "") + suffix + ".npz")
        tasks.append((g["file"], g["sides"], gi, tids, mg, g["shard"]))
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
        ck = _game_content_key(games[gi], t[3], t[4])
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
    failed = [{"gi": gi, "key": games[gi]["key"], "error": errs[gi]} for gi in sorted(errs)]
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
        # Do not leave a stale complete-looking product behind: a previous
        # successful run's meta.json / merged npy files in this out dir
        # would otherwise be mistaken for this run's output. Shards and the
        # manifest are kept (they certify only good games) so a retry after
        # fixing the bad replay reuses them.
        for name in ["meta.json"] + [k + ".npy" for k in fields]:
            try:
                os.remove(os.path.join(out, name))
            except OSError:
                pass
        sys.exit(1)
    total = sum(ns.values())
    mm = {k: np.lib.format.open_memmap(os.path.join(out, k + ".npy"), mode="w+", dtype=dt,
                                       shape=(total,) + sh) for k, (dt, sh) in fields.items()}
    o = 0
    for gi in range(len(games)):
        if not ns.get(gi):
            continue
        D = np.load(games[gi]["shard"])
        n = len(D["y"])
        for k in FIELDS:
            mm[k][o:o + n] = D[k] if k != "gid" else gi
        if weighted:
            # rows are side-major (each side's steps ascending): a step reset starts the next side
            st_ = D["step"].astype(np.int64)
            seg = np.concatenate([[0], np.cumsum(np.diff(st_) < 0)])
            mm["w"][o:o + n] = np.asarray(games[gi]["wts"], np.float32)[seg]
        o += n
    for k in mm:
        mm[k].flush()
    for g in games:
        g["n"] = ns.get(g["gi"], 0)
        if g["gi"] in errs:
            g["failed"] = errs[g["gi"]]
    json.dump({"games": games, "total": total, "teachers": F.TEACHERS, "cfg_fp": cfg_fp,
               "failed_games": failed, "partial": bool(failed),
               "valid_games": valid},
              open(os.path.join(out, "meta.json"), "w"))
    from collections import Counter
    c = Counter(F.TEACHERS[t] for g in games if ns.get(g["gi"], 0) > 0
                for t in [teacher_id(g["names"][s]) for s in g["sides"]])
    rep = {"games": valid, "samples": total,
           "val_games": sum(g["val"] for g in games if ns.get(g["gi"], 0) > 0),
           "teacher_sides": c, "cfg_fp": cfg_fp}
    if failed:
        rep["failed_games"] = [f_["key"] for f_ in failed]
        rep["partial"] = True
    if weighted:
        w = np.asarray(mm["w"], np.float32)
        rep["ours_loss_games"] = sum(1 for g in games if g.get("ours_loss"))
        rep["ours_loss_samples"] = int(sum(g["n"] for g in games if g.get("ours_loss")))
        rep["weight_hist"] = {str(v): int((w == v).sum()) for v in np.unique(w)}
        rep["weight_mean"] = round(float(w.mean()), 3)
    print(json.dumps(rep))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
