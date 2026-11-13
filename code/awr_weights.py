"""awr_weights DS CORPUS_DIR [CORPUS_DIR ...] -- AWR weights on a build_awr dataset.

Reward (recomputed from each replay's final state at step 100, our side minus theirs):
  cash + cargo + sum over standing trees of  4-neighbour carbon * (1 - 0.9625^min(50 - age, 30)) * p_survive,
  p_survive = 0.3 if an enemy planter stands within 1 step of the tree (it is about to be seized), else 0.8
  (2026-11-09 review: a reward taken at the step-100 instant overvalues trees planted for the opponent to take).
Whole games (300 steps in the replay, awr_gen with AWR_STEPS=300): the reward is the final cash difference instead.
Advantage: leave-one-out baseline over all games of the same seed (the greedy game and the K sampled ones):
  A_i = R_i - mean(R_j, j != i)   (lower variance than "minus the greedy game" alone).
Weights: w = exp(A / beta); beta starts at std(A) and grows until the effective sample size
  (sum w)^2 / sum w^2 is at least ESS_MIN (default 20%) of the games -- otherwise a few lucky games dominate.
uw (per unit) = 1 for planters and base slots, 0 for our collectors (cars stay the net's own).
AWR_CAR_ANCHOR=c (default 0): car tokens get the constant weight c regardless of the game's weight -- a self-anchor that
keeps the cars where they are when no separate anchor dataset is mixed in (use 1).

F02: rescore follows the canonical terminal contract (eval_v53): a replay that finished by extinction scores
(extinct: steps - EPISODE_STEPS - 1; survivor: cash carried into the terminal step), taken from the generator's
info.scores when present, else reconstructed from the first non-ACTIVE state; an extinction finish is never a
mirror game.
F06: the weight chain stays float32 end to end. ESS / w_max give no positive lower bound, and the collector
anchor's mean(w)/w inversion turns a tiny positive w into an uw that overflows float16 downstream (inf -> NaN
loss), so positive game weights are floored at AWR_WMIN (default 1e-3; mirror games keep w = 0 exactly). Every
floored game is counted and named in the output JSON -- never silent. w / uw are hard-gated for finiteness
before anything is saved: a non-finite entry aborts with the offending game keys, it is not published.
P2-3 (Codex review 20261114): w / uw are derived-layer products, never generation fields -- the same
contract set_uw already follows. On a dataset with a generation pointer the base generation is pinned
and hash-verified BEFORE any derivation input is read (inputs come from the certified generation only,
never the flat root mirror), and after the save the weights are registered via
cnet2_build.register_derived with the pinned base passed as expected_base; a loader-equivalent
identity self-check must then pass, else the tool exits non-zero. (The previous code np.save'd straight
into the dataset root with no registration: on a sealed dataset it exited 0 while cnet2_train's strict
loader refused the very weights it had just written.) Only a dataset with NO generation pointer at all
keeps the legacy path: root reads, root write, weights left UNREGISTERED with a loud warning.
R2 (Codex re-review 20261114): a WEIGHTED build owns w itself -- the generation manifest declares
"w" as a generation field, baked from the build's own per-side weights. The first P2-3 fix still
registered AWR's w as a derived file on such a dataset and its self-check passed, but the loader
then refused the dataset outright ("w.npy is BOTH a field of generation ... and a declared derived
file"): derived registration over a generation-owned field is an ownership conflict, not a
registration variant. AWR now refuses BEFORE writing any byte when a file it would register
(w / uw) is already a field of the pinned generation, and the post-registration self-check carries
the same BOTH condition the loader applies, so this tool can never exit 0 over a conflict its own
consumer rejects."""
import gzip
import hashlib
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cnet2_build import (assert_not_in_generation, load_generation,  # noqa: E402
                         register_derived)


def side_value(obs, carbon, p):
    P = obs["players"]
    cash = float(P[p][0])
    cargo = sum(float(w[1]) for w in P[p][2].values())
    epl = [int(w[0]) for w in P[1 - p][2].values() if w[2] == "PLANTER"]
    v = 0.0
    for pos, age in P[p][3].values():
        pos = int(pos); r, c = divmod(pos, 15)
        nb = sum(carbon[((r + dr) % 15) * 15 + (c + dc) % 15] for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)))
        near = any(min(abs(e // 15 - r), 15 - abs(e // 15 - r)) + min(abs(e % 15 - c), 15 - abs(e % 15 - c)) <= 1 for e in epl)
        v += nb * (1.0 - 0.9625 ** max(0, min(50 - int(age), 30))) * (0.3 if near else 0.8)
    return cash + cargo + v


def _replay_result(st, info, side):
    """Canonical terminal contract (F02) for a stored replay; returns
    (reward, whole, extinct) or None when the replay is not a finished game.
    Prefers the generator-recorded info.scores; else reconstructs from the
    first non-ACTIVE state: the extinct player is the one with no workers,
    no trees and (no base or cash below the recruit price -- carbon.json
    prices both recruits at 30, the fact eval_v53._extinct relies on); its
    score is i - EPISODE_STEPS - 1 for terminal state index i, and the
    survivor keeps its cash from the state BEFORE the terminal transition.
    A both-DONE time-out finish scores the final cash difference."""
    if info.get("scores"):
        sc = info["scores"]
        return float(sc[side]) - float(sc[1 - side]), True, info.get("terminal") == "extinction"
    for i, row in enumerate(st):
        if any(e.get("status") not in (None, "ACTIVE") for e in row):
            players = row[0]["observation"]["players"]
            extinct = [not P[2] and not P[3] and (not P[1] or float(P[0]) < 30.0) for P in players]
            if sum(extinct) == 1:
                pre = st[i - 1][0]["observation"]["players"] if i >= 1 else players
                scores = [float(i - 301) if extinct[p] else float(pre[p][0]) for p in (0, 1)]
                return float(scores[side] - scores[1 - side]), True, True
            return float(players[side][0]) - float(players[1 - side][0]), True, False
    return None


def rescore(corpus):
    games = [json.loads(l) for l in open(os.path.join(corpus, "awr.jsonl"))]
    for g in games:
        d = json.load(gzip.open(os.path.join(corpus, g["file"])))
        st = d["steps"]; info = d.get("info") or {}
        o = st[-1][0]["observation"]; carbon = o.get("full_carbon") or o["carbon"]
        rr = _replay_result(st, info, g["side"])
        if rr is not None:
            g["reward2"], g["whole"], g["extinct"] = rr
        elif len(st) >= 300 or os.environ.get("AWR_FINAL"):      # whole game: the reward is the final score difference
            g["whole"] = True; g["extinct"] = False
            g["reward2"] = float(o["players"][g["side"]][0]) - float(o["players"][1 - g["side"]][0])
        else:
            g["whole"] = False; g["extinct"] = False
            g["reward2"] = side_value(o, carbon, g["side"]) - side_value(o, carbon, 1 - g["side"])
        g["corpus"] = os.path.basename(corpus.rstrip("/"))
    by = defaultdict(list)
    for g in games:
        # a 300-step game that ends exactly level is a perfect mirror game (our net against an identical deterministic
        # copy on the 4-fold symmetric map: both sides play the same moves, units cancel on the symmetry lines, some
        # games never develop at all). It carries no information: weight 0 and not part of the seed's baseline.
        g["mirror"] = bool(g.get("whole") and not g.get("extinct") and abs(g["reward2"]) < 0.05)       # same rule as awr_signal (reward rounded to 0.1); F02: an extinction finish is never a mirror game
        if not g["mirror"]:
            by[g["seed"]].append(g)
    for g in games:
        g["adv2"] = 0.0
    for gs in by.values():
        tot = sum(x["reward2"] for x in gs)
        for x in gs:
            x["adv2"] = x["reward2"] - (tot - x["reward2"]) / max(1, len(gs) - 1) if len(gs) > 1 else 0.0
    return games


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def _derived_layer_problems(ds_abs, base_gen, gen_total, gen_fields=()):
    """Post-registration self-check (P2-3): the loader-equivalent identity
    gate -- mirrors, for the two weight files this tool just registered,
    the checks cnet2_train._validate_derived_layer applies before any
    weight enters a loss (sidecar present and mirrored into the flat
    meta; per file: field, base_generation == the pinned generation,
    rows == generation total, dtype/shape, sha256 of the file bytes).
    R2: the loader's ownership condition is mirrored too -- a file that
    is BOTH a field of the generation and a declared derived file is a
    refusal there, so it is a problem here (main() refuses such datasets
    before writing; this is the same check on the post-write state).
    Returns a list of problems; empty means the training loader accepts
    the derived layer as verified."""
    probs = []
    sp = os.path.join(ds_abs, "derived.json")
    if not os.path.exists(sp):
        return ["derived.json sidecar missing after registration"]
    try:
        side = json.load(open(sp))
    except Exception as e:  # noqa: BLE001 -- named in the refusal
        return [f"derived.json sidecar unreadable: {e}"]
    try:
        mirror = json.load(open(os.path.join(ds_abs, "meta.json"))).get("derived_layer")
    except Exception:  # noqa: BLE001
        mirror = None
    if mirror is not None and mirror != side:
        probs.append("derived.json and meta.json 'derived_layer' mirror disagree")
    entries = side.get("files", {}) if isinstance(side, dict) else {}
    for name in ("w.npy", "uw.npy"):
        e = entries.get(name)
        if e is None:
            probs.append(f"{name} is not declared in the derived.json sidecar")
            continue
        if name[:-4] in gen_fields:
            probs.append(f"{name} is BOTH a field of generation {base_gen} and a "
                         f"declared derived file (ownership conflict -- the loader "
                         f"refuses this dataset)")
            continue
        arr = np.load(os.path.join(ds_abs, name), mmap_mode="r")
        if e.get("field") != name[:-4]:
            probs.append(f"{name}: sidecar field {e.get('field')!r} != {name[:-4]!r}")
        if e.get("base_generation") != base_gen:
            probs.append(f"{name}: base_generation {e.get('base_generation')!r} != pinned {base_gen!r}")
        if int(e.get("rows", -1)) != gen_total:
            probs.append(f"{name}: sidecar rows {e.get('rows')} != generation total {gen_total}")
        if int(arr.shape[0]) != gen_total:
            probs.append(f"{name}: file rows {int(arr.shape[0])} != generation total {gen_total}")
        if e.get("dtype") != str(arr.dtype):
            probs.append(f"{name}: dtype {e.get('dtype')!r} != file {str(arr.dtype)!r}")
        if [int(x) for x in arr.shape] != [int(x) for x in e.get("shape", [])]:
            probs.append(f"{name}: shape {e.get('shape')} != file {list(arr.shape)}")
        if _sha256_file(os.path.join(ds_abs, name)) != e.get("sha256"):
            probs.append(f"{name}: sha256 mismatch (file bytes differ from registration)")
    return probs


def main():
    ds, corpora = sys.argv[1], sys.argv[2:]
    ds_abs = os.path.abspath(ds)
    assert_not_in_generation(ds_abs, "awr_weights write weights into")
    # P2-3, set_uw's accepted discipline: pin and VERIFY the base
    # generation before touching any derivation input, and refuse a torn
    # publish outright (BUILD_FAILED.json in the root means an interrupted
    # publish left the flat mirror mixing generations). The legacy
    # fallback is ONLY for a dataset with no generation pointer at all:
    # with a pointer in force, ANY load_generation failure (missing file,
    # hash mismatch, unreadable manifest) is a hard refusal BEFORE
    # anything is written, and the error names the underlying cause.
    _bf = os.path.join(ds_abs, "BUILD_FAILED.json")
    if os.path.exists(_bf):
        sys.exit(f"awr_weights: {_bf} is present -- the dataset root is a torn flat "
                 f"mirror left by an interrupted publish; rebuild the dataset "
                 f"before deriving weights (refusing to run)")
    _ptr = os.path.join(ds_abs, "current.json")
    if os.path.exists(_ptr):
        try:
            gen_dir, _man = load_generation(ds_abs, verify=True)
        except Exception as e:  # noqa: BLE001 -- the refusal must name it
            sys.exit(f"awr_weights: generation pointer {_ptr} exists but the sealed "
                     f"generation failed verification: {type(e).__name__}: {e}; "
                     f"refusing to derive weights (no legacy fallback while a "
                     f"pointer is in force)")
        base_gen = _man["generation"]; gen_total = int(_man["total"])
        gen_fields = set(_man.get("fields", {}))
        feat_dir = gen_dir          # derivation inputs: certified generation ONLY
        # R2 ownership gate (Codex re-review 20261114): a weighted build
        # OWNS w -- the manifest declares it a generation field, and the
        # loader refuses any dataset whose derived sidecar also claims
        # it ("BOTH a field of generation ... and a declared derived
        # file"). Registering AWR's weights over a generation-owned
        # field is an ownership conflict, not a registration variant:
        # refuse here, BEFORE any byte is written, naming the conflict.
        # The generation's own weights stay authoritative; deriving AWR
        # weights requires a dataset whose generation does not own them.
        _owned = [f"{k}.npy" for k in ("w", "uw") if k in gen_fields]
        if _owned:
            sys.exit("awr_weights: " + "; ".join(
                f"{n} is owned by generation {base_gen} as a field; "
                f"AWR derived registration refused" for n in _owned) +
                f" (dataset {ds_abs}: the generation's own weights are "
                f"authoritative -- rebuild without the weighted field "
                f"before deriving AWR weights)")
    else:
        # Pre-generation legacy dataset: no pointer exists, the flat root
        # is the only source there is (registration stays impossible and
        # the consumer will flag the weights unverified, as before).
        gen_dir = None; base_gen = None; gen_total = None; gen_fields = set(); feat_dir = ds_abs
    games = [g for c in corpora for g in rescore(c)]
    nmir = sum(g["mirror"] for g in games)
    A = np.array([g["adv2"] for g in games if not g["mirror"]])
    # F06 hard gate 0: a non-finite advantage (NaN reward upstream) is rejected by name here --
    # downstream, Python's min(wmax, nan) would silently turn it into wmax and publish it.
    if not np.isfinite(A).all():
        bad = [g["key"] for g in games if not g["mirror"] and not np.isfinite(g["adv2"])][:10]
        sys.exit(f"awr_weights: non-finite advantage for {bad} -- fix the corpus/rewards; not publishing")
    ess_min = float(os.environ.get("AWR_ESS_MIN", "0.2")); wmax = float(os.environ.get("AWR_WMAX", "20"))
    beta = float(A.std()) or 1.0
    for _ in range(40):
        w_ = np.minimum(wmax, np.exp(A / beta))
        ess = w_.sum() ** 2 / (w_ ** 2).sum() / len(w_)
        if ess >= ess_min:
            break
        beta *= 1.25
    wkey = {g["key"]: 0.0 if g["mirror"] else float(np.minimum(wmax, np.exp(g["adv2"] / beta))) for g in games}     # np.minimum (not min) so a NaN would propagate to gate 1 instead of becoming wmax
    meta = json.load(open(os.path.join(feat_dir, "meta.json"))); gid = np.load(os.path.join(feat_dir, "gid.npy"))
    if len(wkey) != len(games):
        sys.exit(f"awr_weights: {len(games)} games but only {len(wkey)} distinct keys -- the corpora share game keys "
                 f"(same seed range). Run awr_rekey.py on each corpus (a different TAG each) and rebuild the dataset.")
    if len(meta["games"]) != len(games):
        sys.exit(f"awr_weights: the dataset holds {len(meta['games'])} games, the corpora {len(games)} -- rebuild the dataset "
                 f"from exactly these corpora (cnet2_build drops games whose file name it has already seen).")
    gw = np.array([wkey[g["key"].replace(".json.gz", "")] for g in meta["games"]], np.float32)
    # F06 hard gate 1: a non-finite game weight (NaN advantage upstream) is rejected by name, never published.
    if not np.isfinite(gw).all():
        bad = [meta["games"][i]["key"] for i in np.where(~np.isfinite(gw))[0][:10]]
        sys.exit(f"awr_weights: non-finite game weight for {bad} (ds={ds}) -- fix the corpus/advantages; not publishing")
    # F06 floor: positive weights are floored at AWR_WMIN (mirror games keep w = 0 exactly and stay excluded).
    # Floored games are counted and named in the output JSON below -- never silent.
    w_floor = float(os.environ.get("AWR_WMIN", "1e-3"))
    fl = (gw > 0) & (gw < w_floor)
    floored_keys = [meta["games"][i]["key"] for i in np.where(fl)[0]]
    gw[fl] = np.float32(w_floor)
    w = gw[gid].astype(np.float32)
    floored_rows = int(fl[gid].sum())
    kind = np.load(os.path.join(feat_dir, "tok_kind.npy"), mmap_mode="r")
    S = np.load(os.path.join(feat_dir, "y.npy"), mmap_mode="r").shape[1]
    uw = np.ones((len(gid), S), np.float32)
    uw[:, :10] = (np.asarray(kind[:, :10]) != 0).astype(np.float32)
    # Self-anchor for the cars. With weight 0 nothing holds the car outputs while the shared decoder is trained on the
    # planter / base rows (round sp1 without an anchor set: car agreement 99.2% -> 93.0%). The cars in these games are
    # never sampled, so their labels are the net's own moves: give every car token the constant weight AWR_CAR_ANCHOR,
    # whatever the game's weight (the trainer multiplies w * uw, hence the division).
    car_anchor = float(os.environ.get("AWR_CAR_ANCHOR", "0"))
    if car_anchor > 0:
        iscar = np.asarray(kind[:, :10]) == 0
        uw[:, :10] = np.where(iscar, car_anchor / np.maximum(w, w_floor)[:, None], uw[:, :10])     # F06: inversion denominator is the floored weight, so uw stays bounded
    # F06 hard gate 2: nothing non-finite is saved -- abort naming the games whose rows carry it.
    if not (np.isfinite(w).all() and np.isfinite(uw).all()):
        rows_bad = np.unique(np.concatenate([np.where(~np.isfinite(w))[0],
                                             np.where(~np.isfinite(uw).any(axis=1))[0]]))[:10]
        keys_bad = [meta["games"][int(gid[i])]["key"] for i in rows_bad]
        sys.exit(f"awr_weights: non-finite w/uw in rows of games {keys_bad} (ds={ds}) -- not publishing")
    np.save(os.path.join(ds, "w.npy"), w); np.save(os.path.join(ds, "uw.npy"), uw)
    # Derived-layer registration (P2-3, set_uw's contract): w/uw are
    # post-processing products, never generation fields -- register them
    # in the dataset root sidecar so cnet2_train.load can verify their
    # identity (base generation, rows, sha256) before consuming them. The
    # pinned base is passed as expected_base so a pointer that moved
    # since the pin is refused inside register_derived BEFORE the
    # sidecar is written; afterwards the binding is re-resolved once
    # more, and a late move rolls the sidecar back to its previous
    # bytes. Finally the loader-equivalent self-check must pass: this
    # tool never exits 0 leaving weights its own consumer would refuse.
    if gen_dir is not None:
        _sp = os.path.join(ds_abs, "derived.json")
        _prev_side = None
        if os.path.exists(_sp):
            with open(_sp, "rb") as _fh:
                _prev_side = _fh.read()

        def _rollback_sidecar():
            if _prev_side is None:
                try:
                    os.remove(_sp)
                except OSError:
                    pass
            else:
                with open(_sp, "wb") as _fh:
                    _fh.write(_prev_side)

        _side = register_derived(ds_abs, [os.path.join(ds_abs, "uw.npy"), os.path.join(ds_abs, "w.npy")],
                                 params={"tool": "awr_weights", "beta": float(beta),
                                         "ess_min": ess_min, "w_max": wmax,
                                         "car_anchor": car_anchor, "w_floor": w_floor,
                                         "corpora": [os.path.basename(c.rstrip("/")) for c in corpora],
                                         "mirror_dropped": int(nmir),
                                         "floored_games": len(floored_keys)},
                                 expected_base=base_gen)
        try:
            _now = load_generation(ds_abs, verify=False)[1]["generation"]
        except Exception:  # noqa: BLE001
            _now = None
        if _now != base_gen:
            _rollback_sidecar()
            sys.exit(f"awr_weights: generation pointer moved during registration "
                     f"({base_gen} -> {_now}); the derived registration was "
                     f"rolled back -- rerun awr_weights")
        _probs = _derived_layer_problems(ds_abs, base_gen, gen_total, gen_fields)
        if _probs:
            _rollback_sidecar()
            sys.exit(f"awr_weights: derived layer failed the loader-equivalent "
                     f"self-check: {'; '.join(_probs)}; the derived registration "
                     f"was rolled back -- not publishing as verified")
        derived_info = {"derived_layer": _side["format"], "base_generation": base_gen,
                        "derived_check": "verified"}
    else:
        derived_info = {"derived_layer": "WARNING: no generation pointer in dataset; w/uw left "
                                         "UNREGISTERED (legacy dataset) -- consumers will flag "
                                         "derived identity as unverified",
                        "base_generation": None, "derived_check": "legacy-unverified"}
    r_old = np.array([g["reward"] for g in games]); r_new = np.array([g["reward2"] for g in games])
    print(json.dumps({"rows": int(len(gid)), "games": len(games), "mirror_dropped": int(nmir), "beta": round(beta, 1), "ess_frac": round(float(ess), 3),
                      "adv_sd": round(float(A.std()), 1), "reward_old_mean": round(float(r_old.mean()), 1),
                      "reward_new_mean": round(float(r_new.mean()), 1), "corr_old_new": round(float(np.corrcoef(r_old, r_new)[0, 1]), 3),
                      "w_max": round(float(gw.max()), 2), "car_anchor": car_anchor,
                      "w_floor": w_floor, "floored_games": len(floored_keys), "floored_keys": floored_keys[:20],
                      "floored_rows": floored_rows, **derived_info}), flush=True)


if __name__ == "__main__":
    main()
