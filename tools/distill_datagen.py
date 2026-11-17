"""distill_datagen.py -- small-teacher distillation data generation (stage 2 harness).

Registration: reports/REGISTRATION_SMALL_TEACHER_DISTILL_20261117.md (9bc75b46);
design: reports/DESIGN_SMALL_TEACHER_20261117.md (cb6efbe7) section 2, as signed
off in reports/SIGNOFF_SMALL_TEACHER_DESIGN_20261117.md (5fcdaab4).

Two routes produce (x, aux) positions from the Stage B train pool, labelled with
the CURRENT teacher's own per-position outputs (labels / logits / u / value --
the teacher himself, NOT search-modified targets; registration section 2):

  replay  (route A): per game, replay the recorded selfplay with both seats'
          Feat trackers observing every step exactly once (the gate's
          extract_scene discipline), collecting (x, aux) at every step x seat.
  harvest (route B): run the pinned evaluate_root (K=16/M=8/H=2, belief=sample,
          teacher_style=2) on selected roots of selected games with a
          CachePolicy subclass whose act() records a private copy of every
          (x, aux, teacher) it is asked to evaluate -- the branch-world
          positions route A cannot see. The capture hook lives ONLY in this
          tool; no production file is modified. Guard re-decodes inside
          evaluate_root go through enc_for/decode_b (not act), so they add no
          new positions beyond the root itself, which act() already captured.

Labels come from an OFFLINE batched path (never the pinned annotate path):
  --labeler np-batch (DEFAULT): cnet2_search.encode_b/decode_b on CPU,
          multi-core via --workers (per-game processes).
  --labeler torch-gpu: NON-DEFAULT option retained by flag only (owner seat
          principle 2026-11-17 08:54/08:55: annotation is CPU-fleet work;
          the GPU path must not be the built-in default). cnet2_model.CNet2
          on --device, batched encode plus a batched autoregressive decode
          mirroring Policy.act's choices.
Both produce t_logits [n,12,5] fp32, t_labels [n,12] int64 (the decode's own
chosen labels; equal to argmax of the masked logits), t_u, t_value.

SEAT / SHARDING (owner seat principle 2026-11-17 08:54, corrected 08:55):
generation runs on the CPU spot fleet (4 machines: spot-34 + spot-36/37/38;
spot-34 joins when its exam seat is idle -- EXAMS PREEMPT its shard).
--shard-count / --shard-index split the kept game list deterministically
(sorted order, strided) across however many seats the dispatch assigns
(3 or 4 -- a pure CLI parameter, never hardcoded). Shards are IDEMPOTENT
and RESUMABLE: each game's shard is written via tmp+rename and recorded in
the shard manifest with its md5; re-running a shard (e.g. after spot-34 is
preempted by an exam) skips games whose shard md5 still matches, so landed
games are never recomputed. Assembly/merge runs on the L20 side:
  distill_datagen.py --assemble --shard-dir D1 [--shard-dir D2 ...] \
      --out-dir MERGED
verifies cross-shard identity (same code/teacher/exclusion/shard-plan),
pairwise game disjointness and every shard file's md5, then copies the
shards into one merged output dir with a merged manifest.json that
distill_train.py consumes directly.

EXCLUSION HARD GATE (fail-closed, sign-off section 3 item 7):
  --slice-file and --slice-md5 are BOTH required. At startup the slice file's
  md5 is verified BEFORE anything is loaded; a mismatch aborts the run.
  The exclusion set = slice games + appendix games (default 994400/994401,
  the section-3 measurement set) + any --registered-set files. Candidate
  games are filtered against it; the manifest records the check verbatim
  (overlap after filtering must be 0). A game explicitly named via
  --only-games that falls in the exclusion set aborts the run.

NO PATH IS HARDCODED: --corpus-root, --train-split, --slice-file and
--out-dir are all required CLI parameters (the train replays are delivered
to the executing seat separately; their landing point is not known here).

Output: one npz shard per game in --out-dir + manifest.json (rewritten
atomically after every game; --resume skips games whose shard md5 still
matches the manifest entry).

Shard schema (per game):
  x_maps uint8 [n,33,225]; x_scal [n,20]; x_tok_pos/kind/mask/cargo [n,24];
  x_hist_cell/cargo/move [n,24,8]; x_cash/x_slots/x_left [n];
  aux_<12 keys> [n,225] (native Feat dtypes);
  t_logits fp32 [n,12,5]; t_labels int64 [n,12]; t_u fp32 [n]; t_value fp32 [n];
  meta_game [n] str; meta_step int32 [n]; meta_seat int8 [n];
  meta_source [n] str in {replay, harvest}; meta_teacher_style int8 [n];
  shard_meta: 0-d JSON string (game, route, teacher identity, code md5s).
"""

import argparse
import gzip
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

X_FIELDS = ("maps", "scal", "tok_pos", "tok_kind", "tok_mask", "tok_cargo",
            "hist_cell", "hist_cargo", "hist_move")
AUX_FIELDS = ("est", "en_min", "en_max", "en_pl", "thr", "dob", "own_base",
              "en_base", "own_tree", "en_tree", "toll", "own_adj")
TEACHER_CODE_MD5 = "c6ade5ced42e2c54a77c1499163aa151"
TEACHER_WEIGHTS_MD5 = "06a03d9cfeec239af3065a49fefec382"


def die(msg):
    print(f"distill_datagen: REFUSING TO RUN: {msg}", file=sys.stderr)
    sys.exit(2)


def md5_bytes(b):
    return hashlib.md5(b).hexdigest()


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_id(gid):
    """Game identity for exclusion matching: basename, replay suffixes and the
    selfplay- prefix stripped, so 'selfplay-994400', '994400' and a corpus file
    '<hash>-1.json.gz' all compare on their core id."""
    s = str(gid).replace("\\", "/").rsplit("/", 1)[-1]
    for suf in (".json.gz", ".gz", ".json"):
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    if s.endswith("-1"):
        s = s[:-2]
    if s.startswith("selfplay-"):
        s = s[len("selfplay-"):]
    return s


# ---------------------------------------------------------------- exclusions

def load_id_file(path):
    """Ids from a JSON list / {'games': [...]} (entries str or {'id': ...})
    or a newline text file. Returns list of raw id strings."""
    raw = open(path, "rb").read()
    try:
        obj = json.loads(raw.decode("utf-8"))
    except Exception:
        return [ln.strip() for ln in raw.decode("utf-8").splitlines() if ln.strip()]
    if isinstance(obj, dict):
        obj = obj.get("games", [])
    out = []
    for e in obj:
        out.append(str(e["id"]) if isinstance(e, dict) else str(e))
    return out


def load_exclusions(slice_file, slice_md5_expected, appendix_ids, registered_files):
    """Fail-closed exclusion assembly. Verifies the slice file md5 BEFORE
    loading it. Returns (excluded_normalized_ids, record dict)."""
    if not slice_file or not slice_md5_expected:
        die("--slice-file and --slice-md5 are both required (sign-off #7 hard gate)")
    if not os.path.isfile(slice_file):
        die(f"slice file not found: {slice_file}")
    actual = md5_file(slice_file)
    if actual != slice_md5_expected.lower():
        die(f"slice file md5 mismatch: expected {slice_md5_expected}, got {actual}; "
            "refusing to load any exclusion set from an unverified file")
    slice_ids = load_id_file(slice_file)
    if not slice_ids:
        die("slice file parsed to zero game ids")
    excluded = {normalize_id(g) for g in slice_ids}
    record = {"slice_file": slice_file, "slice_md5_expected": slice_md5_expected.lower(),
              "slice_md5_actual": actual, "n_slice_games": len(slice_ids),
              "appendix_ids": sorted(app_index for app_index in appendix_ids),
              "registered_sets": []}
    for g in appendix_ids:
        excluded.add(normalize_id(g))
    for path in registered_files or []:
        ids = load_id_file(path)
        for g in ids:
            excluded.add(normalize_id(g))
        record["registered_sets"].append(
            {"file": path, "md5": md5_file(path), "n_games": len(ids)})
    record["n_excluded_total"] = len(excluded)
    return excluded, record


def check_candidates(candidate_ids, excluded, only_games=None):
    """Filter candidates against the exclusion set; returns (kept, record).
    An explicitly requested (--only-games) excluded game is a hard abort."""
    if only_games:
        bad = [g for g in only_games if normalize_id(g) in excluded]
        if bad:
            die(f"--only-games names excluded games {bad[:5]}; refusing to run")
    kept = [g for g in candidate_ids if normalize_id(g) not in excluded]
    overlap_after = [g for g in kept if normalize_id(g) in excluded]
    record = {"n_candidates": len(candidate_ids),
              "n_excluded_present": len(candidate_ids) - len(kept),
              "n_kept": len(kept), "overlap_after_filter": len(overlap_after)}
    if overlap_after:
        die("internal error: overlap after filtering is nonzero")
    return kept, record


# ---------------------------------------------------------------- game pool

def load_train_split(path):
    """Train split: JSON list / {'games': [...]} of ids or {'id','file'}
    entries, or newline ids. Returns (ids, {id: file_hint or None})."""
    raw = open(path, "rb").read()
    hints = {}
    try:
        obj = json.loads(raw.decode("utf-8"))
    except Exception:
        ids = [ln.strip() for ln in raw.decode("utf-8").splitlines() if ln.strip()]
        return ids, hints
    if isinstance(obj, dict):
        obj = obj.get("games", [])
    ids = []
    for e in obj:
        if isinstance(e, dict):
            gid = str(e["id"])
            ids.append(gid)
            if e.get("file"):
                hints[gid] = str(e["file"])
        else:
            ids.append(str(e))
    return ids, hints


def resolve_game_path(gid, hint, corpus_root):
    """Resolution order: hint as given -> hint basename under corpus root ->
    glob <normid>*.json.gz under corpus root (recursive, sorted)."""
    cands = []
    if hint:
        cands.append(hint)
        cands.append(os.path.join(corpus_root, os.path.basename(hint)))
    norm = normalize_id(gid)
    hits = []
    for dirpath, _dirs, files in os.walk(corpus_root):
        for fn in files:
            if fn.endswith(".json.gz") and normalize_id(fn) == norm:
                hits.append(os.path.join(dirpath, fn))
    cands.extend(sorted(hits))
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return None


def load_game_steps(path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        d = json.load(fh)
    return d["steps"]


# ---------------------------------------------------------------- collectors

def collect_replay_positions(F, FastCarbon, CFG, steps):
    """Route A: replay with both seats' Feat trackers, one observe per step.
    Returns list of position dicts {x, aux, step, seat}."""
    from stage1_teacher_gate import first_terminal_step as gate_ft
    ft = gate_ft(steps)
    n_steps = min(len(steps) - 1, ft)
    sim = FastCarbon(CFG).reset(seed=1, full_carbon=steps[0][0]["observation"]["full_carbon"])
    feats = [F.Feat(0), F.Feat(1)]
    out = []

    def snap(step):
        for seat in (0, 1):
            x = feats[seat].observe(sim.obs(seat))
            out.append({"x": {k: np.array(x[k], copy=True) for k in X_FIELDS},
                        "cash": x["cash"], "slots": x["slots"], "left": x["left"],
                        "aux": {k: np.array(feats[seat].aux[k], copy=True) for k in AUX_FIELDS},
                        "step": step, "seat": seat, "source": "replay"})
    snap(0)
    for t in range(n_steps):
        sim.step(steps[t + 1][0].get("action") or {}, steps[t + 1][1].get("action") or {})
        snap(t + 1)
    return out


def collect_harvest_positions(mods, game_path, game_id, steps_wanted, seed_teacher=7000,
                              K=16, M=8, H=2, teacher_style=2):
    """Route B: evaluate_root on the wanted (step, seat) roots with an
    act-recording CachePolicy subclass. Returns (positions, n_roots_run)."""
    G, T = mods["gate"], mods["teacher"]
    CachePolicy = mods["CachePolicy"]
    captured = []

    class CapPol(CachePolicy):
        def act(self, x, aux, **kw):
            captured.append({"x": {k: np.array(x[k], copy=True) for k in X_FIELDS},
                             "cash": x["cash"], "slots": x["slots"], "left": x["left"],
                             "aux": {k: np.array(aux[k], copy=True) for k in AUX_FIELDS},
                             "step": -1, "seat": -1, "source": "harvest"})
            return super().act(x, aux, **kw)

    pol = CapPol(mods["weights"])
    n_roots = 0
    for (step, me) in steps_wanted:
        scene = G.extract_scene({"game": game_id, "path": game_path, "step": step, "me": me})
        before = len(captured)
        r = T.evaluate_root(G._fresh_sim(scene), me, pol,
                            scene["x_me"], scene["aux_me"], scene["x_opp"], scene["aux_opp"],
                            G._ns(scene["fm"]), G._ns(scene["fo"]),
                            teacher=teacher_style, K=K, M=M, H=H,
                            seed_base=seed_teacher + step,
                            feats=(scene["fm"], scene["fo"]),
                            guard_snaps=G._teacher_guard_snaps(scene, "sample"),
                            belief="sample")
        G._expect_teacher(r, "sample")
        for p in captured[before:]:
            p["step"] = step
            p["seat"] = me
        n_roots += 1
    return captured, n_roots


# ---------------------------------------------------------------- labellers

def stack_X(positions):
    X = {k: np.stack([p["x"][k] for p in positions]) for k in X_FIELDS}
    X["cash"] = [p["cash"] for p in positions]
    X["slots"] = [p["slots"] for p in positions]
    X["left"] = [p["left"] for p in positions]
    return X


def label_np_batch(mods, pol, positions, teacher_style=2, batch=256):
    """cnet2_search.encode_b/decode_b labelling; returns (logits, labels, u, value)."""
    S = mods["search"]
    n = len(positions)
    logits = np.zeros((n, 12, 5), np.float32)
    labels = np.zeros((n, 12), np.int64)
    us = np.zeros(n, np.float32)
    vals = np.zeros(n, np.float32)
    w = pol.w
    for lo in range(0, n, batch):
        chunk = positions[lo:lo + batch]
        X = stack_X(chunk)
        tok, ctx = S.encode_b(pol, X, np.full(len(chunk), teacher_style))
        lg = np.zeros((len(chunk), 12, 5), np.float32)
        lab = S.decode_b(pol, X, [p["aux"] for p in chunk], tok, ctx, lg_out=lg)
        logits[lo:lo + len(chunk)] = lg
        labels[lo:lo + len(chunk)] = lab
        z = ctx @ w["utility.weightT"][:, 0] + w["utility.bias"][0]
        us[lo:lo + len(chunk)] = 1.0 / (1.0 + np.exp(-z))
        zv = ctx @ w["value.weightT"][:, 0] + w["value.bias"][0]
        vals[lo:lo + len(chunk)] = np.tanh(zv)
    return logits, labels, us, vals


def label_torch_gpu(mods, positions, weights_bytes, teacher_style=2, batch=256, device="cuda"):
    """Torch CNet2 labelling: batched encode + batched autoregressive decode
    mirroring cnet2_np.Policy.act's own choices (no guard, no sampling)."""
    import torch
    import cnet2_feat as F
    from cnet2_model import CNet2, migrate_state_dict
    w = dict(np.load(io.BytesIO(weights_bytes)))
    meta = w.get("meta")
    cfg = (json.loads(str(meta)) if meta is not None else {}).get("cfg")
    model = CNet2(**cfg).to(device).eval()
    sd = {k: torch.as_tensor(v) for k, v in w.items() if k != "meta"}
    migrate_state_dict(model, sd, source="teacher weights")
    n = len(positions)
    logits = np.zeros((n, 12, 5), np.float32)
    labels = np.zeros((n, 12), np.int64)
    us = np.zeros(n, np.float32)
    vals = np.zeros(n, np.float32)
    NEGT = -1e9
    with torch.no_grad():
        for lo in range(0, n, batch):
            chunk = positions[lo:lo + batch]
            B = len(chunk)
            b = {"maps": torch.as_tensor(np.stack([p["x"]["maps"] for p in chunk]).astype(np.float32) / 255.0, device=device),
                 "tok_kind": torch.as_tensor(np.stack([p["x"]["tok_kind"] for p in chunk]), device=device),
                 "tok_cargo": torch.as_tensor(np.stack([p["x"]["tok_cargo"] for p in chunk]).astype(np.float32), device=device),
                 "tok_pos": torch.as_tensor(np.stack([p["x"]["tok_pos"] for p in chunk]), device=device),
                 "tok_mask": torch.as_tensor(np.stack([p["x"]["tok_mask"] for p in chunk]), device=device),
                 "hist_cell": torch.as_tensor(np.stack([p["x"]["hist_cell"] for p in chunk]), device=device),
                 "hist_cargo": torch.as_tensor(np.stack([p["x"]["hist_cargo"] for p in chunk]).astype(np.float32), device=device),
                 "hist_move": torch.as_tensor(np.stack([p["x"]["hist_move"] for p in chunk]), device=device),
                 "scal": torch.as_tensor(np.stack([p["x"]["scal"] for p in chunk]).astype(np.float32), device=device),
                 "teacher": torch.full((B,), teacher_style, dtype=torch.long, device=device)}
            tok, ctx, _aux = model.encode(b)
            u_t, _wdl, _mk = model.heads(ctx)
            us[lo:lo + B] = u_t.cpu().numpy()
            vals[lo:lo + B] = torch.tanh(model.value(ctx)).squeeze(1).cpu().numpy()
            prefs = [F.Prefix(p["x"], p["aux"]) for p in chunk]
            h = model.h0(ctx)
            prev_np = np.full(B, 8, np.int64)
            lab = np.full((B, 12), -1, np.int64)
            mask_all = np.stack([p["x"]["tok_mask"] for p in chunk])
            for s in range(12):
                uf = np.zeros((B, F.UF), np.float32)
                mf = np.zeros((B, 5, F.MF), np.float32)
                legal = np.zeros((B, 5), bool)
                for i, pf in enumerate(prefs):
                    if mask_all[i, s]:
                        u_, m_, l_ = pf.feats(s)
                        uf[i], mf[i], legal[i] = u_, m_, l_
                prev = torch.as_tensor(prev_np, device=device)
                inp = model.din(torch.cat([tok[:, s], ctx,
                                           torch.as_tensor(uf, device=device),
                                           torch.as_tensor(mf, device=device).reshape(B, -1)], -1)) \
                    + model.aemb(prev)
                hn = model.gru(inp, h)
                m_t = torch.as_tensor(mask_all[:, s][:, None], device=device)
                h = torch.where(m_t, hn, h)  # masked slots do not advance (Policy.act skips them)
                if s < 10:
                    base = model.whead(h)
                else:
                    base = torch.cat([model.bhead(h), torch.zeros(B, 2, device=device)], -1)
                lg = base + (model.V(torch.as_tensor(mf, device=device)) * model.U(h)[:, None, :]).sum(-1)
                lg = torch.where(torch.as_tensor(legal, device=device), lg,
                                 torch.full_like(lg, NEGT))
                lg_np = lg.cpu().numpy().astype(np.float32)
                lg_np[~mask_all[:, s]] = NEGT  # Policy.act leaves masked slots at NEG
                a = lg_np.argmax(-1)
                for i, pf in enumerate(prefs):
                    if mask_all[i, s]:
                        lab[i, s] = int(a[i])
                        pf.advance(s, int(a[i]))
                logits[lo:lo + B, s] = lg_np
                prev_np = np.where(mask_all[:, s], a + (0 if s < 10 else 5), prev_np)
            labels[lo:lo + B] = lab
    return logits, labels, us, vals


# ---------------------------------------------------------------- shards

def write_shard(path, game_id, route, positions, labels_out, teacher_style, code_md5s):
    logits, labels, us, vals = labels_out
    d = {}
    for k in X_FIELDS:
        d["x_" + k] = np.stack([p["x"][k] for p in positions])
    d["x_cash"] = np.array([p["cash"] for p in positions], np.float64)
    d["x_slots"] = np.array([p["slots"] for p in positions], np.float64)
    d["x_left"] = np.array([p["left"] for p in positions], np.float64)
    for k in AUX_FIELDS:
        d["aux_" + k] = np.stack([p["aux"][k] for p in positions])
    d["t_logits"] = logits.astype(np.float32)
    d["t_labels"] = labels.astype(np.int64)
    d["t_u"] = us.astype(np.float32)
    d["t_value"] = vals.astype(np.float32)
    n = len(positions)
    d["meta_game"] = np.array([game_id] * n)
    d["meta_step"] = np.array([p["step"] for p in positions], np.int32)
    d["meta_seat"] = np.array([p["seat"] for p in positions], np.int8)
    d["meta_source"] = np.array([p.get("source", route) for p in positions])
    d["meta_teacher_style"] = np.full(n, teacher_style, np.int8)
    d["shard_meta"] = np.str_(json.dumps(
        {"game": game_id, "route": route, "n_positions": n,
         "teacher_code_md5": TEACHER_CODE_MD5, "teacher_weights_md5": TEACHER_WEIGHTS_MD5,
         "code_md5s": code_md5s}))
    np.savez(path, **d)
    return md5_file(path)


def verify_shard(path, expect_n=None):
    d = np.load(path, allow_pickle=False)
    n = int(d["t_logits"].shape[0])
    assert d["t_logits"].shape == (n, 12, 5), d["t_logits"].shape
    assert d["t_labels"].shape == (n, 12)
    assert d["x_maps"].shape == (n, 33, 225)
    for k in AUX_FIELDS:
        assert d["aux_" + k].shape == (n, 225), k
    if expect_n is not None:
        assert n == expect_n, (n, expect_n)
    return n


# ---------------------------------------------------------------- assembly

def assemble(shard_dirs, out_dir):
    """L20-side merge of per-seat shard outputs into one training-ready dir.
    Verifies cross-shard identity, the shard plan, pairwise game
    disjointness and every shard file's md5; copies shards in and writes a
    merged manifest.json (distill_train.py consumes it directly)."""
    import shutil
    if not shard_dirs:
        die("--assemble needs at least one --shard-dir")
    mans = []
    for d in shard_dirs:
        mp = os.path.join(d, "manifest.json")
        if not os.path.isfile(mp):
            die(f"no manifest.json in shard dir {d}")
        mans.append((d, json.load(open(mp)), md5_file(mp)))
    base_d, base, _ = mans[0]
    if base.get("kind") != "small_teacher_distill_data" or "shard" not in base:
        die(f"{base_d}: not a sharded small-teacher data manifest")
    ref = {"datagen_code_md5": base["datagen_code_md5"], "teacher": base["teacher"],
           "code_md5s": base["code_md5s"],
           "slice_md5_actual": base["exclusion"]["slice_md5_actual"],
           "candidates": base["exclusion"]["candidates"],
           "shard_count": base["shard"]["count"],
           "kept_list_sha256": base["shard"]["kept_list_sha256"],
           "n_kept_total": base["shard"]["n_kept_total"]}
    seen_idx, seen_games, merged_games, total_pos = set(), {}, [], 0
    for d, m, _mm in mans:
        if m.get("kind") != "small_teacher_distill_data" or "shard" not in m:
            die(f"{d}: not a sharded small-teacher data manifest")
        checks = {"datagen_code_md5": m["datagen_code_md5"], "teacher": m["teacher"],
                  "code_md5s": m["code_md5s"],
                  "slice_md5_actual": m["exclusion"]["slice_md5_actual"],
                  "candidates": m["exclusion"]["candidates"],
                  "shard_count": m["shard"]["count"],
                  "kept_list_sha256": m["shard"]["kept_list_sha256"],
                  "n_kept_total": m["shard"]["n_kept_total"]}
        for k, v in ref.items():
            if checks[k] != v:
                die(f"{d}: field {k} differs from {base_d}; refusing to merge")
        if m["exclusion"]["candidates"]["overlap_after_filter"] != 0:
            die(f"{d}: exclusion overlap_after_filter != 0")
        if m["shard"]["index"] in seen_idx:
            die(f"duplicate shard index {m['shard']['index']}")
        seen_idx.add(m["shard"]["index"])
        for e in m["games"]:
            gid = e["game"]
            if gid in seen_games:
                die(f"game {gid} present in two shards ({seen_games[gid]} and {d})")
            seen_games[gid] = d
            sp = os.path.join(d, e["shard"])
            if not os.path.isfile(sp) or md5_file(sp) != e["shard_md5"]:
                die(f"{d}/{e['shard']}: missing or md5 mismatch")
            merged_games.append(dict(e, source_shard_dir=d,
                                     source_shard_index=m["shard"]["index"]))
            total_pos += e["n_positions"]
    if seen_idx != set(range(ref["shard_count"])):
        die(f"shard indices {sorted(seen_idx)} do not cover 0..{ref['shard_count'] - 1}")
    os.makedirs(out_dir, exist_ok=True)
    for e in merged_games:
        dst = os.path.join(out_dir, e["shard"])
        if not (os.path.isfile(dst) and md5_file(dst) == e["shard_md5"]):
            tmp = dst + ".tmp"
            shutil.copyfile(os.path.join(e["source_shard_dir"], e["shard"]), tmp)
            os.replace(tmp, dst)
            if md5_file(dst) != e["shard_md5"]:
                die(f"copy of {e['shard']} failed md5 verification")
    merged = {"kind": "small_teacher_distill_data", "assembled": True,
              "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "datagen_code_md5": ref["datagen_code_md5"], "teacher": ref["teacher"],
              "code_md5s": ref["code_md5s"], "exclusion": base["exclusion"],
              "shard": base["shard"],
              "assembled_from": [{"dir": d, "manifest_md5": mm,
                                  "shard_index": m["shard"]["index"]} for d, m, mm in mans],
              "games": merged_games,
              "totals": {"n_games": len(merged_games), "n_positions": total_pos},
              "complete": len(merged_games) == ref["n_kept_total"]}
    tmp = os.path.join(out_dir, "manifest.json.tmp")
    with open(tmp, "w") as fh:
        json.dump(merged, fh, indent=1)
    os.replace(tmp, os.path.join(out_dir, "manifest.json"))
    print(f"assembled: {len(merged_games)}/{ref['n_kept_total']} games, {total_pos} "
          f"positions, complete={merged['complete']} -> {out_dir}")
    return 0


# ---------------------------------------------------------------- main

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="small-teacher distillation data generation")
    ap.add_argument("--corpus-root", default="", help="dir holding the train-pool replay .json.gz files [required for generation]")
    ap.add_argument("--train-split", default="", help="train split file (ids / {'games': [...]}) [required for generation]")
    ap.add_argument("--slice-file", default="", help="Tier-2 slice game list (exclusion, sign-off #7) [required for generation]")
    ap.add_argument("--slice-md5", default="", help="expected md5 of --slice-file (verified before load) [required for generation]")
    ap.add_argument("--out-dir", required=True, help="shard + manifest output dir (merged dir in --assemble mode)")
    ap.add_argument("--code-dir", default="", help="pinned code stack dir (cnet2_np/search/feat/fastsim/stage0_teacher/stage1_teacher_gate/eval_v53) [required for generation]")
    ap.add_argument("--teacher-weights", default="", help="current teacher weights npz (06a03d9c) [required for generation]")
    ap.add_argument("--registered-set", action="append", default=[], help="extra consumed-set id file (repeatable)")
    ap.add_argument("--appendix-ids", default="994400,994401", help="comma ids always excluded (section-3 set)")
    ap.add_argument("--route", choices=("replay", "harvest", "both"), default="both")
    ap.add_argument("--labeler", choices=("np-batch", "torch-gpu"), default="np-batch",
                    help="np-batch (default, CPU multi-core); torch-gpu is a NON-DEFAULT option (seat principle 08:54/08:55)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--label-batch", type=int, default=256)
    ap.add_argument("--teacher-style", type=int, default=2)
    ap.add_argument("--max-games", type=int, default=0, help="0 = all kept games")
    ap.add_argument("--only-games", default="", help="comma ids to run (must survive exclusion)")
    ap.add_argument("--harvest-steps", default="200,250", help="root steps for route B")
    ap.add_argument("--harvest-max-games", type=int, default=50)
    ap.add_argument("--seed-teacher", type=int, default=7000)
    ap.add_argument("--shard-index", type=int, default=0, help="this seat's shard index, 0-based")
    ap.add_argument("--shard-count", type=int, default=1,
                    help="number of shards the dispatch split the fleet into (3 or 4 in practice; pure parameter)")
    ap.add_argument("--workers", type=int, default=1,
                    help="per-game worker processes on this seat (np-batch labelling is CPU; spot seats have 64 vCPU)")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--check-exclusions-only", action="store_true",
                    help="verify exclusions + candidate filtering, print the record, exit")
    ap.add_argument("--assemble", action="store_true",
                    help="assembly mode (L20 side): merge --shard-dir outputs into --out-dir")
    ap.add_argument("--shard-dir", action="append", default=[],
                    help="a completed shard output dir (repeatable; --assemble mode)")
    return ap.parse_args(argv)


_G = {}  # per-process generation context (fork-shared when --workers > 1)


def _run_game(gid):
    """Collect + label + atomically write one game's shard; returns the
    manifest entry. Uses the _G context (built once per process)."""
    a, hints = _G["args"], _G["hints"]
    path = resolve_game_path(gid, hints.get(gid), a.corpus_root)
    if path is None:
        die(f"replay file for game {gid} not found under {a.corpus_root}")
    steps = load_game_steps(path)
    positions, routes_done = [], []
    if a.route in ("replay", "both"):
        positions += collect_replay_positions(_G["F"], _G["FastCarbon"], _G["CFG"], steps)
        routes_done.append("replay")
    if a.route in ("harvest", "both") and gid in _G["harvest_set"]:
        want = [(s, me) for s in _G["harvest_steps"] for me in (0, 1)
                if s < len(steps) - 1]
        hp, nroots = collect_harvest_positions(_G["mods"], path, gid, want,
                                               seed_teacher=a.seed_teacher)
        positions += hp
        routes_done.append(f"harvest({nroots} roots)")
    if not positions:
        die(f"game {gid} produced zero positions")
    if a.labeler == "np-batch":
        lab = label_np_batch({"search": _G["S"]}, _G["pol"], positions,
                             teacher_style=a.teacher_style, batch=a.label_batch)
    else:
        lab = label_torch_gpu(_G["mods"], positions, _G["weights_bytes"],
                              teacher_style=a.teacher_style,
                              batch=a.label_batch, device=a.device)
    shard_name = f"shard_{normalize_id(gid)}.npz"
    shard_path = os.path.join(a.out_dir, shard_name)
    route_tag = "+".join(routes_done)
    smd5 = write_shard(shard_path + ".tmp.npz", gid, route_tag, positions, lab,
                       a.teacher_style, _G["code_md5s"])
    os.replace(shard_path + ".tmp.npz", shard_path)
    if md5_file(shard_path) != smd5:
        die(f"shard {shard_name} md5 changed across rename")
    n = verify_shard(shard_path, expect_n=len(positions))
    return {"game": gid, "path": path, "route": route_tag, "shard": shard_name,
            "shard_md5": smd5, "n_positions": n}


def main(argv=None):
    a = parse_args(argv)
    if a.assemble:
        return assemble(a.shard_dir, a.out_dir)
    for req in ("corpus_root", "train_split", "slice_file", "slice_md5",
                "code_dir", "teacher_weights"):
        if not getattr(a, req):
            die(f"--{req.replace('_', '-')} is required for generation (fail-closed)")
    if not (0 <= a.shard_index < a.shard_count):
        die(f"bad shard plan: index {a.shard_index} of count {a.shard_count}")
    if a.workers > 1 and a.labeler == "torch-gpu":
        die("--workers > 1 is for the np-batch labeler; torch-gpu runs single-process")

    appendix_ids = [s.strip() for s in a.appendix_ids.split(",") if s.strip()]
    excluded, excl_record = load_exclusions(a.slice_file, a.slice_md5, appendix_ids,
                                            a.registered_set)
    ids, hints = load_train_split(a.train_split)
    only = [s.strip() for s in a.only_games.split(",") if s.strip()] or None
    kept, cand_record = check_candidates(ids, excluded, only_games=only)
    if only:
        kept = [g for g in kept if g in only]
    if a.max_games:
        kept = kept[: a.max_games]
    excl_record["candidates"] = cand_record
    excl_record["train_split"] = a.train_split
    excl_record["train_split_md5"] = md5_file(a.train_split)
    if a.check_exclusions_only:
        print(json.dumps(excl_record, indent=1))
        return 0

    # Deterministic fleet sharding (seat principle 08:54/08:55): sorted kept
    # list, strided partition; harvest games are chosen on the FULL kept list
    # so the fleet-wide harvest total stays --harvest-max-games.
    kept_sorted = sorted(kept)
    kept_sha = hashlib.sha256(json.dumps(kept_sorted).encode()).hexdigest()
    harvest_set = set(kept_sorted[: a.harvest_max_games]) \
        if a.route in ("harvest", "both") else set()
    my_games = kept_sorted[a.shard_index:: a.shard_count]
    shard_record = {"index": a.shard_index, "count": a.shard_count,
                    "n_kept_total": len(kept_sorted), "kept_list_sha256": kept_sha,
                    "n_games_this_shard": len(my_games),
                    "harvest_games_fleet_total": len(harvest_set)}

    sys.path.insert(0, a.code_dir)
    import cnet2_feat as F
    from fastsim import FastCarbon
    from eval_v53 import CFG
    import stage1_teacher_gate as G
    import stage0_teacher as T
    import cnet2_search as S
    from cnet2_search import CachePolicy, load_weights

    code_md5s = {}
    for m in ("cnet2_np", "cnet2_search", "cnet2_feat", "fastsim", "stage0_teacher",
              "stage1_teacher_gate", "eval_v53"):
        code_md5s[m] = md5_file(os.path.join(a.code_dir, m + ".py"))
    if code_md5s["stage0_teacher"] != TEACHER_CODE_MD5:
        die(f"teacher code md5 {code_md5s['stage0_teacher']} != pinned {TEACHER_CODE_MD5}")
    with open(a.teacher_weights, "rb") as fh:
        weights_bytes = fh.read()
    if md5_bytes(weights_bytes) != TEACHER_WEIGHTS_MD5:
        die(f"teacher weights md5 {md5_bytes(weights_bytes)} != pinned {TEACHER_WEIGHTS_MD5}")
    weights = dict(np.load(io.BytesIO(weights_bytes)))
    pol = CachePolicy(weights)
    mods = {"gate": G, "teacher": T, "CachePolicy": CachePolicy, "weights": weights}
    _G.update({"args": a, "hints": hints, "F": F, "FastCarbon": FastCarbon, "CFG": CFG,
               "S": S, "pol": pol, "mods": mods, "weights_bytes": weights_bytes,
               "code_md5s": code_md5s, "harvest_set": harvest_set,
               "harvest_steps": [int(s) for s in a.harvest_steps.split(",") if s.strip()]})

    os.makedirs(a.out_dir, exist_ok=True)
    manifest_path = os.path.join(a.out_dir, "manifest.json")
    manifest = {"kind": "small_teacher_distill_data", "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "datagen_code_md5": md5_file(os.path.abspath(__file__)),
                "teacher": {"code_md5": TEACHER_CODE_MD5, "weights_md5": TEACHER_WEIGHTS_MD5},
                "code_md5s": code_md5s, "exclusion": excl_record, "shard": shard_record,
                "config": {"route": a.route, "labeler": a.labeler,
                           "teacher_style": a.teacher_style, "label_batch": a.label_batch,
                           "harvest_steps": a.harvest_steps,
                           "harvest_max_games": a.harvest_max_games,
                           "workers": a.workers},
                "games": []}
    done = {}
    if not a.no_resume and os.path.isfile(manifest_path):
        old = json.load(open(manifest_path))
        if old.get("datagen_code_md5") != manifest["datagen_code_md5"]:
            die("existing manifest was written by a different datagen build; refusing to mix")
        osh = old.get("shard", {})
        if (osh.get("kept_list_sha256"), osh.get("index"), osh.get("count")) != \
                (kept_sha, a.shard_index, a.shard_count):
            die("existing manifest belongs to a different shard plan; refusing to mix")
        for g in old.get("games", []):
            done[g["game"]] = g
        manifest["games"] = list(done.values())

    def save_manifest():
        tmp = manifest_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(manifest, fh, indent=1)
        os.replace(tmp, manifest_path)

    todo = []
    for gid in my_games:
        if gid in done:
            shard = os.path.join(a.out_dir, done[gid]["shard"])
            if os.path.isfile(shard) and md5_file(shard) == done[gid]["shard_md5"]:
                print(f"[skip-resume] {gid}")
                continue
        todo.append(gid)

    def record(entry):
        manifest["games"] = [g for g in manifest["games"]
                             if g["game"] != entry["game"]] + [entry]
        save_manifest()
        print(f"[done] {entry['game']}: {entry['n_positions']} positions "
              f"({entry['route']}) shard md5 {entry['shard_md5']}", flush=True)

    if a.workers > 1 and len(todo) > 1:
        import multiprocessing as mp
        with mp.Pool(a.workers) as pool:  # fork: children inherit _G read-only
            for entry in pool.imap_unordered(_run_game, todo):
                record(entry)
    else:
        for gid in todo:
            record(_run_game(gid))
    manifest["totals"] = {"n_games": len(manifest["games"]),
                          "n_positions": sum(g["n_positions"] for g in manifest["games"])}
    save_manifest()
    print(json.dumps(manifest["totals"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
