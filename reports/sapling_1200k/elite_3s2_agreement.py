"""Compare two FP32 Sapling policies on held-out, teacher-forced Master3s2 decisions."""
import argparse
import gzip
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from build_sapling_elite_bc import REVISION, STYLE_SLOT, digest, git, key_of, validate
from recover_public import repository_index
import cnet2_feat as F
from cnet2_model import CNet2
from init_sapling import resolve_preset_metadata

EXPERT = "claude-AlphaCarbon-Master3s2"
HERE = Path(__file__).resolve().parent
WEIGHTS = ROOT / "data/weights/sapling_1200k"
SELECTION = "sapling-master3s2-direct-holdout-v1"
TICKS = np.linspace(0, 298, 20, dtype=int).tolist()
ROLES = ("collector", "planter", "base")


def holdout(manifest):
    revision, paths = repository_index(ROOT, REVISION)
    cards = json.loads(git("show", revision + ":data/master3s2_matches.json"))
    archived = [r for r in cards if r.get("state") == "done" and key_of(r) in paths]
    ordered = sorted(archived, key=lambda r: digest((SELECTION + "|" + key_of(r)).encode()))
    used_ids = {x["id"].rsplit("-", 1)[0] for x in manifest["games"]}
    used_seeds = {x["seed"] for x in manifest["games"]}
    for seat in (0, 1):
        count = 0
        for r in ordered:
            if r["id"] in used_ids or r["name_" + r["order"][seat]] != EXPERT:
                continue
            source = sorted(paths[key_of(r)], key=lambda p: p["path"])[0]
            raw = git("cat-file", "blob", source["git_blob"])
            env = json.loads(gzip.decompress(raw))
            check = validate(env, r)
            seed = check["random_seed"]
            if not isinstance(seed, int) or seed in used_seeds or 20291101 <= seed <= 20292100:
                continue
            if check["statuses"] != ["DONE", "DONE"] or check["last_step"] != 299:
                continue
            if check["seat_model_names"][seat] != EXPERT:
                raise ValueError("teacher seat mismatch")
            used_ids.add(r["id"])
            used_seeds.add(seed)
            yield dict(id=key_of(r), seat=seat, random_seed=seed,
                       sha256=digest(raw), **source), env
            count += 1
            if count == 2:
                break
        if count != 2:
            raise ValueError("insufficient distinct held-out matches")


def sample_rows(steps, seat):
    feat = F.Feat(seat)
    samples, cut_roots, cut_slots = [], 0, 0
    for t, pair in enumerate(steps[:-1]):
        obs = pair[0]["observation"]
        x = feat.observe(dict(step=t, players=obs["players"], player=seat,
                              carbon=pair[seat]["observation"]["carbon"]))
        if t not in TICKS:
            continue
        logged = steps[t + 1][seat].get("action") or {}
        labels = feat.labels(logged)
        if any(cmd not in F.MOVES + F.BASE_ACTS for cmd in logged.values() if cmd):
            raise ValueError("unrecognized raw action")
        pf = F.Prefix(x, feat.aux)
        uf = np.zeros((F.OWN, F.UF), np.float16)
        mf = np.zeros((F.OWN, 5, F.MF), np.float16)
        legal = np.zeros((F.OWN, 5), bool)
        for s in np.flatnonzero(x["tok_mask"][:F.OWN]):
            u, m, choices = pf.feats(s)
            if not choices[labels[s]]:
                # Every later decoder slot depends on this now-incompatible prefix.
                cut_roots += 1
                cut_slots += int((labels[s:] >= 0).sum())
                labels[s:] = -1
                break
            uf[s], mf[s], legal[s] = u, m, choices
            pf.advance(s, int(labels[s]))
        row = {k: x[k] for k in ("maps", "scal", "tok_pos", "tok_kind", "tok_mask",
                                 "tok_cargo", "hist_cell", "hist_cargo", "hist_move")}
        # cnet2_train.to_dev requires the auxiliary target; no hidden target is supplied.
        row.update(y=labels, legal=legal, uf=uf, mf=mf, teacher=STYLE_SLOT,
                   carbon=np.zeros(F.N, np.uint8))
        samples.append(row)
    return samples, dict(roots=len(samples), cut_roots=cut_roots, cut_slots=cut_slots)


@torch.inference_mode()
def score(netfile, rows, groups, trainer):
    with np.load(netfile, allow_pickle=False) as z:
        preset, cfg, _ = resolve_preset_metadata(json.loads(str(z["meta"])))
        if preset != "sapling-1.2m":
            raise ValueError("expected 1.2m Sapling")
        net = CNet2(**cfg)
        net.load_state_dict({k: torch.from_numpy(z[k]) for k in z if k != "meta"}, strict=True)
    net.eval()
    losses, predictions = [], []
    for start in range(0, len(rows), 16):
        block = rows[start:start + 16]
        batch = trainer.to_dev({k: np.stack([r[k] for r in block]) for k in block[0]}, aug=False)
        logits = net(batch)[0]
        if not torch.isfinite(logits).all():
            raise ValueError("non-finite logits")
        ce = torch.nn.functional.cross_entropy(logits.transpose(1, 2), batch["y"],
                                                ignore_index=-1, reduction="none")
        losses.extend(ce.numpy())
        predictions.extend(logits.argmax(-1).numpy())
    y = np.stack([r["y"] for r in rows])
    free = (y >= 0) & (np.stack([r["legal"] for r in rows]).sum(-1) >= 2)
    kinds = np.stack([r["tok_kind"][:F.OWN] for r in rows])
    hits = np.asarray(predictions) == y
    losses = np.asarray(losses)
    def metric(mask):
        n = int(mask.sum())
        return dict(slots=n, correct=int(hits[mask].sum()),
                    accuracy=float(hits[mask].mean()) if n else None,
                    ce=float(losses[mask].mean(dtype=np.float64)) if n else None)
    return dict(weights_sha256=digest(netfile.read_bytes()), free_choices=metric(free),
                by_role={role: metric(free & (kinds == i)) for i, role in enumerate(ROLES)},
                by_game={key: metric(free & (groups == key)[:, None]) for key in dict.fromkeys(groups)},
                forced_slots_excluded=int(((y >= 0) & ~free).sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", type=Path, default=WEIGHTS / "sqrt64.npz")
    p.add_argument("--after", type=Path, default=WEIGHTS / "elite_bc_v1.npz")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists() or any(k.startswith(("BC_", "CNET2_")) for k in os.environ):
        raise ValueError("use a fresh output and unset BC_*/CNET2_* switches")
    os.environ["BC_DEVICE"] = "cpu"
    import cnet2_train as trainer
    torch.set_num_threads(1)
    manifest = (HERE / "elite_bc_data.json").read_bytes()
    games, rows, groups = [], [], []
    for record, env in holdout(json.loads(manifest)):
        sampled, cuts = sample_rows(env["steps"], record["seat"])
        games.append({**record, **cuts})
        rows.extend(sampled)
        groups.extend([record["id"]] * len(sampled))
    groups = np.asarray(groups)
    before = score(args.before, rows, groups, trainer)
    after = score(args.after, rows, groups, trainer)
    report = dict(revision=REVISION, selection_seed=SELECTION, ticks=TICKS,
                  data_manifest_sha256=digest(manifest), evaluator_sha256=digest(Path(__file__).read_bytes()),
                  precision="FP32 CPU", style_slot=STYLE_SLOT, games=games, before=before, after=after,
                  meaning="Expert command agreement under the logged prefix on four disjoint official games. "
                          "Metrics pool slots with at least two prefix-legal commands. No closed-loop strength claim.")
    with args.out.open("x") as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    for name, result in (("before", before), ("after", after)):
        print(name, result["free_choices"], flush=True)


if __name__ == "__main__":
    main()
