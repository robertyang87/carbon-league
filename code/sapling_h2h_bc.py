"""Continue Sapling elite BC on both seats of matches between top-five players.

All valid decisions receive action imitation loss; pre-action attack/defense
opportunities receive an additional capped multiplier. No extra inputs are
required by the deployed policy. Keep the prescribed final update regardless
of validation metrics. Use --tactical-factor 1 for a same-data weight ablation.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as nnf

from cnet2_model import CNet2
from init_sapling import resolve_preset_metadata
from sapling_bootstrap import effective_unit_weights, opportunity_counts, role_action_weights, sha
from sapling_elite_bc import load_games

MASK_KEYS = ("attack", "defense", "cargo_defense", "tree_defense")
ROLES = ("collector", "planter", "base")
DEFAULT_PRIOR = Path(__file__).resolve().parents[1] / "reports/sapling_1200k/elite_bc_data.json"


def load_masks(manifest_path, games, datasets):
    masks = {group: {key: [] for key in MASK_KEYS} for group in datasets}
    for game in games:
        path = manifest_path.parent / game["mask_shard"]
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != game["mask_sha256"]:
            raise ValueError(f"mask hash mismatch: {path}")
        group = game["split"]
        with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
            for key in MASK_KEYS:
                arr = arrays[key]
                if arr.shape != (game["rows"], datasets[group]["y"].shape[1]) or arr.dtype != np.bool_:
                    raise ValueError(f"unexpected {key} mask shape or dtype: {path}")
                masks[group][key].append(arr)
    out = {}
    for group, parts in masks.items():
        merged = {key: np.concatenate(arr) for key, arr in parts.items()}
        if not np.array_equal(merged["defense"], merged["cargo_defense"] | merged["tree_defense"]):
            raise ValueError("defense subtype coverage mismatch")
        if np.any((merged["attack"] | merged["defense"]) & (datasets[group]["y"] < 0)):
            raise ValueError("tactical mask refers to an empty unit")
        out[group] = merged
    return out


def grouped_metrics(rows, ce, preds, data, masks):
    y = data["y"][rows]
    valid = y >= 0
    kind = data["tok_kind"][rows, :y.shape[1]]
    ids = np.broadcast_to(data["gid"][rows, None], y.shape)
    masks = {key: arr[rows] for key, arr in masks.items()}
    attack, defense = masks["attack"], masks["defense"]
    masks.update(all=valid, tactical=attack | defense, overlap=attack & defense,
                 other=valid & ~(attack | defense))

    def measure(selected):
        n = int(selected.sum())
        return dict(slots=n, games=int(np.unique(ids[selected]).size),
                    ce=float(ce[selected].mean(dtype=np.float64)) if n else None,
                    agreement=float((y[selected] == preds[selected]).mean()) if n else None)

    by_context = {}
    for key, selected in masks.items():
        selected = valid & selected
        metric = measure(selected)
        metric["by_role"] = {role: measure(selected & (kind == idx)) for idx, role in enumerate(ROLES)}
        by_context[key] = metric
    by_role = opportunity_counts(y, kind, data["legal"][rows], preds)
    return {"contexts": by_context, "roles": by_role}


@torch.no_grad()
def evaluate(net, data, masks, rows, trainer, batch_size):
    net.eval()
    all_ce, all_pred = [], []
    rows = np.sort(rows)
    for start in range(0, len(rows), batch_size):
        batch = trainer.to_dev(trainer.fetch(data, rows[start:start + batch_size]), aug=False)
        logits, _, _ = net(batch)
        if not torch.isfinite(logits).all():
            raise ValueError("non-finite evaluation logits")
        y = batch["y"]
        ce = nnf.cross_entropy(logits.flatten(0, 1), y.clamp_min(0).flatten(),
                               reduction="none").reshape_as(y)
        all_ce.append(ce.cpu().numpy())
        all_pred.append(logits.argmax(-1).cpu().numpy())
    return grouped_metrics(rows, np.concatenate(all_ce), np.concatenate(all_pred), data, masks)


def run(args):
    if args.out.exists() or args.receipt.exists() or args.out == args.receipt:
        raise FileExistsError("new, distinct --out and --receipt paths required")
    switches = [k for k in os.environ if k.startswith(("CNET2_", "BC_"))]
    if switches:
        raise ValueError(f"unset environment training switches: {switches}")
    os.environ["BC_DEVICE"] = args.device
    import cnet2_train as trainer

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    start = time.monotonic()
    datasets, games, manifest_hash = load_games(args.manifest, trainer)
    prior = json.loads(args.prior_manifest.read_bytes())["games"]
    for key in ("id", "seed"):
        vals = [g[key] for g in games]
        if len(set(vals)) != len(vals) or set(vals) & {g[key] for g in prior}:
            raise ValueError(f"duplicate H2H {key} or overlap with the v1 dataset")
    masks = load_masks(args.manifest, games, datasets)
    train, val = datasets["train"], datasets["val"]
    src = args.init.read_bytes()
    init_sha = hashlib.sha256(src).hexdigest()
    with np.load(io.BytesIO(src), allow_pickle=False) as archive:
        meta = json.loads(str(archive["meta"]))
        preset, cfg, params = resolve_preset_metadata(meta)
        if preset != "sapling-1.2m" or meta.get("training_status") != "elite_policy_bc":
            raise ValueError("an elite-BC Sapling 120万 checkpoint is required")
        if meta["elite_bc"]["dataset_manifest_sha256"] != sha(args.prior_manifest):
            raise ValueError("v1 checkpoint and its prior data manifest disagree")
        net = CNet2(**cfg)
        net.load_state_dict({k: torch.from_numpy(archive[k]) for k in archive.files if k != "meta"}, strict=True)
    if params != sum(p.numel() for p in net.parameters()):
        raise ValueError("Sapling 120万 parameter count changed")
    net.to(trainer.DEV)
    for name, param in net.named_parameters():
        if name.startswith(("value.", "utility.", "wdl.", "margin.", "aux.")):
            param.requires_grad_(False)

    labels = train["y"].copy()
    kind = train["tok_kind"][:, :labels.shape[1]]
    forced_none = (kind == 2) & ~train["legal"][..., 1:3].any(-1)
    labels[forced_none] = -1
    plain, _ = role_action_weights({"y": labels, "tok_kind": kind}, cap=8.)
    weights = effective_unit_weights(plain, labels >= 0, "sqrt-role-action")
    tactical = masks["train"]["attack"] | masks["train"]["defense"]
    plain_share = float(weights[tactical].sum() / weights.sum())
    weights[tactical] *= args.tactical_factor
    weighted_share = float(weights[tactical].sum() / weights.sum())
    train["uw"] = weights

    check_rows = {"train": np.linspace(0, len(train["y"]) - 1, min(256, len(train["y"])), dtype=int),
                  "val": np.arange(len(val["y"]))}

    def metrics():
        return {group: evaluate(net, data, masks[group], check_rows[group], trainer, args.bs)
                for group, data in datasets.items()}

    before = metrics()
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-4)
    losses = []
    seen_rows = np.zeros(len(train["y"]), dtype=bool)
    exposures = {key: 0 for key in MASK_KEYS}
    net.train()
    for step in range(1, args.updates + 1):
        ii = rng.choice(len(train["y"]), min(args.bs, len(train["y"])), replace=False)
        seen_rows[ii] = True
        for key in exposures:
            exposures[key] += int(masks["train"][key][ii].sum())
        b = trainer.to_dev(trainer.fetch(train, ii), aug=False)
        logits, _, _ = net(b)
        y = b["y"]
        ce = nnf.cross_entropy(logits.flatten(0, 1), y.clamp_min(0).flatten(),
                               reduction="none").reshape_as(y)
        weights_batch = b["uw"]
        if not weights_batch.sum().item() > 0:
            raise ValueError("no supervised action in a minibatch")
        loss = (ce * weights_batch).sum() / weights_batch.sum()
        if not torch.isfinite(loss):
            raise ValueError("non-finite training loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        losses.append(float(loss.detach()))
        if step % 64 == 0:
            print(f"update {step}/{args.updates} loss={loss:.4f} grad={norm:.3f}", flush=True)
    after = metrics()
    settings = {k: v for k, v in vars(args).items() if not isinstance(v, Path)}
    recipe = dict(parent_sha256=init_sha, dataset_manifest_sha256=manifest_hash,
                  prior_dataset_manifest_sha256=sha(args.prior_manifest), settings=settings, style=2,
                  runner_sha256=sha(__file__))
    report = dict(format="sapling-top5-h2h-bc/2", **recipe, params=params,
                  train_games=sum(g["split"] == "train" for g in games),
                  val_games=sum(g["split"] == "val" for g in games), train_rows=len(train["y"]),
                  val_rows=len(val["y"]), train_rows_seen=int(seen_rows.sum()),
                  train_games_seen=int(np.unique(train["gid"][seen_rows]).size),
                  tactical_slot_exposures=exposures, weighting="sqrt role/action BC, forced base NONE excluded; attack or defense x factor once",
                  tactical_fraction_unweighted_decisions=float(tactical.sum() / (labels >= 0).sum()),
                  weight_mass_tactical_before=plain_share, weight_mass_tactical_after=weighted_share,
                  train_tactical_slot_counts={k: int(v.sum()) for k, v in masks["train"].items()},
                  objective="policy CE; no terminal returns, full-carbon targets or future signals",
                  before=before, after=after, reported_train_rows=len(check_rows["train"]),
                  logged_loss=losses, torch_version=str(torch.__version__), numpy_version=np.__version__,
                  elapsed_seconds=time.monotonic() - start)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as stage:
        path = Path(stage) / "weights.npz"
        trainer.export(net, str(path), dict(cfg=net.cfg, model_name="Sapling", preset=preset,
                       params=params, untrained=False, value_heads_untrained=True, training_status="h2h_policy_bc",
                       top5_h2h_bc=recipe))
        report["trained_sha256"] = sha(path)
        receipt = json.dumps(report, indent=2, allow_nan=False) + "\n"
        os.link(path, args.out)
    with args.receipt.open("x") as dst:
        dst.write(receipt)
    print(json.dumps({k: v["ce"] for k, v in after["val"]["contexts"].items()}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("manifest", "init", "out", "receipt"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--prior-manifest", type=Path, default=DEFAULT_PRIOR)
    p.add_argument("--updates", type=int, default=256)
    p.add_argument("--bs", type=int, default=24)
    p.add_argument("--lr", type=float, default=0.00025)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--tactical-factor", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=551189)
    args = p.parse_args()
    if min(args.updates, args.bs, args.threads) < 1 or not (0 < args.lr < 1 and 1 <= args.tactical_factor <= 2):
        p.error("positive updates, bs, threads, lr<1 and tactical factor in [1,2] required")
    run(args)
