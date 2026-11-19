"""Fine-tune Sapling on elite replay actions. Pass a game-split shard manifest.

Each game has id, split (train/val), shard (path relative to the manifest)
and sha256. Shards follow cnet2_build's typed schema and use deployment style 2.
The final update is the candidate. Holdout scores never select a checkpoint.
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


def load_games(manifest_path, trainer):
    manifest_bytes = manifest_path.read_bytes()
    games = json.loads(manifest_bytes)["games"]
    ids = [g["id"] for g in games]
    hashes = [g["sha256"] for g in games]
    if len(set(ids)) != len(ids) or len(set(hashes)) != len(hashes):
        raise ValueError("duplicate game or shard in split")
    groups = {"train": [], "val": []}
    for game in games:
        path = manifest_path.parent / game["shard"]
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != game["sha256"]:
            raise ValueError(f"shard identity mismatch: {path}")
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            part = {key: archive[key] for key in trainer.FIELDS}
        if not len(part["y"]) or len(part["y"]) != game["rows"]:
            raise ValueError(f"bad row count: {path}")
        if any(len(v) != len(part["y"]) for v in part.values()) or not np.all(part["teacher"] == 2):
            raise ValueError(f"bad row alignment or style: {path}")
        y = part["y"]
        if np.any((y < -1) | (y > 4)) or not (y >= 0).any():
            raise ValueError(f"invalid action labels: {path}")
        legal = np.take_along_axis(part["legal"], np.maximum(y, 0)[..., None], -1)[..., 0]
        if not np.all(legal[y >= 0]):
            raise ValueError(f"teacher action outside prefix legality: {path}")
        groups[game["split"]].append(part)
    if not all(groups.values()):
        raise ValueError("need both training and validation games")
    datasets = {
        split: {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
        for split, parts in groups.items()
    }
    return datasets, games, hashlib.sha256(manifest_bytes).hexdigest()


@torch.no_grad()
def evaluate(net, data, rows, batch_size, trainer):
    net.eval()
    rows = np.sort(rows)
    predictions = []
    total_ce = 0.
    support = 0
    for start in range(0, len(rows), batch_size):
        idx = rows[start:start + batch_size]
        batch = trainer.to_dev(trainer.fetch(data, idx), aug=False)
        logits, _, _ = net(batch)
        y = batch["y"]
        valid = y >= 0
        if not torch.isfinite(logits).all():
            raise ValueError("non-finite evaluation logits")
        total_ce += nnf.cross_entropy(logits[valid], y[valid], reduction="sum").item()
        support += valid.sum().item()
        predictions.append(logits.argmax(-1).cpu().numpy())
    labels = data["y"][rows]
    predictions = np.concatenate(predictions)
    metrics = opportunity_counts(labels, data["tok_kind"][rows, :labels.shape[1]],
                                 data["legal"][rows], predictions)
    metrics.update(unweighted_policy_ce=total_ce / support, valid_slots=support, rows=len(rows))
    return metrics


def run(args):
    if args.out.exists() or args.receipt.exists() or args.out == args.receipt:
        raise FileExistsError("new, distinct --out and --receipt paths required")
    switches = [k for k in os.environ if k.startswith(("CNET2_", "BC_"))]
    if switches:
        raise ValueError(f"use command arguments and unset environment switches: {switches}")
    os.environ["BC_DEVICE"] = args.device
    import cnet2_train as trainer

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    start = time.monotonic()
    datasets, games, manifest_hash = load_games(args.manifest, trainer)
    train, val = datasets["train"], datasets["val"]
    init_bytes = args.init.read_bytes()
    init_sha = hashlib.sha256(init_bytes).hexdigest()
    with np.load(io.BytesIO(init_bytes), allow_pickle=False) as archive:
        init_meta = json.loads(str(archive["meta"]))
        preset, cfg, expected = resolve_preset_metadata(init_meta)
        if preset != "sapling-1.2m":
            raise ValueError("expected 120万 Sapling architecture")
        net = CNet2(**cfg)
        net.load_state_dict({k: torch.from_numpy(archive[k]) for k in archive.files if k != "meta"},
                            strict=True)
    if sum(p.numel() for p in net.parameters()) != expected:
        raise ValueError("model parameter count mismatch")
    net.to(trainer.DEV)
    for name, param in net.named_parameters():
        if name.startswith(("value.", "utility.", "wdl.", "margin.", "aux.")):
            param.requires_grad_(False)

    # A forced base command has zero policy CE. Counting it as a target when
    # estimating class frequency would still distort all recruitment weights.
    decision_y = train["y"].copy()
    kind = train["tok_kind"][:, :decision_y.shape[1]]
    forced = (kind == 2) & ~train["legal"][..., 1:3].any(-1) & (decision_y >= 0)
    decision_y[forced] = -1
    weights, weight_metadata = role_action_weights({"y": decision_y, "tok_kind": kind}, cap=8.)
    train["uw"] = effective_unit_weights(weights, decision_y >= 0, "sqrt-role-action")
    train_metric_rows = np.linspace(0, len(train["y"]) - 1, min(256, len(train["y"])), dtype=int)
    metric_rows = {"train": train_metric_rows, "val": np.arange(len(val["y"]))}

    def metrics():
        return {key: evaluate(net, datasets[key], idx, args.bs, trainer)
                for key, idx in metric_rows.items()}

    before = metrics()
    net.train()
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-4)
    updates = []
    for step in range(1, args.updates + 1):
        idx = rng.choice(len(train["y"]), min(args.bs, len(train["y"])), replace=False)
        batch = trainer.to_dev(trainer.fetch(train, idx), aug=False)
        logits, _, _ = net(batch)
        y = batch["y"]
        per_slot_ce = nnf.cross_entropy(logits.flatten(0, 1), y.clamp_min(0).flatten(),
                                        reduction="none").reshape_as(y)
        w = batch["uw"]
        if not w.sum().item() > 0:
            raise ValueError("empty supervised minibatch")
        loss = (w * per_slot_ce).sum() / w.sum()
        if not torch.isfinite(loss):
            raise ValueError("non-finite training loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        updates.append(float(loss.detach()))
        if step % 64 == 0:
            print(f"update {step}/{args.updates} loss={updates[-1]:.4f} grad={grad_norm:.3f}", flush=True)
    after = metrics()
    trainer._check_finite_arrays({k: v.detach().cpu().numpy() for k, v in net.state_dict().items()},
                                 "trained model")
    settings = {k: v for k, v in vars(args).items() if not isinstance(v, Path)}
    recipe = dict(parent_sha256=init_sha, dataset_manifest_sha256=manifest_hash,
                  style=2, settings=settings, trainer_sha256=sha(__file__))
    report = dict(format="sapling-elite-bc/1", **recipe, params=expected,
                  dataset_games=[{k: g[k] for k in ("id", "model", "seat", "split", "rows", "sha256")}
                                 for g in games],
                  before=before, after=after, train_rows=len(train["y"]), val_rows=len(val["y"]),
                  unweighted_validation=True, train_metric_sampling="evenly spaced 256 rows or fewer",
                  optimizer="AdamW", weight_decay=1e-4, weighting="sqrt role/action normalized on actionable slots",
                  forced_train_base_none_excluded=int(forced.sum()), weights_before_sqrt=weight_metadata,
                  objective="teacher-forced action cross-entropy only", augmentations=False,
                  terminal_or_auxiliary_targets_used=False, losses=updates, torch_version=str(torch.__version__),
                  numpy_version=np.__version__, elapsed_seconds=time.monotonic() - start)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as stage:
        weights_path = Path(stage) / "weights.npz"
        trainer.export(net, str(weights_path), dict(cfg=net.cfg, model_name="Sapling", preset=preset,
                       params=expected, untrained=False, value_heads_untrained=True, training_status="elite_policy_bc",
                       elite_bc=recipe))
        report["trained_sha256"] = sha(weights_path)
        receipt = json.dumps(report, indent=2, allow_nan=False) + "\n"
        os.link(weights_path, args.out)
    with args.receipt.open("x") as out:
        out.write(receipt)
    print(f"validation CE {before['val']['unweighted_policy_ce']:.4f} -> "
          f"{after['val']['unweighted_policy_ce']:.4f}, weights {report['trained_sha256']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "init", "out", "receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--updates", type=int, default=320)
    parser.add_argument("--bs", type=int, default=24)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=551188)
    args = parser.parse_args()
    if min(args.updates, args.bs, args.threads) < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("updates, batch size, threads, and finite learning rate must be positive")
    run(args)
