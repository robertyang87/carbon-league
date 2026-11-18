"""Bounded policy-only Sapling warm-up on newly generated OWN-policy demonstrations.

This is a diagnostic, not the formal distillation/selection experiment. It never
reads ladder corpora or exam replays. It reuses cnet2_build's replay alignment and
cnet2_train's weighted BC loss. Value/auxiliary targets are NOT trained: an opening
prefix is not a terminal result. Validation is split by seed (both seats together).

Example (CPU, fixed budget; creates a NEW output directory):
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python code/sapling_bootstrap.py \
    --init /path/sapling_seed551187.npz --out /tmp/sapling-bootstrap \
    --balance role-action --updates 64

Output trained.npz is compatible with cnet2_pack and BC_INIT, but has no strength
certification. All weights/metrics use the declared style slot (default 2).
Both registered init_sapling presets are accepted and strictly identified from
the weight metadata and tensor shapes; no architecture override or fallback.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROLES = ("collector", "planter", "base")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def role_action_weights(data, cap=8.0):
    """Train-only balancing via the existing per-unit `uw` loss contract.

    Equal mass per supported role, then per observed action within that role,
    capped before renormalization. This deliberately changes the BC objective;
    it is not an inverse-probability estimate of the natural action distribution.
    """
    y = data["y"]
    kind = data["tok_kind"][:, :y.shape[1]]
    valid = y >= 0
    if not valid.any():
        raise ValueError("no valid finite BC supervision")
    counts = np.zeros((3, 5), np.int64)
    for k in range(3):
        for a in range(5):
            counts[k, a] = np.count_nonzero(valid & (kind == k) & (y == a))
    supported = counts > 0
    nroles = np.count_nonzero(supported.any(1))
    table = np.zeros((3, 5), np.float32)
    for k in range(3):
        na = np.count_nonzero(supported[k])
        if na:
            table[k, supported[k]] = np.minimum(
                valid.sum() / (nroles * na * counts[k, supported[k]]), cap)
    weights = np.zeros_like(y, dtype=np.float32)
    weights[valid] = table[kind[valid], y[valid]]
    scale = float(weights[valid].mean()) if valid.any() else 1.0
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("no valid finite BC supervision")
    weights /= scale
    return weights, {"counts": counts.tolist(), "multipliers": (table / scale).tolist(),
                     "cap_before_normalization": cap, "normalization": scale}


def effective_unit_weights(weights, valid, mode):
    """The sqrt ablation tempers the SAME train-only weights, without new fitting."""
    if mode == "none":
        return np.where(valid, 1.0, 0.0).astype(np.float32)
    if mode == "role-action":
        return weights.copy()
    if mode == "sqrt-role-action":
        out = np.sqrt(weights)
        out /= out[valid].mean()
        return out
    raise ValueError(f"unknown balance mode: {mode}")


def opportunity_counts(y, kind, legal, pred):
    """Keep forced base NONE distinct from a decision with recruitment available.

    `legal` is CNet2 Prefix's approximate legality, not an environment certificate
    (it uses fixed recruitment price 30). Full runtime remains a separate check.
    """
    out = {}
    valid = y >= 0
    for k, name in enumerate(ROLES):
        mask = valid & (kind == k)
        cm = np.zeros((5, 5), np.int64)
        np.add.at(cm, (y[mask], pred[mask]), 1)
        counts = cm.sum(1)
        support = counts > 0
        out[name] = {"n": int(mask.sum()), "label_counts": counts.tolist(),
                     "prediction_counts": cm.sum(0).tolist(), "confusion_label_by_pred": cm.tolist(),
                     "accuracy": float(np.trace(cm) / mask.sum()) if mask.any() else None,
                     "macro_recall_supported_actions": float(np.mean(cm.diagonal()[support] / counts[support])) if support.any() else None}
    recruit = valid & (kind == 2) & legal[..., 1:3].any(-1)
    forced = valid & (kind == 2) & ~legal[..., 1:3].any(-1)
    out["base_recruit_available"] = {
        "n": int(recruit.sum()),
        "teacher_recruits": int(np.count_nonzero(recruit & (y > 0))),
        "predicted_recruits": int(np.count_nonzero(recruit & (pred > 0))),
        "correct": int(np.count_nonzero(recruit & (pred == y))),
    }
    out["base_forced_none_n"] = int(forced.sum())
    return out


def generate_demo(seed, replay_path, max_steps):
    """New self-play by our old rule policy; no external strong-model replay input."""
    from fastsim import FastCarbon
    from eval_v53 import CFG, _extinct
    import policy_v0_5
    random.seed(seed)
    np.random.seed(seed)
    sim = FastCarbon(CFG).reset(seed=seed)
    agents = [policy_v0_5.make_agent(), policy_v0_5.make_agent()]
    steps = [[dict(observation=sim.obs(p), status="ACTIVE", action={}) for p in (0, 1)]]
    for _ in range(max_steps):
        acts = [agents[p](sim.obs(p), CFG) for p in (0, 1)]
        sim.step(*acts)
        post = sim.get_state()
        done = sim.step_count >= 299 or any(_extinct(post, p) for p in (0, 1))
        steps.append([dict(observation=sim.obs(p), status="DONE" if done else "ACTIVE", action=acts[p])
                      for p in (0, 1)])
        if done:
            break
    with gzip.open(replay_path, "wt") as f:
        json.dump({"steps": steps, "seed": seed, "source": "policy_v0_5-selfplay",
                   "purpose": "policy_only_bootstrap", "terminal_labels_used": False}, f)
    return len(steps) - 1


@torch.no_grad()
def diagnose(net, data, trainer, bs, arrays_path=None):
    net.eval()
    preds, totals = [], []
    for start in range(0, len(data["y"]), bs):
        idx = np.arange(start, min(start + bs, len(data["y"])))
        b = trainer.to_dev(trainer.fetch(data, idx), aug=False)
        lg, _, _ = net(b)
        if not torch.isfinite(lg).all():
            raise ValueError("non-finite diagnostic logits")
        preds.append(lg.argmax(-1).numpy())
        ce = torch.nn.functional.cross_entropy(lg.flatten(0, 1), b["y"].clamp(min=0).flatten(), reduction="none")
        mask = b["y"].flatten() >= 0
        totals.append((float(ce[mask].sum()), int(mask.sum())))
    pred = np.concatenate(preds)
    out = opportunity_counts(data["y"], data["tok_kind"][:, :12], data["legal"], pred)
    if arrays_path is not None:
        np.savez_compressed(arrays_path, y=data["y"], kind=data["tok_kind"][:, :12],
                            legal=data["legal"], pred=pred, gid=data["gid"], step=data["step"])
    out["unweighted_policy_ce"] = sum(t[0] for t in totals) / sum(t[1] for t in totals)
    out["interpretation"] = "teacher-forced replay diagnostic; not closed-loop strength"
    return out


def run(args):
    # Reject inherited switches that would silently alter this bounded experiment.
    inherited = [k for k in os.environ if k.startswith(("BC_", "CNET2_"))]
    if inherited:
        raise ValueError("unset inherited training/builder switches: " + ",".join(sorted(inherited)))
    os.environ["BC_DEVICE"] = "cpu"
    import cnet2_build as builder
    import cnet2_train as trainer
    from cnet2_model import CNet2
    from init_sapling import resolve_preset_metadata

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.optim_seed)
    rng = np.random.default_rng(args.optim_seed)
    init_bytes = Path(args.init).read_bytes()
    with np.load(io.BytesIO(init_bytes), allow_pickle=False) as archive:
        meta = json.loads(str(archive["meta"]))
        state = {k: archive[k].copy() for k in archive.files if k != "meta"}
    preset, cfg, expected_params = resolve_preset_metadata(meta)
    trainer._check_finite_arrays(state, args.init)
    net = CNet2(**cfg).cpu()
    net.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()}, strict=True)
    trainer._check_finite_arrays({k: v.detach().numpy() for k, v in net.state_dict().items()}, "loaded Sapling")
    actual_params = sum(p.numel() for p in net.parameters())
    if actual_params != expected_params:
        raise ValueError("unexpected Sapling parameter count")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    source_names = ("sapling_bootstrap.py", "policy_v0_5.py", "fastsim.py", "cnet2_feat.py",
                    "cnet2_build.py", "cnet2_train.py", "cnet2_model.py", "init_sapling.py", "eval_v53.py")
    source_hashes = {p: sha(HERE / p) for p in source_names}
    (out / "source").mkdir()
    for name in source_names:
        (out / "source" / name).write_bytes((HERE / name).read_bytes())
    train_seeds = list(range(args.seed0, args.seed0 + args.train_games))
    val_seeds = list(range(args.seed0 + args.train_games, args.seed0 + args.train_games + args.val_games))
    sources, groups = [], [[], []]
    t0 = time.monotonic()
    for group, seeds in enumerate((train_seeds, val_seeds)):
        for seed in seeds:
            replay = out / f"own-{seed}.json.gz"
            shard = out / f"own-{seed}.npz"
            transitions = generate_demo(seed, replay, args.steps)
            # Dummy builder margins are never trained or interpreted as terminal labels.
            result = builder.build_game((str(replay), [0, 1], seed, [args.style] * 2, [0, 0], str(shard)))
            if result[2] is not None or not result[1]:
                raise ValueError(f"builder failed: {result}")
            with np.load(shard, allow_pickle=False) as archive:
                groups[group].append({k: archive[k].copy() for k in archive.files})
            sources.append(dict(seed=seed, split="train" if group == 0 else "validation",
                                transitions=transitions, rows=result[1], replay_sha256=sha(replay), shard_sha256=sha(shard)))
    train, val = [{k: np.concatenate([d[k] for d in parts]) for k in parts[0]} for parts in groups]
    if set(train["gid"]) & set(val["gid"]):
        raise ValueError("game split leakage")
    uw, weight_report = role_action_weights(train, args.weight_cap)
    effective_uw = effective_unit_weights(uw, train["y"] >= 0, args.balance)
    if args.balance != "none":
        train["uw"] = effective_uw
    np.save(out / "train-unit-weights.npy", effective_uw, allow_pickle=False)
    weight_report["effective_unit_weights_sha256"] = sha(out / "train-unit-weights.npy")
    weight_report["effective_formula"] = {
        "none": "1 on valid labels; no uw passed to loss",
        "role-action": "train-only role/action balancing, capped then mean-normalized",
        "sqrt-role-action": "sqrt(role-action weights), re-normalized to mean 1 on valid labels",
    }[args.balance]
    if any(sum(weight_report["counts"][k]) == 0 for k in range(3)) or not sum(weight_report["counts"][2][1:3]):
        raise ValueError("demonstrations lack role or recruitment supervision")
    before = {"train": diagnose(net, train, trainer, args.bs),
              "validation": diagnose(net, val, trainer, args.bs, out / "validation-before.npz")}
    # No fabricated value/aux target enters this loss; freeze those output layers.
    frozen = ("value.", "utility.", "wdl.", "margin.", "aux.")
    snapshot = {k: v.detach().clone() for k, v in net.state_dict().items() if k.startswith(frozen)}
    for name, param in net.named_parameters():
        param.requires_grad_(not name.startswith(frozen))
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    losses = []
    net.train()
    for update in range(args.updates):
        ii = rng.choice(len(train["y"]), min(args.bs, len(train["y"])), replace=False)
        b = trainer.to_dev(trainer.fetch(train, ii), aug=True)
        loss, _, _, _ = trainer.losses(net, b)
        if not torch.isfinite(loss):
            raise ValueError("non-finite policy loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0, error_if_nonfinite=True)
        opt.step()
        losses.append(dict(update=update + 1, policy_loss=float(loss), grad_norm=float(norm)))
    if any(not torch.equal(net.state_dict()[k], v) for k, v in snapshot.items()):
        raise ValueError("unused output heads changed")
    if source_hashes != {p: sha(HERE / p) for p in source_names}:
        raise ValueError("source changed during run; refusing publication")
    training_args = {k: v for k, v in vars(args).items() if k not in ("init", "out")}
    report = dict(format="sapling-own-policy-bootstrap/1", status="diagnostic_only",
                  params=actual_params, preset=preset, cfg=net.cfg, style=args.style, args=training_args,
                  invocation_paths={"init": str(Path(args.init).resolve()), "out": str(out.resolve())},
                  init_sha256=hashlib.sha256(init_bytes).hexdigest(), source_policy="policy_v0_5.py",
                  source_hashes=source_hashes,
                  external_or_exam_replays_consumed=0, train_seeds=train_seeds, validation_seeds=val_seeds,
                  sources=sources, weighting=weight_report, effective_balance=args.balance,
                  loss="existing cnet2_train.losses policy CE only", value_labels_used=False,
                  unused_output_heads_bitwise_unchanged=True, before=before,
                  after={"train": diagnose(net, train, trainer, args.bs),
                         "validation": diagnose(net, val, trainer, args.bs, out / "validation-after.npz")},
                  updates=losses, seconds=time.monotonic() - t0,
                  limitations="Few self-generated rule demonstrations; teacher-forced validation only. No A4 selection, U/value training, formal distillation or strength certification. Recruitment opportunity uses existing approximate Prefix legality.")
    trainer.export(net, str(out / "trained.npz"), dict(cfg=net.cfg, model_name="Sapling", params=actual_params, preset=preset,
                   untrained=False, value_heads_untrained=True, training_status="policy_only_diagnostic",
                   sapling_bootstrap={k: report[k] for k in ("format", "status", "init_sha256", "source_hashes", "train_seeds", "validation_seeds", "style", "args")}))
    report["trained_sha256"] = sha(out / "trained.npz")
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"out": str(out), "seconds": report["seconds"], "train_rows": len(train["y"]), "validation_rows": len(val["y"]),
                      "validation_ce_before": before["validation"]["unweighted_policy_ce"],
                      "validation_ce_after": report["after"]["validation"]["unweighted_policy_ce"],
                      "trained_sha256": report["trained_sha256"]}, allow_nan=False))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--init", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--train-games", type=int, default=4)
    p.add_argument("--val-games", type=int, default=2)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--updates", type=int, default=64)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.0003)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed0", type=int, default=620000000)
    p.add_argument("--optim-seed", type=int, default=551187)
    p.add_argument("--style", type=int, choices=range(8), default=2)
    p.add_argument("--balance", choices=("none", "role-action", "sqrt-role-action"), default="none")
    p.add_argument("--weight-cap", type=float, default=8.0)
    args = p.parse_args()
    if min(args.train_games, args.val_games, args.updates, args.bs, args.threads) < 1 or not 1 <= args.steps <= 299:
        p.error("positive budgets required; steps must be 1..299")
    if not np.isfinite(args.lr) or args.lr <= 0 or not np.isfinite(args.weight_cap) or args.weight_cap <= 0:
        p.error("learning rate and weight cap must be positive and finite")
    if not 0 <= args.seed0 <= 2**31 - args.train_games - args.val_games:
        p.error("seed range must fit signed int32 game IDs")
    run(args)


if __name__ == "__main__":
    main()
