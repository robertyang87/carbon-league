"""Train the 1.2M Sapling actor on economic decisions visited by frozen V2.

The student never receives the comparison teacher's other actions at deployment.
Training uses the documented teacher-forced prefix. Validation reports changes
on held-out maps, including economic slots where the two policies disagreed.
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
import torch.nn.functional as functional

from cnet2_model import CNet2
from init_sapling import resolve_preset_metadata
from sapling_bootstrap import sha
from sapling_elite_bc import load_games

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "data/weights/sapling_1200k/top5_h2h_v2.npz"
PARENT_SHA256 = "cef478dcbb96b1ac9304b9da56623cff578994283e71a6893dd6530978a33646"
ANCHOR_MANIFEST = ROOT / "reports/sapling_1200k/top5_h2h_data.json"
TRAINEE_SEED = 551190
MIX = 0.70
ECON_ROOTS, ANCHOR_ROOTS = 16, 8
UPDATES = 320
LEARNING_RATE = 0.0003


def checked_shard(manifest, game):
    raw = (manifest.parent / game["shard"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != game["sha256"]:
        raise ValueError(f"shard changed for game {game['id']}")
    return np.load(io.BytesIO(raw), allow_pickle=False)


def load_corrections(manifest, trainer):
    data, games, digest = load_games(manifest, trainer)
    fingerprints = [g["seed"] for g in games]
    expected = {
        "train": set(range(620113400, 620113416)) | set(range(620113500, 620113506)),
        "val": set(range(620113416, 620113420)) | set(range(620113506, 620113508)),
    }
    if any({g["seed"] for g in games if g["split"] == group} != expected[group] for group in expected):
        raise ValueError("cash V3 training needs the complete frozen 22/6-map correction split")
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("correction dataset repeats a map")
    for group, d in data.items():
        masks, executed = [], []
        for g in games:
            if g["split"] != group:
                continue
            with checked_shard(manifest, g) as z:
                economic, behavior = z["econ_mask"], z["execution_y"]
                if economic.dtype != np.bool_ or economic.shape != z["y"].shape or behavior.shape != economic.shape:
                    raise ValueError("invalid slot masks or execution labels")
                if np.any(economic & (z["y"] < 0)) or not np.array_equal(behavior >= 0, z["y"] >= 0):
                    raise ValueError("masks or executed labels disagree on occupied slots")
                masks.append(economic)
                executed.append(behavior)
        d["econ"] = np.concatenate(masks) & free_slots(d)
        d["executed"] = np.concatenate(executed)
        d["opponents"] = np.concatenate([np.full(g["rows"], g["opponent"])
                                          for g in games if g["split"] == group])
        if not d["econ"].any():
            raise ValueError(f"zero economic decisions in {group}")
    return data, games, digest


def free_slots(d):
    return (d["y"] >= 0) & (d["legal"].sum(-1) >= 2)


def load_anchor(manifest, trainer, correction_games):
    if sha(manifest) != sha(ANCHOR_MANIFEST):
        raise ValueError("anchor manifest differs from the frozen V2 corpus")
    games = json.loads(manifest.read_text())["games"]
    seen = {g["seed"] for g in correction_games}
    if seen & {g["seed"] for g in games}:
        raise ValueError("anchor and correction games share a map")
    rng = np.random.default_rng(TRAINEE_SEED + 1)
    parts = {"train": [], "val": []}
    for g in games:
        # Cap rehearsal memory and use every source match. Select rows
        # without inspecting labels, future cash, or the model's scores.
        take = 64 if g["split"] == "train" else 16
        rows = np.sort(rng.choice(g["rows"], size=min(take, g["rows"]), replace=False))
        with checked_shard(manifest, g) as archive:
            one = {k: archive[k][rows] for k in trainer.LOAD}
        if not (one["teacher"] == 2).all():
            raise ValueError("anchor style differs from deployment")
        parts[g["split"]].append(one)
    return {group: {k: np.concatenate([part[k] for part in group_parts]) for k in group_parts[0]}
            for group, group_parts in parts.items()}


def metrics(ce, correct, mask):
    n = int(mask.sum())
    return dict(slots=n, correct=int((correct & mask).sum()),
                agreement=float(correct[mask].mean()) if n else None,
                ce=float(ce[mask].mean(dtype=np.float64)) if n else None)


@torch.inference_mode()
def evaluate(net, data, trainer):
    net.eval()
    ce_parts, predictions = [], []
    for start in range(0, len(data["y"]), 32):
        indices = np.arange(start, min(start + 32, len(data["y"])))
        batch = trainer.to_dev(trainer.fetch(data, indices), aug=False)
        logits = net(batch)[0]
        if not torch.isfinite(logits).all():
            raise ValueError("non-finite validation logits")
        ce = functional.cross_entropy(logits.transpose(1, 2), batch["y"].clamp_min(0), reduction="none")
        ce_parts.append(ce.cpu().numpy())
        predictions.append(logits.argmax(-1).cpu().numpy())
    ce = np.concatenate(ce_parts)
    correct = np.concatenate(predictions) == data["y"]
    result = dict(all_free=metrics(ce, correct, free_slots(data)))
    if "econ" in data:
        econ = data["econ"]
        result["economic"] = metrics(ce, correct, econ)
        conflict = econ & (data["y"] != data["executed"])
        result["expert_differs_from_executed"] = metrics(ce, correct, conflict)
        result["economic_by_opponent"] = {
            str(opp): metrics(ce, correct, econ & (data["opponents"] == opp)[:, None])
            for opp in np.unique(data["opponents"])
        }
        kind = data["tok_kind"][:, :econ.shape[1]]
        result["economic_by_role"] = {role: metrics(ce, correct, econ & (kind == i))
                                      for i, role in enumerate(("collector", "planter", "base"))}
    return result


def run(a):
    if a.out.exists() or a.receipt.exists() or a.out.resolve() == a.receipt.resolve():
        raise FileExistsError("fresh, distinct weight and receipt paths required")
    if any(k.startswith(("BC_", "CNET2_")) for k in os.environ):
        raise ValueError("unset BC_* and CNET2_* switches")
    os.environ["BC_DEVICE"] = a.device
    import cnet2_train as trainer

    start = time.monotonic()
    torch.set_num_threads(a.threads)
    torch.manual_seed(TRAINEE_SEED)
    rng = np.random.default_rng(TRAINEE_SEED)
    raw_parent = a.init.read_bytes()
    if hashlib.sha256(raw_parent).hexdigest() != PARENT_SHA256:
        raise ValueError("cash V3 requires the frozen public Top5 H2H V2")
    with np.load(io.BytesIO(raw_parent), allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        preset, cfg, params = resolve_preset_metadata(meta)
        if preset != "sapling-1.2m" or params != 1206476:
            raise ValueError("wrong Sapling architecture")
        net = CNet2(**cfg)
        net.load_state_dict({k: torch.from_numpy(z[k]) for k in z.files if k != "meta"}, strict=True)
    net.to(trainer.DEV)
    for name, parameter in net.named_parameters():
        if name.startswith(("value.", "utility.", "wdl.", "margin.", "aux.")):
            parameter.requires_grad_(False)

    correction, games, generation = load_corrections(a.manifest, trainer)
    anchor = load_anchor(a.anchor_manifest, trainer, games)
    econ = correction["train"]
    replay = anchor["train"]
    econ_rows = np.flatnonzero(econ["econ"].any(-1))
    replay_mask = free_slots(replay)
    replay_rows = np.flatnonzero(replay_mask.any(-1))

    def validation():
        return dict(correction=evaluate(net, correction["val"], trainer),
                    rehearsal=evaluate(net, anchor["val"], trainer))

    before = validation()
    net.train()
    optimizer = torch.optim.AdamW((p for p in net.parameters() if p.requires_grad),
                                  lr=LEARNING_RATE, weight_decay=1e-4)
    losses = []
    exposures = np.zeros_like(econ["y"], dtype=np.int32)
    for step in range(UPDATES):
        i = np.sort(rng.choice(econ_rows, ECON_ROOTS, replace=len(econ_rows) < ECON_ROOTS))
        j = np.sort(rng.choice(replay_rows, ANCHOR_ROOTS, replace=len(replay_rows) < ANCHOR_ROOTS))
        x, z = trainer.fetch(econ, i), trainer.fetch(replay, j)
        batch = trainer.to_dev({k: np.concatenate([x[k], z[k]]) for k in trainer.LOAD}, aug=False)
        logits = net(batch)[0]
        ce = functional.cross_entropy(logits.transpose(1, 2), batch["y"].clamp_min(0), reduction="none")
        econ_mask = torch.from_numpy(econ["econ"][i]).to(trainer.DEV)
        broad_mask = torch.from_numpy(replay_mask[j]).to(trainer.DEV)
        loss = MIX * ce[:len(i)][econ_mask].mean() + (1 - MIX) * ce[len(i):][broad_mask].mean()
        if not torch.isfinite(loss):
            raise ValueError("non-finite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        np.add.at(exposures, i, econ["econ"][i].astype(np.int32))
        losses.append(float(loss.detach()))
        if (step + 1) % 64 == 0:
            print(f"{step+1}/{UPDATES} loss {losses[-1]:.4f}", flush=True)

    after = validation()
    trainer._check_finite_arrays({k: v.detach().cpu().numpy() for k,v in net.state_dict().items()},
                                 "cash V3")
    recipe = dict(parent_sha256=PARENT_SHA256, data_manifest_sha256=generation,
                  top5_manifest_sha256=sha(a.anchor_manifest), trainer_sha256=sha(__file__),
                  seed=TRAINEE_SEED, updates=UPDATES, learning_rate=LEARNING_RATE,
                  optimizer="AdamW", weight_decay=1e-4, grad_clip=5., aug=False,
                  correction_rows_per_batch=ECON_ROOTS, anchor_rows_per_batch=ANCHOR_ROOTS,
                  correction_fraction_of_loss=MIX, anchor_choice_per_game_train=64,
                  anchor_choice_per_game_validation=16, style=2,
                  inference_policy="unchanged Sapling int8 actor",
                  loss="two independently normalized unweighted means over legal choices, "
                       "corrections restricted to pre-action economic slots",
                  non_policy_heads="frozen; backbone trained, no value supervision")
    receipts = dict(format="sapling-cash-dagger/1", params=params, **recipe, before=before, after=after,
                    games=[{k: g[k] for k in ("id","seed","split","rows","sha256")} for g in games],
                    correction_train_root_count=len(econ["y"]),
                    train_rows_with_economic_choice=len(econ_rows),
                    economic_slots=int(econ["econ"].sum()), unique_economic_slots_seen=int((exposures>0).sum()),
                    corrected_slots=int((econ["econ"] & (econ["y"] != econ["executed"])).sum()),
                    sampled_economic_slots=int(exposures.sum()), losses=losses,
                    runtime=dict(device=str(trainer.DEV), threads=a.threads, torch=str(torch.__version__),
                                 numpy=np.__version__, elapsed_seconds=time.monotonic()-start))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=a.out.parent) as d:
        stage = Path(d) / "cash.npz"
        trainer.export(net, str(stage), dict(cfg=net.cfg, model_name="Sapling", preset=preset,
                       params=params, untrained=False, value_heads_untrained=True,
                       training_status="student_state_economic_correction", cash_bc=recipe))
        receipts["trained_sha256"] = sha(stage)
        report = json.dumps(receipts, indent=2, allow_nan=False) + "\n"
        with tempfile.NamedTemporaryFile(dir=a.receipt.parent, mode="w", delete=False) as f:
            f.write(report)
            staged_report = Path(f.name)
        try:
            os.link(stage, a.out)
            try:
                os.link(staged_report, a.receipt)
            except BaseException:
                a.out.unlink()
                raise
        finally:
            staged_report.unlink(missing_ok=True)
    print(receipts["trained_sha256"], "held-out economic", before["correction"]["economic"],
          after["correction"]["economic"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--anchor-manifest", type=Path, required=True)
    p.add_argument("--init", type=Path, default=PARENT)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--device", choices=("cpu","cuda"), default="cpu")
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    if args.threads < 1:
        p.error("--threads must be positive")
    run(args)
