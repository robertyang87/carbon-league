"""Fit and assess a native Sapling policy with optional training-only worker destinations."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
NATIVE = HERE.parent / "cargo_tree_action_learning_dev" / "native_actor_corpus"
sys.path.insert(0, str(NATIVE))
import fit as native

common = sys.modules["common"]
RULE = common.read(HERE / "learning.json")
SPLITS = {"train": (24, 64), "val": (4, 48), "dev": (4, 48)}
N_WORKER = 10
ROLES = ("collector", "planter", "base")


class Future(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.direction = nn.Linear(width, 5)
        self.survival = nn.Linear(width, 1)
        for parameter in self.parameters():
            nn.init.zeros_(parameter)


def canonical_directions(direction, seats):
    import cnet2_feat as feat
    result = direction.copy()
    if not np.isin(seats, (0, 1)).all() or not np.isin(direction, (-1, 0, 1, 2, 3, 4)).all():
        raise ValueError("invalid raw direction or physical seat")
    valid = (seats[:, None] == 1) & (direction >= 0)
    result[valid] = feat.CANON_MOVE[direction[valid]]
    return result


def check_direction_frame():
    import cnet2_feat as feat
    # Test both directions across each axis at the wrap seam, standstill, and a masked worker.
    cells = [(14, 0), (0, 14), (0, 210), (210, 0), (7, 7)]
    source = [feat.move_between(a, b) for a, b in cells] + [-1]
    expected = [feat.move_between(feat.CANON[a], feat.CANON[b]) for a, b in cells] + [-1]
    actual = canonical_directions(np.array([source, source]), np.array([0, 1]))
    if actual.tolist() != [source, expected] or expected != [4, 2, 1, 3, 0, -1]:
        raise ValueError("direction target disagrees with the native seat-normalized movement frame")


def read_split(split):
    """Read only the selected partition, with independently checkable bytes and equal maps."""
    summary = common.read(HERE / "data.json")
    terrains = {s: {c["terrain"] for c in summary["shards"] if c["split"] == s} for s in SPLITS}
    if sum(map(len, terrains.values())) != len(set.union(*terrains.values())):
        raise ValueError("overlapping physical map between partitions")
    if any(len(terrains[s]) != shape[0] for s, shape in SPLITS.items()):
        raise ValueError("unexpected physical map count")
    parts = {}
    geography = []
    seats = []
    expected_roots = SPLITS[split][1]
    for card in sorted((c for c in summary["shards"] if c["split"] == split),
                       key=lambda c: (c["terrain"], c["expert_seat"])):
        path = HERE / card["file"]
        if common.sha(path) != card["sha256"]:
            raise ValueError(f"changed feature archive {path}")
        with np.load(path, allow_pickle=False) as src:
            roots = len(src["y"])
            for key in (*common.XKEYS, "uf", "mf", "y", "legal", "eligible",
                        "fixed", "parent", "dst12", "survive12"):
                parts.setdefault(key, []).append(src[key])
        geography.extend([card["terrain"]] * roots)
        seats.extend([int(card["expert_seat"])] * roots)
    if set(Counter(zip(geography, seats)).values()) != {expected_roots // 2}:
        raise ValueError("map and expert seat not equally represented")
    if len(geography) != len(terrains[split]) * expected_roots:
        raise ValueError("incomplete sample")
    data = {k: np.concatenate(values) for k, values in parts.items()}
    direction, survival = data["dst12"], data["survive12"]
    if (direction.shape != data["y"].shape or survival.shape != data["y"].shape or
            not np.isin(survival, (-1, 0, 1)).all() or not np.isin(direction, (-1, 0, 1, 2, 3, 4)).all() or
            np.any((direction >= 0) != (survival == 1))):
        raise ValueError("future labels inconsistent")
    kinds = data["tok_kind"][:, :survival.shape[1]]
    active = data["tok_mask"][:, :survival.shape[1]]
    if np.any((survival >= 0) & (~active | (kinds == 2))) or np.any(survival[:, N_WORKER:] >= 0):
        raise ValueError("worker identity in auxiliary data")
    data["terrain"] = np.array(geography)
    data["seat"] = np.array(seats)
    data["dst12"] = canonical_directions(direction, data["seat"])
    return data


def outputs(model, batch, auxiliary=None):
    """Capture exactly the worker hidden state consumed by the native per-worker action head."""
    hidden = []
    if auxiliary is not None:
        def worker_state(module, args):
            hidden.append(args[0])
        hook = model.whead.register_forward_pre_hook(worker_state)
    try:
        student_logits = model(batch)[0]
    finally:
        if auxiliary is not None:
            hook.remove()
    scores = (batch["residual"] + 0.25 * student_logits).masked_fill(~batch["legal"], -1e9)
    if auxiliary is None:
        return scores, None, None
    if len(hidden) != N_WORKER:
        raise ValueError("unexpected native worker GRU/head layout")
    h = torch.stack(hidden, dim=1)
    return scores, auxiliary.direction(h), auxiliary.survival(h).squeeze(-1)


def immediate_loss(scores, b, parent):
    logp = scores.log_softmax(-1)
    log0 = parent.masked_fill(~b["legal"], -1e9).log_softmax(-1)
    ce = -logp.gather(-1, b["y"].clamp_min(0).unsqueeze(-1)).squeeze(-1)
    kl = (log0.exp() * (log0 - logp)).sum(-1)
    weights = b["eligible"].float()
    role = b["tok_kind"][:, :weights.shape[1]]
    by_role = torch.zeros_like(weights)
    count_roles = torch.zeros_like(weights[:, :1])
    for k in range(len(ROLES)):
        in_role = weights * (role == k)
        count = in_role.sum(1, keepdim=True)
        by_role += in_role / count.clamp_min(1)
        count_roles += count > 0
    w = .5 * (by_role / count_roles.clamp_min(1) + weights / weights.sum(1, keepdim=True).clamp_min(1))
    return (w * ce).sum(1).mean(), (w * kl).sum(1).mean()


def future_loss(logits, labels, logistic):
    mask = labels >= 0
    if logistic:
        loss = F.binary_cross_entropy_with_logits(logits, labels.clamp_min(0).float(), reduction="none")
    else:
        loss = F.cross_entropy(logits.transpose(1, 2), labels.clamp_min(0), reduction="none")
    return ((loss * mask).sum(1) / mask.sum(1).clamp_min(1)).mean()


def batch(data, ids):
    b = native.batch(data, ids)
    return (b, torch.from_numpy(data["parent"][ids]),
            torch.from_numpy(data["dst12"][ids, :N_WORKER]).long(),
            torch.from_numpy(data["survive12"][ids, :N_WORKER]).long())


def training_priors(data):
    counts = {}
    for k, classes in (("dst12", 5), ("survive12", 2)):
        labels = data[k][:, :N_WORKER].ravel()
        counts[k] = np.bincount(labels[labels >= 0], minlength=classes).tolist()
    return counts


def training():
    data = read_split("train")
    m = common.package()
    if common.sha(common.BASE) != RULE["parent_weights_sha256"] or common.sha(common.PACK) != RULE["parent_package_sha256"]:
        raise ValueError("incorrect Sapling baseline")
    with np.load(common.BASE, allow_pickle=False) as archive:
        if native.repack.quantize(common.BASE, archive)["_W64"] != m._W64:
            raise ValueError("weight archive does not reproduce parent package")
    arms = ("bc", "joint")
    models, optimizers = {}, {}
    for arm in arms:
        models[arm] = common.torch_model(m, 1)
        for name, p in models[arm].named_parameters():
            p.requires_grad_(not name.startswith(native.VALUE))
    aux = Future(models["joint"].cfg["D"])
    for arm in arms:
        params = [p for p in models[arm].parameters() if p.requires_grad]
        if arm == "joint":
            params.extend(aux.parameters())
        optimizers[arm] = torch.optim.AdamW(params, lr=RULE["lr"], weight_decay=RULE["weight_decay"])
    parity = native.parity_check(data, models["bc"], root_count=SPLITS["train"][1] // 2)
    order = np.random.default_rng(RULE["seed"]).permutation(len(data["y"])).reshape(-1, RULE["batch"])
    if len(order) != RULE["steps"] or len(np.unique(order)) != len(data["y"]):
        raise ValueError("training exposure mismatch")
    log = []
    start = time.monotonic()
    for step, ids in enumerate(order, 1):
        b, parent, direction, survival = batch(data, ids)
        row = {}
        for arm in arms:
            model, opt = models[arm], optimizers[arm]
            opt.zero_grad(set_to_none=True)
            logits, d, s = outputs(model, b, aux if arm == "joint" else None)
            ce, kl = immediate_loss(logits, b, parent)
            loss = ce + RULE["kl"] * kl
            metrics = {"ce": ce, "kl": kl}
            if arm == "joint":
                dc, sc = future_loss(d, direction, False), future_loss(s, survival, True)
                loss = loss + RULE["direction_ce"] * dc + RULE["survival_bce"] * sc
                metrics.update(direction=dc, survival=sc)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training objective")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(opt.param_groups[0]["params"], RULE["clip_gradient_norm"],
                                                   error_if_nonfinite=True)
            if step == 1 and (model.stem.weight.grad.abs().sum() == 0 or model.gru.weight_ih.grad.abs().sum() == 0):
                raise ValueError("update missing in policy stem or GRU")
            opt.step()
            row[arm] = {k: float(v.detach()) for k, v in dict(metrics, gradient_norm=norm).items()}
        log.append(row)
        if step % 32 == 0:
            print("trained", step, "of", len(order), round(time.monotonic()-start, 1), flush=True)
    common.save(HERE / "loss.json.gz", log)
    np.savez(HERE / "joint_aux.npz", **{k: v.detach().numpy() for k, v in aux.state_dict().items()})
    released = {arm: native.export(models[arm], m, arm, HERE) for arm in arms}
    for info in released.values():
        changed = info["quantized_changed_by_tensor"]
        if not all(changed[k] > 0 for k in ("stem.weight", "gru.weight_ih", "whead.weight")):
            raise ValueError("export lost an essential trainable block")
        if info["package_bytes"] > 10_000_000:
            raise ValueError("single-file package exceeds platform size")
    receipt = dict(arms=released, parent_sha256=common.sha(common.PACK), parent_weights_sha256=common.sha(common.BASE),
                   learning_sha256=common.sha(HERE/"learning.json"), data_sha256=common.sha(HERE/"data.json"),
                   fit_code_sha256=common.sha(Path(__file__)), priors=training_priors(data), parity=parity,
                   roots=len(data["y"]), terrains=len(set(data["terrain"])), updates=len(order),
                   order_int64_sha256=hashlib.sha256(order.astype("<i8").tobytes()).hexdigest(),
                   loss_sha256=common.sha(HERE/"loss.json.gz"), aux_sha256=common.sha(HERE/"joint_aux.npz"),
                   torch=torch.__version__, numpy=np.__version__, seconds=time.monotonic()-start,
                   input_code={p.name: common.sha(p) for p in (NATIVE/"fit.py", NATIVE/"common.py",
                               Path(native.repack.__file__), common.ROOT/"code"/"cnet2_model.py")})
    common.save(HERE / "fit.json", receipt)
    print("released", {a: r["package_sha256"] for a, r in released.items()}, flush=True)


def summary(logits, target, mask):
    y = target[mask]
    lp = logits[mask]
    if not len(y):
        return dict(n=0, correct=0, nll_sum=0.0, accuracy=None, nll=None)
    correct = int((lp.argmax(1) == y).sum())
    loss = float(-lp[np.arange(len(y)), y].sum(dtype=np.float64))
    return dict(n=len(y), correct=correct, nll_sum=loss, accuracy=correct/len(y), nll=loss/len(y))


def summarize_slices(data, lp, labels, valid):
    n = valid.shape[1]
    roles = data["tok_kind"][:, :n]
    return dict(overall=summary(lp, labels, valid),
                by_map={terrain: summary(lp, labels, valid & (data["terrain"][:, None] == terrain))
                        for terrain in sorted(set(data["terrain"]))},
                by_expert_seat={str(s): summary(lp, labels, valid & (data["seat"][:, None] == s))
                                for s in (0, 1)},
                by_role={role: summary(lp, labels, valid & (roles == i)) for i, role in enumerate(ROLES)})


def held():
    release = common.read(HERE/"fit.json")
    if (release["fit_code_sha256"] != common.sha(Path(__file__)) or
            release["learning_sha256"] != common.sha(HERE/"learning.json") or
            release["data_sha256"] != common.sha(HERE/"data.json")):
        raise ValueError("evaluation program/registration/data changed since fitting")
    models = {"parent": common.torch_model(common.package(), 1)}
    for arm in ("bc", "joint"):
        path = HERE/(arm+".py")
        if common.sha(path) != release["arms"][arm]["package_sha256"]:
            raise ValueError("held evaluation package mismatch")
        models[arm] = common.torch_model(common.package(path), 1)
    if common.sha(HERE/"joint_aux.npz") != release["aux_sha256"]:
        raise ValueError("different training-only readout")
    aux = Future(models["joint"].cfg["D"])
    with np.load(HERE/"joint_aux.npz", allow_pickle=False) as archive:
        aux.load_state_dict({k: torch.from_numpy(archive[k]) for k in archive.files})
    all_scores = {}
    for split in ("val", "dev"):
        data = read_split(split)
        result = {}
        for arm, model in models.items():
            commands, directions, survivals = [], [], []
            with torch.inference_mode():
                for ids in np.array_split(np.arange(len(data["y"])), len(data["y"])//RULE["batch"]):
                    b, _, _, _ = batch(data, ids)
                    scores, d, s = outputs(model, b, aux if arm == "joint" else None)
                    commands.append(scores.log_softmax(-1).numpy())
                    if arm == "joint":
                        directions.append(d.log_softmax(-1).numpy())
                        survivals.append(torch.stack((F.logsigmoid(-s), F.logsigmoid(s)), -1).numpy())
            result[arm] = dict(current=summarize_slices(data, np.concatenate(commands), data["y"], data["eligible"]))
            if arm == "joint":
                for key, parts in (("dst12", directions), ("survive12", survivals)):
                    truth = data[key][:, :N_WORKER]
                    lp = np.concatenate(parts)
                    head = summarize_slices(data, lp, truth, truth >= 0)
                    counts = np.array(release["priors"][key], np.float64)
                    empirical = counts / counts.sum()
                    if np.any(empirical <= 0):
                        raise ValueError("training-frequency reference has unsupported class")
                    baseline = np.broadcast_to(np.log(empirical), lp.shape)
                    result[arm][key] = dict(head=head, train_frequency_baseline=summarize_slices(
                        data, baseline, truth, truth >= 0), train_prior=empirical.tolist())
        all_scores[split] = result
    common.save(HERE/"held.json", dict(fit_sha256=common.sha(HERE/"fit.json"),
                meaning="Held teacher-forced expert prefixes; each arm uses its deployed quantized student. No game result.",
                splits=all_scores))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fit", "held"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(RULE["seed"])
    torch.use_deterministic_algorithms(True)
    check_direction_frame()
    if args.stage == "fit":
        training()
    else:
        held()
