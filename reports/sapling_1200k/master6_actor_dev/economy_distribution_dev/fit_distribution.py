"""Matched real-policy fits using bounded train-only supports for future public cash."""
from pathlib import Path
import gzip
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
AUX = HERE.parent / "economy_aux_dev"
sys.path.insert(0, str(AUX))
import train as old

base, P = old.base, old.P
TARGETS = [f"{name}_{h}" for h in (10, 30) for name in old.NAMES]


def read(path):
    return json.loads(path.read_bytes())


def support(cash, mask):
    edges = np.empty((8, 7))
    means = np.zeros((8, 9))
    counts = np.zeros((8, 9), int)
    alerts = np.zeros_like(means)
    sd = np.zeros(8)
    maximum = np.zeros(8)
    q95 = np.zeros(8)
    for j in range(8):
        v = cash[mask[:, j], j]
        edges[j] = np.quantile(v[v > 1e-8], np.arange(1, 8) / 8)
        group = categories(v, edges[j])
        for k in np.unique(group):
            subset = v[group == k]
            counts[j, k] = len(subset)
            means[j, k] = subset.mean()
            alerts[j, k] = (subset >= 500).mean()
        sd[j] = max(25, v.std())
        maximum[j], q95[j] = v.max(), np.quantile(v, .95)
    return dict(edges=edges, means=means, counts=counts, within_alert=alerts,
                sd=sd, maximum=maximum, q95=q95)


def categories(v, edges):
    return np.where(v <= 1e-8, 0, 1 + np.searchsorted(edges, v, side="left"))


def corpus():
    actor = old.ACTOR
    split = actor / "data_split.json"
    receipts = json.loads(gzip.decompress((actor / "data_receipts.json.gz").read_bytes()))
    if old.sha(split) != receipts["split_sha256"]:
        raise ValueError("split changed")
    records = [g for g in read(split)["games"] if g["split"] in old.PARTS and
               "master3s2" not in g["opponent"].lower()]
    return records, {g["id"]: g["sha256"] for g in receipts["games"] if "sha256" in g}


def fit():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    records, receipts = corpus()
    d, hashes = base.public("train", records, old.SHARDS, receipts)
    ylog, mask, sources = old.targets(records, d)
    previous = read(AUX / "result.json")
    if {"shards": hashes, "replays": sources} != previous["sources"]["train"] or len(sources) != 216:
        raise ValueError("training exposure mismatch")
    cash = np.expm1(ylog.astype(np.float64))
    support_data = support(cash, mask)
    np.savez(HERE / "support.npz", **support_data)
    groups = np.stack([categories(cash[:, j], support_data["edges"][j]) for j in range(8)], 1)
    t_cash = torch.tensor(cash, dtype=torch.float32)
    t_mask = torch.from_numpy(mask)
    t_group = torch.from_numpy(groups)
    means = torch.tensor(support_data["means"], dtype=torch.float32)
    sd = torch.tensor(support_data["sd"], dtype=torch.float32)
    valid = torch.from_numpy(support_data["counts"] > 0)
    probs = support_data["counts"] / support_data["counts"].sum(1, keepdims=True)
    active = base.actionable(d)
    rows = np.flatnonzero(active.any(1))
    order = np.random.default_rng(P["seed"]).permutation(rows)[:P["steps"] * P["batch"]]
    order_sha = base.digest(order.tobytes())
    if order_sha != previous["training_order_sha256"]:
        raise ValueError("wrong training batch order")
    kinds = d["tok_kind"][:, :active.shape[1]]
    counts = [(active & (kinds == j)).sum() for j in range(3)]
    role = torch.tensor(np.array(P["role_weights"]) * len(rows) / counts, dtype=torch.float32)
    opts = read(old.ACTOR / "training_protocol.json")["learn"]
    path = base.WEIGHTS / "master6_actor_v1.npz"
    parent, meta = base.model(path, P["starting_actor"])
    teacher = base.predict(parent, d)
    print("verified train", len(sources), len(order), flush=True)
    products = {}
    for arm in ("control", "candidate"):
        torch.manual_seed(P["seed"])
        net, _ = base.model(path, P["starting_actor"])
        head = torch.nn.Linear(net.h0.in_features, 72)
        with torch.no_grad():
            head.bias.copy_(torch.tensor(np.log(np.maximum(probs, 1e-12)).reshape(-1)))
        learn = [[], []]
        for n, w in net.named_parameters():
            w.requires_grad_(n.startswith(tuple(opts)))
            if w.requires_grad:
                learn[int(not n.startswith(tuple(opts[:2])))].append(w)
        opt = torch.optim.AdamW([{"params": learn[0], "lr": P["encoder_lr"]},
                                 {"params": learn[1], "lr": P["decoder_lr"]}], weight_decay=1e-4)
        opt_head = torch.optim.AdamW(head.parameters(), lr=P["readout_lr"], weight_decay=1e-4)
        net.train()
        for step, ix in enumerate(order.reshape(-1, P["batch"]), 1):
            b = base.batch(d, ix)
            tok, ctx, _ = net.encode(b)
            action = old.objective(net.decode(b, tok, ctx), torch.from_numpy(teacher[ix]), b, role)
            raw = head(ctx if arm == "candidate" else ctx.detach()).reshape(-1, 8, 9)
            logits = raw.masked_fill(~valid, -1e9)
            bin_ce = F.cross_entropy(logits.reshape(-1, 9), t_group[ix].flatten(),
                                     reduction="none").reshape(-1, 8)
            expectation = (logits.softmax(-1) * means).sum(-1)
            err = ((expectation - t_cash[ix]) / sd).square()
            aux = (bin_ce / np.log(9) + .25 * err)[t_mask[ix]].mean()
            loss = action + P["income_coefficient"] * aux
            if not torch.isfinite(loss):
                raise ValueError("nonfinite loss")
            opt.zero_grad(set_to_none=True)
            opt_head.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(learn[0] + learn[1], 2., error_if_nonfinite=True)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 2., error_if_nonfinite=True)
            opt.step()
            opt_head.step()
            if step % 30 == 0:
                print(arm, step, float(aux.detach()), flush=True)
        actor_file, head_file = HERE / (arm + ".npz"), HERE / (arm + "_readout.npz")
        base.trainer.export(net, str(actor_file), dict(meta, economic_distribution=arm))
        np.savez(head_file, weight=head.weight.detach().numpy(), bias=head.bias.detach().numpy())
        previous_control, _ = base.model(AUX / "control.npz", previous["weights"]["control"]["actor_sha256"])
        products[arm] = {"actor_sha256": old.sha(actor_file), "readout_sha256": old.sha(head_file),
                         "actor_parameters": sum(w.numel() for w in net.parameters()),
                         "changed_parameters_vs_parent": sum((p != dict(parent.named_parameters())[n]).sum().item()
                             for n, p in net.named_parameters()),
                         "changed_parameters_vs_previous_detached": sum((p != dict(previous_control.named_parameters())[n]).sum().item()
                             for n, p in net.named_parameters())}
    base.write_json(HERE / "fit_receipt.json", {"weights": products, "protocol_sha256": old.sha(HERE / "protocol.json"),
        "fit_source_sha256": old.sha(Path(__file__)), "train_sources": sources, "train_shards": hashes,
        "order_sha256": order_sha, "starting_actor_sha256": P["starting_actor"],
        "support_sha256": old.sha(HERE / "support.npz"), "train_roots": len(cash), "optimizer_roots": len(order),
        "valid_training_targets": mask.sum(0).tolist(), "previous_fit_reference": old.sha(AUX / "result.json")})
    print(products, flush=True)


if __name__ == "__main__":
    fit()
