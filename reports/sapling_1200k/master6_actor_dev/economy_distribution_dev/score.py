"""Score public decisions, cash forecasts and fixed alert costs on locked ladders."""
import gzip
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from fit_distribution import HERE, AUX, TARGETS, base, old, read, corpus

sys.path.insert(0, str(old.ACTOR))
import data_build

FRESH_CACHE = Path("/workspace/sapling-cash-distribution-panel-v1")
TRANSFER_CACHE = Path("/workspace/sapling-income-transfer-public-v1")


def load_model(directory, packet, arm):
    ids = packet["weights"][arm]
    net, _ = base.model(directory / (arm + ".npz"), ids["actor_sha256"])
    hp = directory / (arm + "_readout.npz")
    if old.sha(hp) != ids["readout_sha256"]:
        raise ValueError("head mismatched with registered encoder")
    with np.load(hp) as z:
        head = {k: torch.tensor(z[k]) for k in z.files}
    return net.eval(), base.quantized(net).eval(), head


@torch.no_grad()
def forward(models, data, support):
    means = torch.tensor(support["means"], dtype=torch.float32)
    valid = torch.from_numpy(support["counts"] > 0)
    alert = torch.tensor(support["within_alert"], dtype=torch.float32)
    out = {}
    for name, (enc, deploy, head) in models.items():
        lead = []
        guesses = []
        nll = []
        risk = []
        for lo in range(0, len(data["y"]), 32):
            ix = np.arange(lo, min(lo + 32, len(data["y"])))
            b = base.batch(data, ix)
            public = {k: v for k, v in b.items() if k not in ("value", "carbon")}
            _, ctx, _ = enc.encode(public)
            lg = deploy(public)[0]
            if lo == 0:
                altered = dict(public, y=torch.zeros_like(public["y"]))
                if not torch.equal(ctx, enc.encode(altered)[1]) or not torch.equal(ctx, enc.encode(b)[1]):
                    raise ValueError("cash head uses future or privileged features")
                for slot in range(public["y"].shape[1]):
                    altered["y"] = public["y"].clone()
                    altered["y"][:, slot:] = 0
                    if not torch.equal(deploy(altered)[0][:, slot], lg[:, slot]):
                        raise ValueError("action depends on a current/future logged label")
            z = F.linear(ctx, head["weight"], head["bias"])
            if name.startswith("log"):
                cash = torch.expm1((z * head["scale"] + head["center"]).clamp(min=0))
            else:
                prob = z.reshape(-1, 8, 9).masked_fill(~valid, -1e9).softmax(-1)
                cash = (prob * means).sum(-1)
                risk.extend((prob * alert).sum(-1).numpy())
            lead.extend(cash.numpy())
            guesses.extend(lg.argmax(-1).numpy())
            nll.extend(old.chosen_nll(lg.softmax(-1).numpy(), data["y"][ix]))
        out[name] = (np.asarray(lead), np.asarray(guesses), np.asarray(nll), np.asarray(risk))
    return out


def agg(v):
    return float(np.mean(v)) if len(v) else None


def stats(y, pred, threshold, cap, max_train):
    err = pred - y
    gold_alert = y >= 500
    issued = pred >= 500
    false = issued & ~gold_alert
    missed = ~issued & gold_alert
    tail = y >= threshold
    beyond = y > cap + .001
    return {"n": len(err), "mean_actual": agg(y), "mean_pred": agg(pred), "bias": agg(err),
            "mae": agg(abs(err)), "max_forecast": float(pred.max()) if len(pred) else None,
            "tail_n": int(tail.sum()), "tail_mae": agg(abs(err[tail])), "tail_bias": agg(err[tail]),
            "above_largest_support_n": int(beyond.sum()), "above_largest_support_bias": agg(err[beyond]),
            "above_observed_training_max_n": int(np.sum(y > max_train + .001)),
            "fp": int(false.sum()), "fn": int(missed.sum()), "flagged": int(issued.sum()),
            "cost_per_root": agg(false.astype(float) + 2 * missed)}


def paired(values, seed=231119):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return None
    r = np.random.default_rng(seed)
    boot = a[r.integers(len(a), size=(5000, len(a)))].mean(1)
    return {"maps": len(a), "mean": agg(a), "positive": int((a > 0).sum()),
            "bootstrap95": np.quantile(boot, [.025, .975]).tolist()}


def collector_counts(cards, d):
    n = np.full(len(d["gid"]), -1, int)
    for card in cards:
        ix = np.flatnonzero(d["gid"] == card["gid"])
        if not len(ix):
            continue
        raw = (old.CACHE / (card["id"] + ".json.gz")).read_bytes()
        if base.digest(raw) != card["sha256"]:
            raise ValueError("wrong replay for workload grouping")
        frames = json.loads(gzip.decompress(raw))["steps"]
        for i in ix:
            f = frames[d["step"][i]][0]["observation"]
            n[i] = sum(w[2] == "COLLECTOR" for w in f["players"][1-card["seat"]][2].values())
    if (n < 0).any():
        raise ValueError("unassigned map")
    return n


def summarize(cards, data, gold, mask, out, support):
    names = list(out)
    gids = data["gid"]
    eligible = base.actionable(data)
    rival_count = collector_counts(cards, data)
    lookup = {card["gid"]: card for card in cards}
    seen = {g["opponent"] for g in corpus()[0] if g["split"] == "train"}
    def stratify(key):
        return np.asarray([lookup[int(g)][key] for g in gids])
    outcomes = np.array([np.sign(lookup[int(g)]["margin"]) for g in gids])
    strata = {"all": np.ones(len(gids), bool), "seat0": stratify("seat") == 0, "seat1": stratify("seat") == 1,
              "known_opponent_registration": np.isin(stratify("opponent"), list(seen)),
              "unseen_opponent_registration": ~np.isin(stratify("opponent"), list(seen)),
              "expert_won": outcomes > 0, "expert_lost": outcomes < 0, "expert_drew": outcomes == 0,
              "three_or_more_visible_rival_collectors": rival_count >= 3}
    results = {}
    metric = lambda y, p, j: stats(y, p, support["q95"][j], support["means"][j].max(), support["maximum"][j])
    for group, use in strata.items():
        if not use.any():
            continue
        r = results[group] = {"eligible_roots": int(use.sum()), "maps": int(len(np.unique(gids[use]))), "scores": {}}
        for name, (cash, guess, nll, risk) in out.items():
            valid = eligible & use[:, None]
            rootmask = mask[:, 6] & use
            rr = r["scores"][name] = {"action_slots": int(valid.sum()), "action_nll": agg(nll[valid]),
                "action_agreement": agg((guess == data["y"])[valid]), "rival_delivery_30_pooled":
                metric(gold[rootmask, 6], cash[rootmask, 6], 6)}
            if len(risk):
                rr["alert_brier"] = agg((risk[rootmask,6] - (gold[rootmask,6] >= 500)) ** 2)
            by_map = {}
            for gid in np.unique(gids[rootmask]):
                m = rootmask & (gids == gid)
                by_map[str(gid)] = metric(gold[m,6], cash[m,6], 6)
            rr["map_equal_rival_30"] = {k: agg([v[k] for v in by_map.values()]) for k in
                ("mean_actual","mean_pred","mae","bias","cost_per_root")}
            if group == "all":
                rr["by_map"] = by_map
                rr["cash_all_channels"] = {}
                for j, target in enumerate(TARGETS):
                    m = mask[:,j] & use
                    rr["cash_all_channels"][target] = metric(gold[m,j], cash[m,j], j)
        if group == "all":
            compare = {}
            for baseline in ("log_candidate","dist_control"):
                diff = {}
                for k in ("mae","bias","cost_per_root"):
                    d = r["scores"]["dist_candidate"]["by_map"]
                    c = r["scores"][baseline]["by_map"]
                    diff[k] = paired([d[m][k]-c[m][k] for m in d])
                compare[baseline] = diff
            r["paired_dist_candidate_minus_baseline_map_bootstrap"] = compare
            r["q8_action_changes_dist_joint_vs_dist_detach"] = agg(
                (out["dist_candidate"][1] != out["dist_control"][1])[eligible])
    indices = np.flatnonzero(mask[:, 6])
    idx = indices[np.argmax(abs(out["log_candidate"][0][indices,6] - gold[indices,6]))]
    card = lookup[int(gids[idx])]
    results["largest_old_rival_error"] = {"game_id": card["id"], "opponent": card["opponent"], "seat": card["seat"],
        "step":int(data["step"][idx]), "public_rival_collectors":int(rival_count[idx]),
        "actual_cash_next_30": float(gold[idx, 6]),
        "forecasts": {k:float(v[0][idx,6]) for k,v in out.items()}}
    return results, rival_count


def run(panel):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    reference = read(AUX / "result.json")
    fit = read(HERE / "fit_receipt.json")
    models = {prefix + "_" + arm: load_model(directory, packet, arm)
              for prefix, directory, packet in [("log", AUX, reference), ("dist", HERE, fit)]
              for arm in ("control","candidate")}
    support_file = HERE / "support.npz"
    if old.sha(support_file) != fit["support_sha256"]:
        raise ValueError("training support mismatch")
    with np.load(support_file) as z:
        support = {k: z[k] for k in z.files}
    if panel == "holdout":
        all_cards, receipts = corpus()
        cards = [c for c in all_cards if c["split"] == panel]
        storage, part = old.SHARDS, panel
    else:
        is_fresh = panel == "fresh"
        path = HERE / "fresh_panel.json" if is_fresh else AUX / "ladder_transfer" / "selection.json"
        cards = read(path)["cards"]
        storage = FRESH_CACHE if is_fresh else TRANSFER_CACHE
        storage.mkdir(exist_ok=True)
        receipts = {c["id"]: data_build.prep(c, storage)["sha256"] for c in cards}
        part = cards[0]["split"]
    data, hashes = base.public(part, cards, storage, receipts)
    y, mask, sources = old.targets(cards, data)
    if set(sources) != {c["id"] for c in cards}:
        raise ValueError("selected map missing public roots")
    gold = np.expm1(y.astype(np.float64))
    out = forward(models, data, support)
    results, counts = summarize(cards, data, gold, mask, out, support)
    trace = HERE / (panel + "_predictions.npz")
    vals = dict(gid=data["gid"], step=data["step"], cash=gold, full_horizon=mask,
                collector_count=counts, legal_choice=base.actionable(data), action=data["y"])
    for name, (cash, action, nll, alert) in out.items():
        vals.update({name+"_cash": cash, name+"_action": action, name+"_nll": nll})
        if len(alert):
            vals[name+"_p500"] = alert
    np.savez_compressed(trace, **vals)
    packet = {"panel":panel,"fit_receipt_sha256":old.sha(HERE / "fit_receipt.json"),
              "implementation_sha256":old.sha(Path(__file__)),"by_stratum": results,
              "source_replays":sources,"source_shards":hashes,"predictions_sha256":old.sha(trace),
              "cards":[{k:c.get(k) for k in ("id","gid","seed","terrain_sha256","seat","opponent","margin")} for c in cards]}
    base.write_json(HERE / (panel + "_result.json"), packet)
    print(panel,len(cards),len(gold),results["all"]["scores"]["dist_candidate"]["map_equal_rival_30"],flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
