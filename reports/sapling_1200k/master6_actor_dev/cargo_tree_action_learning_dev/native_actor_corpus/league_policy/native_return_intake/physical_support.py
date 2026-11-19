"""Compare the old and updated probabilities of actual full-team orders at training opportunities."""
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def order_key(order):
    # A missing unit and an explicit NONE both stay still. Every order is for one public state.
    return tuple(sorted((unit, move) for unit, move in order.items() if move != "NONE"))


def probabilities(orders, probs):
    if len(orders) != len(probs) or not math.isclose(sum(probs), 1, abs_tol=1e-12):
        raise ValueError("Invalid action probabilities")
    result = defaultdict(float)
    for order, prob in zip(orders, probs, strict=True):
        if not math.isfinite(prob) or prob <= 0:
            raise ValueError("Invalid supported action")
        result[order_key(order)] += prob
    return dict(result)


def metrics(event, row):
    old = probabilities(row["old"], row["q0"])
    new = probabilities(row["child"], row["q1"])
    replay = probabilities(event["options"], event["probabilities"])
    if old != replay or order_key(event["issued"]) not in old:
        raise ValueError("Old physical action distribution disagrees with the game")
    union = old.keys() | new.keys()
    tv = sum(abs(old.get(k, 0) - new.get(k, 0)) for k in union) / 2
    lost = sum(p for k, p in old.items() if k not in new)
    kl = None if lost else sum(p * math.log(p / new[k]) for k, p in old.items())
    return tv, lost, kl, order_key(event["issued"]) not in new


def summarize(values):
    finite = [v[2] for v in values if v[2] is not None]
    return {
        "opportunities": len(values),
        "mean_total_variation": sum(v[0] for v in values) / len(values),
        "mean_old_mass_no_longer_reachable": sum(v[1] for v in values) / len(values),
        "infinite_forward_kl_opportunities": len(values) - len(finite),
        "mean_physical_forward_kl": sum(finite) / len(values) if len(finite) == len(values) else None,
        "logged_orders_now_unreachable": sum(v[3] for v in values),
    }


def run():
    manifest_path = HERE / "hosted/training_manifest.json"
    traces_path = HERE / "fit_decisions.json.gz"
    fit = read(HERE / "fit.json")
    manifest = read(manifest_path)
    if digest(traces_path) != fit["decision_audit_sha256"] or digest(manifest_path) != fit["archive_manifest_sha256"]:
        raise ValueError("Wrong training data or action audit")
    parent = {(ep["seed"], ep["seat"], ep["arm"], e["step"]): e
              for ep in manifest["episodes"] for e in ep["events"]}
    curves = {}
    for step, trace in read(traces_path).items():
        observed = {}
        for row in trace["actual_menus"]:
            key = (row["seed"], row["seat"], row["arm"], row["step"])
            if key in observed:
                raise ValueError("Duplicate decision")
            observed[key] = metrics(parent[key], row)
        if observed.keys() != parent.keys():
            raise ValueError("Decision census differs")
        curves[step] = {
            "all": summarize(observed.values()),
            "by_seat": {str(s): summarize([v for k, v in observed.items() if k[1] == s]) for s in (0, 1)},
        }
    result = {
        "parent_policy_sha256": fit["parent_sha256"],
        "selected_policy_sha256": fit.get("selected_package_sha256"),
        "fit_sha256": digest(HERE / "fit.json"),
        "scope": "Same public training states, scheduled decisions only. Measures redistribution across actual complete commands. Does not estimate wins or evaluate off-schedule deterministic changes. A null mean KL means at least one positive-mass old order became unreachable.",
        "by_registered_step": curves,
    }
    path = HERE / "physical_support.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


if __name__ == "__main__":
    run()
