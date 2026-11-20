"""Score the two observed complete orders and bound the missing terms of the full policy."""
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
TRAIN_SEEDS = [621758540, 621758541, 621758542, 621758543]


def load(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def policy_estimates(ticket, arm_values):
    menus = ticket["menu"]
    kinds = ("probability", "prior_probability")
    for name in kinds:
        probs = [m[name] for m in menus]
        if any(not math.isfinite(v) or v < 0 for v in probs) or not math.isclose(sum(probs), 1.0):
            raise ValueError("Invalid registered action probabilities")
    observed = {}
    for arm, order in ticket["orders"].items():
        indices = [i for i, m in enumerate(menus) if m["action"] == order]
        if len(indices) != 1 or not 0 <= arm_values[arm] <= 1:
            raise ValueError("Unmatched physical action or invalid utility")
        observed[arm] = indices[0]
    if set(observed) != {"parent", "alternative"} or len(set(observed.values())) != 2:
        raise ValueError("Expected two distinct arms")
    unknown = [i for i in range(len(menus)) if i not in observed.values()]
    alternative, parent = (menus[observed[arm]] for arm in ("alternative", "parent"))
    def conditional_alt(name):
        return alternative[name] / (alternative[name] + parent[name])

    # A minus B cancels the contribution of unmeasured arms whenever their probabilities agree.
    differences = [m["probability"] - m["prior_probability"] for m in menus]
    measured = sum(differences[i] * arm_values[arm] for arm, i in observed.items())
    trained_known = sum(menus[i]["probability"] * arm_values[a] for a, i in observed.items())
    unseen = sum(menus[i]["probability"] for i in unknown)
    delta = arm_values["alternative"] - arm_values["parent"]
    return {
        "seed": ticket["seed"], "seat": ticket["seat"], "step": ticket["root_step"],
        "known_arms_U": arm_values, "realized_alternative_minus_parent_U": delta,
        "trained_unmeasured_mass": unseen, "prior_unmeasured_mass": sum(menus[i]["prior_probability"] for i in unknown),
        "trained_conditional_p_alternative": conditional_alt("probability"),
        "prior_conditional_p_alternative": conditional_alt("prior_probability"),
        "two_arm_conditional_trained_minus_prior_U": (conditional_alt("probability") - conditional_alt("prior_probability")) * delta,
        "full_trained_minus_prior_U_range": [
            measured + sum(min(differences[i], 0) for i in unknown),
            measured + sum(max(differences[i], 0) for i in unknown),
        ],
        "full_trained_minus_parent_U_range": [trained_known - arm_values["parent"], trained_known + unseen - arm_values["parent"]],
    }


def main():
    roots = load(HERE / "roots.json")["tickets"]
    if [r["seed"] for r in roots] != TRAIN_SEEDS:
        raise ValueError("Registered training map roster changed")
    records = []
    for ticket in roots:
        if not ticket["selected"]:
            raise ValueError("Unselected map has no two-arm evidence")
        directory = HERE / "tickets" / str(ticket["seed"])
        receipt = load(directory / "summary.json")
        if not receipt["root_and_suffix_replayed"] or receipt["original_opening_games_added"] != 0:
            raise ValueError("Expected a verified conditional comparison")
        values = {}
        for arm, item in receipt["arms"].items():
            if item["root_order"] != ticket["orders"][arm] or sha(directory / (arm + ".json.gz")) != item["sha256"]:
                raise ValueError("Unmatched root command or successor tape")
            values[arm] = item["value"]["U"]
        records.append(policy_estimates(ticket, values))
    bounds = ("full_trained_minus_prior_U_range", "full_trained_minus_parent_U_range")
    result = {
        "roots_sha256": sha(HERE / "roots.json"), "samples": records,
        "map_equal_realized_conditional_trained_minus_prior_U": mean(r["two_arm_conditional_trained_minus_prior_U"] for r in records),
        "map_equal_ranges_conditional_on_realized_values": {key: [mean(r[key][i] for r in records) for i in (0, 1)] for key in bounds},
        "future_opponent_sampling_uncertainty_included": False,
    }
    output = HERE / "policy_values.json"
    content = (json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()
    if output.exists() and output.read_bytes() != content:
        raise ValueError("Previous policy-value evidence differs")
    if not output.exists():
        path = output.with_suffix(".tmp")
        path.write_bytes(content)
        path.replace(output)


if __name__ == "__main__":
    main()
