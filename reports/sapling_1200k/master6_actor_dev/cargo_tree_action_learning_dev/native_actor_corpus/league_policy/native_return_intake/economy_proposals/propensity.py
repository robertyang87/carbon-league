"""Score a physical team order on the fixed behavior menu.

This is the factor through which a later learner can update native logits.
It accepts no label or return. Backpropagate each reported d(log p)/d(gap)
through gap = native_alternative_logit - native_baseline_logit. If refitting
changes the membership or the native decoder route, these fixed-menu
probabilities describe a different (conditional) policy than a rebuilt menu.
"""
import math

from proposals import CATEGORIES, CLASS_PROBABILITY, alternative_probability, command_key


def marginal(menu, command, gap_updates=None):
    """Calculate p(command) and its score by unique native (slot, best, other).

    gap_updates optionally maps a witness to a *new* logit difference.
    Multiple paths to the same physical order contribute to the same score.
    A zero-support physical command has p=0 and undefined score (None).
    """
    target = command_key(command)
    baseline = command_key(menu["baseline"])
    mass = 0.0
    derivatives = {}
    for category in CATEGORIES:
        witnesses = menu["classes"][category]["witnesses"]
        if not witnesses and baseline == target:
            mass += CLASS_PROBABILITY
        for w in witnesses:
            is_alternative = command_key(w["command"]) == target
            if not is_alternative and target != baseline:
                continue
            witness_key = (w["slot"], *w["actions"])
            gap = w["logits"][1] - w["logits"][0]
            if gap_updates is not None:
                gap = gap_updates.get(witness_key, gap)
            if not math.isfinite(gap):
                raise ValueError("nonfinite native logit difference")
            alt_prob = alternative_probability(0.0, gap)
            # alt_prob = .25 + .5 * sigmoid(gap).
            sigmoid = (alt_prob - .25) * 2
            weight = CLASS_PROBABILITY / len(witnesses)
            mass += weight * (alt_prob if is_alternative else 1.0 - alt_prob)
            contribution = weight * .5 * sigmoid * (1.0 - sigmoid)
            if not is_alternative:
                contribution = -contribution
            derivatives[witness_key] = derivatives.get(witness_key, 0) + contribution
    return mass, ({key: derivative / mass for key, derivative in derivatives.items()} if mass else None)


def audit(path):
    """Recompute signed order masses and check score derivatives on real menus."""
    import gzip
    import hashlib
    import json

    raw = path.read_bytes()
    tickets = json.loads(gzip.decompress(raw))["tickets"]
    largest_mass_error = largest_derivative_error = 0.0
    orders = coordinates = 0
    perturbation = 1e-5
    for ticket in tickets:
        for row in ticket["rows"]:
            menu = row["catalog"]
            gaps = {(w["slot"], *w["actions"]): w["logits"][1] - w["logits"][0]
                    for category in menu["classes"].values() for w in category["witnesses"]}
            score_sum = dict.fromkeys(gaps, 0.0)
            for item in menu["orders"]:
                command = item["command"]
                p, score = marginal(menu, command)
                if p <= 0 or score is None:
                    raise AssertionError("supported order must have a finite log probability")
                largest_mass_error = max(largest_mass_error, abs(p - item["probability"]))
                for witness, gap in gaps.items():
                    plus, _ = marginal(menu, command, {witness: gap + perturbation})
                    minus, _ = marginal(menu, command, {witness: gap - perturbation})
                    numerical = (math.log(plus) - math.log(minus)) / (2 * perturbation)
                    analytic = score.get(witness, 0.0)
                    largest_derivative_error = max(largest_derivative_error, abs(numerical - analytic))
                    score_sum[witness] += p * analytic
                    coordinates += 1
                orders += 1
            if any(abs(v) > 1e-12 for v in score_sum.values()):
                raise AssertionError("expected policy score must vanish across possible orders")
    if largest_mass_error > 1e-12 or largest_derivative_error > 1e-8:
        raise AssertionError("physical policy probability/gradient mismatch")
    return {"roots_sha256": hashlib.sha256(raw).hexdigest(), "orders": orders,
            "finite_difference_coordinates": coordinates,
            "max_abs_marginal_probability_error": largest_mass_error,
            "max_abs_log_probability_derivative_error": largest_derivative_error}


if __name__ == "__main__":
    import json
    from pathlib import Path

    print(json.dumps(audit(Path(__file__).with_name("roots.json.gz")), indent=2))
