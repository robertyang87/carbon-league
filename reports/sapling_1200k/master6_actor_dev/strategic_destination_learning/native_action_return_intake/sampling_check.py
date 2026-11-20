"""Enumerate every possible draw on existing original-opponent roots and compare with all-arm truth."""
import gzip
import hashlib
import json
import math
from pathlib import Path
from sampling import alternative_probabilities, estimate

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "native_action_policy_economy/steps.json.gz"


def main():
    data = SOURCE.read_bytes()
    metrics = ("bank", "cargo", "intercept", "trees")
    arms = ("trained_minus_parent", "trained_minus_prior")
    maps = []
    total_options = 0
    total_roots = 0
    maximum_error = 0.0
    for game in json.loads(gzip.decompress(data)):
        roots = [r for r in game["rows"] if "parent_outcome" in r]
        report = {"seed": game["seed"], "seat": game["seat"], "decisions": len(roots)}
        variance = {k: {a: 0.0 for a in arms} for k in metrics}
        for root in roots:
            alternatives = alternative_probabilities(root["options"], root["parent_order"])
            total_options += len(root["options"])
            total_roots += 1
            for metric in metrics:
                parent = root["parent_outcome"]["domain"][metric]
                true = {"trained_minus_parent": root["expected_delta"][metric],
                        "trained_minus_prior": root["learned_minus_prior"][metric]}
                draws = [(w, estimate(o, w, o["outcome"]["domain"][metric] - parent))
                         for o, w in alternatives]
                for contrast in arms:
                    expectation = sum(w * e[contrast] for w, e in draws)
                    err = abs(expectation - true[contrast])
                    maximum_error = max(maximum_error, err)
                    if err > 1e-8:
                        raise ValueError("One-pair estimator disagrees with the all-arm record")
                    variance[metric][contrast] += sum(w * (e[contrast] - expectation) ** 2 for w, e in draws)
        report["conditional_draw_sd_per_decision_mean"] = {
            k: {a: math.sqrt(v) / len(roots) for a, v in values.items()} for k, values in variance.items()
        }
        maps.append(report)
    result = {"source_sha256": hashlib.sha256(data).hexdigest(), "roots": total_roots,
              "full_physical_choices_including_parent": total_options, "one_pair_evaluations": 2 * total_roots,
              "max_exact_bias_on_recorded_outcomes": maximum_error, "maps": maps,
              "full_matches_added": 0, "terminal_continuations_added": 0}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
