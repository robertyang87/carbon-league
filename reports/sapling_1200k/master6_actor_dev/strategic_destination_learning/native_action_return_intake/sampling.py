"""Choose one counterfactual per root with known probabilities for full-distribution gain estimates."""
from math import isclose, isfinite


def physical_order(command):
    return tuple(sorted((unit, move) for unit, move in command.items() if move != "NONE"))


def alternative_probabilities(menu, parent):
    """Return (option, draw probability) for each distinct, non-parent physical order."""
    seen = set()
    found_parent = False
    alternatives = []
    for option in menu:
        key = physical_order(option["action"])
        if key in seen:
            raise ValueError("Aggregate paths to the same physical order before sampling")
        seen.add(key)
        masses = (option["probability"], option["prior_probability"])
        if any(not isfinite(v) or v < 0 for v in masses):
            raise ValueError("Invalid policy mass")
        if key == physical_order(parent):
            found_parent = True
        elif sum(masses) > 0:
            alternatives.append((option, sum(masses)))
    if not found_parent or any(not isclose(sum(o[k] for o in menu), 1.0, abs_tol=1e-10)
                               for k in ("probability", "prior_probability")):
        raise ValueError("Invalid parent or incomplete physical menu")
    mass = sum(v for _, v in alternatives)
    return [(o, v / mass) for o, v in alternatives]


def sample_alternative(menu, parent, rng):
    """Use an independently seeded RNG before observing returns; zero alternatives need no branch."""
    alternatives = alternative_probabilities(menu, parent)
    if not alternatives:
        return None
    draw = rng.random()
    cumulative = 0.0
    for option, probability in alternatives:
        cumulative += probability
        if draw < cumulative:
            break
    return {
        "action": option["action"], "inclusion_probability": probability, "uniform_draw": draw,
        "trained_mass": option["probability"], "prior_mass": option["prior_probability"],
    }


def estimate(option, inclusion_probability, realized_alternative_minus_parent):
    if not 0 < inclusion_probability <= 1 or not isfinite(realized_alternative_minus_parent):
        raise ValueError("Expected a sampled alternative and finite outcomes from both arms")
    unit = realized_alternative_minus_parent / inclusion_probability
    return {
        "trained_minus_parent": option["probability"] * unit,
        "trained_minus_prior": (option["probability"] - option["prior_probability"]) * unit,
    }
