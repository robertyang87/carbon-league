"""Check the weighted numpy sampler contract without running the examiner."""
import json
import sys
from pathlib import Path

import numpy as np


def same_state(a, b):
    aa, bb = a.get_state(), b.get_state()
    return aa[0] == bb[0] and np.array_equal(aa[1], bb[1]) and aa[2:] == bb[2:]


def main(out):
    if np.__version__ != "1.26.4":
        raise ValueError("Use the registered ladder NumPy version")
    patterns = [
        (2, 0), (2, 1), (2, 2), (2, 10), (1, 0), (1, 10), (0, 2),
        (0, 0), (2, 4), (2, 3), (2, 9), (2, 6),
    ]
    weights = {
        3: [[1, 0, 0], [.2, .3, .5], [0, .5, .5]],
        5: [[0, 0, 1, 0, 0], [.11, .13, .17, .28, .31], [.5, 0, .5, 0, 0]],
    }
    transitions = choices = 0
    for seed in (0, 17, 105, 12991):
        for keep_gaussian in (False, True):
            ref = np.random.RandomState(seed)
            if keep_gaussian:
                ref.standard_normal()
            fast = np.random.RandomState(0)
            fast.set_state(ref.get_state())
            for i in range(120):
                bases, workers = patterns[i % len(patterns)]
                for j, width in enumerate([3] * bases + [5] * workers):
                    ref.choice(width, p=weights[width][(i+j) % 3])
                fast.random_sample(bases + workers)
                if not same_state(ref, fast):
                    raise AssertionError("Different future draw stream")
                transitions += 1
                choices += bases + workers
    out.write_text(json.dumps({"numpy": np.__version__, "seeds":4, "with_and_without_cached_gaussian":True,
                               "verified_simulated_turns":transitions, "verified_weighted_choices":choices,
                               "public_examiner_calls":0}, indent=2)+"\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
