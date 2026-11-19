"""Compare each net's logits under the action prefix actually taken by the hybrid."""
import json
from pathlib import Path
import random

import external_3s2 as exam
import numpy as np

ROOT = exam.ROOT
PACK = ROOT / "data/weights/sapling_1200k/hybrid_it50_v3.py"
MAP = 620098103
REFERENCE = ROOT / "code/policy_v0_1.py"


def test():
    module = exam.prior.shared.module(PACK, "parity_hybrid")
    opponent = exam.prior.shared.module(REFERENCE, "practice_rules").make_agent()
    sim = exam.FastCarbon(exam.CFG).reset(seed=MAP)
    random.seed(MAP)
    results = []
    for t in range(30):
        observation = sim.obs(0)
        actual = module.agent(observation, sim.config)
        if t == 0:
            f = module.sys.modules["cnet2_feat"].Feat(0)
        shadow = f.observe(observation)
        # Rebuild the final labels using the package's own action mapping, including seat and NONE.
        labels = f.labels(actual)
        expert, student = [net.act(shadow, f.aux, teacher=2, forced=labels)[1] for net in module._NETS]
        expected = np.argmax(0.75 * expert + 0.25 * student, axis=1)
        active = shadow["tok_mask"][:len(labels)].astype(bool)
        if not np.array_equal(labels[active], expected[active]):
            raise AssertionError(f"shared-prefix policy mismatch at t={t}")
        results.append(int(active.sum()))
        acts = [actual, opponent(sim.obs(1), sim.config)]
        sim.step(*acts)
        if any(exam._extinct(sim.get_state(), i) for i in (0, 1)):
            break
    return {"seed": MAP, "seat": 0, "practice_opponent": str(REFERENCE.relative_to(ROOT)),
            "roots": len(results), "matched_active_slot_argmaxes": sum(results),
            "pack_sha256": exam.prior.digest(PACK), "mismatches": 0}


if __name__ == "__main__":
    print(json.dumps(test(), indent=2))
