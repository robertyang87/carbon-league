"""Verify TTA frame restoration and test signature coverage with Feat-produced inputs."""
import copy
from common import setup, fixture_weights, emit


def main():
    setup()
    import numpy as np
    import cnet2_search as S
    import cnet2_np as NP
    import cnet2_feat as F
    import eval_v53
    from fastsim import FastCarbon

    weights = fixture_weights()
    obs = FastCarbon(eval_v53.CFG).reset(seed=970500).obs(0)
    f = F.Feat(0)
    x = f.observe(obs)
    policy = S.CachePolicy(weights)
    delta = lambda a, b: max(float(np.abs(u - v).max()) for u, v in zip(a, b))
    NP.act_tta(policy, x, f.aux, teacher=1, k=2)
    previous = policy.last_enc
    restored = policy.enc_for(x, 1)
    expected = NP.Policy.encode(policy, x, 1)
    rows = [{"test": "tta_restore", "before_delta": delta(previous, expected),
             "after_delta": delta(restored, expected)}]

    # Constant total cargo and saturated map channels, but different per-worker cargo.
    # Both inputs come from Feat.observe, not direct mutation of the feature tensors.
    a = FastCarbon(eval_v53.CFG).reset(seed=970500).obs(0)
    a["step"] = 50
    a["players"][0][2] = {"c1": [65, 1000, "COLLECTOR"], "c2": [66, 800, "COLLECTOR"]}
    b = copy.deepcopy(a)
    b["players"][0][2]["c1"][1] = 1100
    b["players"][0][2]["c2"][1] = 700
    x, y = F.Feat(0).observe(a), F.Feat(0).observe(b)
    policy.encode(x, 1)
    returned = policy.enc_for(y, 1)
    expected = NP.Policy.encode(policy, y, 1)
    rows.append({"test": "signature_coverage", "same_sig": policy._sig(x, 1) == policy._sig(y, 1),
                 "differing_inputs": [k for k in x if isinstance(x[k], np.ndarray) and not np.array_equal(x[k], y[k])],
                 "wrong_encoding_delta": delta(returned, expected)})
    emit("cache_checks.jsonl", rows)


if __name__ == "__main__":
    main()
