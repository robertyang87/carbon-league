"""gen_rl_policy -- generate rl_policy_v1.py from the shadow_alpha.py
source (identical featurization + inference code, guaranteeing the
evaluated artifact is byte-for-byte the inference path the BC/RL
weights were validated with), repointed at rl_weights.npz (the best
PPO checkpoint selected by train_rl.py) with an RL header.

Run after train_rl.py:  python3 gen_rl_policy.py
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))

HEADER = '''"""rl_policy_v1 -- RL policy rl-v0.1 (Carbon League NN line, stage 1).

PPO (clipped surrogate + GAE) warm-started from the BC Shadow Alpha
weights and self-played against a pool of {frozen BC shadow,
self-snapshots, policy_v0_5, example} on FastCarbon; juzi was never
in training (eval-only). Selected checkpoint: best eval margin vs
v0.5 (+ 0.5 x vs shadow) over the run -- see rl_stage1_report.md and
rl_train_log.jsonl.

Inference protocol is shadow_alpha's, unchanged: worker actions
sampled from the learned distribution at T=1 with a fixed per-instance
RNG seed (argmax collapses in closed loop), base actions argmax.
Features use ONLY live-legal information (own visibility-filtered
carbon + the shared players block) -- bc_features source embedded
verbatim below. Weights: rl_weights.npz in this directory.
"""

'''


def main():
    src = open(os.path.join(HERE, "shadow_alpha.py")).read()
    # strip shadow's module docstring (first triple-quoted block)
    end = src.index('"""', src.index('"""') + 3) + 3
    body = src[end:].lstrip("\n")
    body = body.replace("shadow_weights.npz", "rl_weights.npz")
    body = body.replace("Shadow Alpha agent", "rl-v0.1 agent")
    out = HEADER + body
    path = os.path.join(HERE, "rl_policy_v1.py")
    with open(path, "w") as f:
        f.write(out)
    print(f"wrote {path} ({len(out)} bytes)")


if __name__ == "__main__":
    main()
