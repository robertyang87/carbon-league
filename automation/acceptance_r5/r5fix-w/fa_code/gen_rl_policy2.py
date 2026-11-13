"""gen_rl_policy2 -- assemble the deployable rl_policy_v2.py (and its
shield-ablation variants) from the exact sources the RL line trains
with, so the evaluated artifact cannot drift from the training path:

  header + bc_features.py (verbatim)
         + belief.py (verbatim minus its bc_features import line)
         + rl2_common.py ACTOR SECTION (verbatim between markers)
         + weights glue (loads rl2_weights.npz next to the file)

Variants: rl_policy_v2_noopen.py (opening shield off) and
rl_policy_v2_noend.py (endgame shield off), for the stage-2 shield
ablations; they differ only in the SHIELDS_DEFAULT literal.

Run after train_rl2.py:  python3 gen_rl_policy2.py
"""

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

HEADER = '''"""rl_policy_v2 -- RL policy rl-v0.2 (Carbon League NN line,
stage 2).

PPO (stage-1 recipe: clipped surrogate + GAE, per-unit baseline
attribution rewards) warm-started from a stage-2 BC trained on the
53-game Claude-family corpus with belief-augmented features
(four-fold-symmetric belief carbon map, juzi-style reconstruction --
see belief.py docstring), self-played against {shadow v2,
self-snapshots, policy_v0_5, example}; juzi was never in training
(probe/final-eval only). Checkpoint selection: fixed probe set every
iteration (see rl_stage2_report.md). Inference: worker actions are
sampled (T=1) through the decorrelated sequential sampler with the
three hard-rule shields active (danger mask / endgame return /
opening cash-chain protection); base actions argmax with the opening
shield's overrides. Weights: rl2_weights.npz in this directory.
"""

import os as _os
import re as _re

import numpy as _np
np = _np

'''

GLUE = '''

# ----------------------------------------------------------------------
# weights + agent factory (stage-2 glue). _suf/_canonical are
# shadow_alpha.py's, verbatim -- the same parsing the BC extraction
# and stage-1/2 training used.
# ----------------------------------------------------------------------
__CANONICAL_SRC__

_W2 = None


def _load_w2():
    global _W2
    if _W2 is None:
        z = np.load(_os.path.join(
            _os.path.dirname(_os.path.abspath(__file__)),
            "rl2_weights.npz"))
        _W2 = {k: z[k] for k in z.files}
    return _W2


SHIELDS_DEFAULT = __SHIELDS__


def make_agent():
    actor = NumpyActor2(_load_w2(), shields=dict(SHIELDS_DEFAULT))
    fn = lambda observation, configuration: actor(observation,
                                                  configuration)
    fn.actor = actor
    return fn


_GLOBAL_ACTOR = None


def agent(observation, configuration):
    global _GLOBAL_ACTOR
    if _GLOBAL_ACTOR is None:
        _GLOBAL_ACTOR = NumpyActor2(_load_w2(),
                                    shields=dict(SHIELDS_DEFAULT))
    return _GLOBAL_ACTOR(observation, configuration)
'''


def _strip_docstring_and_imports(src, drop_lines=()):
    end = src.index('"""', src.index('"""') + 3) + 3
    body = src[end:].lstrip("\n")
    lines = [ln for ln in body.split("\n")
             if not any(ln.startswith(d) for d in drop_lines)]
    return "\n".join(lines)


def build(shields, out_name):
    bc = open(os.path.join(HERE, "bc_features.py")).read()
    bc_body = _strip_docstring_and_imports(bc)
    bel = open(os.path.join(HERE, "belief.py")).read()
    bel_body = _strip_docstring_and_imports(bel)
    bel_body = re.sub(r"from bc_features import \([^)]*\)\n", "",
                      bel_body, count=1)
    rl2 = open(os.path.join(HERE, "rl2_common.py")).read()
    b = rl2.index("# === DEPLOYABLE ACTOR BEGIN ===")
    e = rl2.index("# === DEPLOYABLE ACTOR END ===")
    actor = rl2[b:e].split("\n", 1)[1]
    sh = open(os.path.join(HERE, "shadow_alpha.py")).read()
    i0 = sh.index("_SUFFIX = ")
    i1 = sh.index("_MOVE = {")
    canonical_src = sh[i0:i1].rstrip() + "\n"
    glue = GLUE.replace("__SHIELDS__", repr(shields)).replace(
        "__CANONICAL_SRC__", canonical_src)
    out = (HEADER + "\n# ===== bc_features.py (verbatim) =====\n"
           + bc_body
           + "\n# ===== belief.py (verbatim) =====\n" + bel_body
           + "\n# ===== rl2_common.py actor section (verbatim) =====\n"
           + actor + glue)
    path = os.path.join(HERE, out_name)
    with open(path, "w") as f:
        f.write(out)
    print(f"wrote {path} ({len(out)} bytes)")


def main():
    full = {"danger": True, "endgame": True, "opening": True}
    build(full, "rl_policy_v2.py")
    build({**full, "opening": False}, "rl_policy_v2_noopen.py")
    build({**full, "endgame": False}, "rl_policy_v2_noend.py")


if __name__ == "__main__":
    main()
