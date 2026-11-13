"""diag_sample -- does probability-sampling inference (instead of
argmax) unlock the shadow's closed-loop game vs example?

Reuses shadow_alpha's parser/featurizer/forward verbatim; only the
action selection changes: workers sample from softmax(logits / T),
bases keep argmax. RNG is seeded per agent instance, so a game is
reproducible. Run: python3 diag_sample.py [temperature]
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "env_src", "carbon_challenge-master"))

import numpy as np  # noqa: E402

from fastsim import FastCarbon  # noqa: E402
from eval_v0_5 import CFG  # noqa: E402
import shadow_alpha as S  # noqa: E402
import importlib.util  # noqa: E402

TEMP = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
MOVE = {1: "UP", 2: "RIGHT", 3: "DOWN", 4: "LEFT"}
REC = {1: "RECCOLLECTOR", 2: "RECPLANTER"}


def make_sampling_agent(seed=12345):
    rng = np.random.default_rng(seed)
    state = S._SeqState()

    def act(observation, configuration):
        (p, carbon, cash, bases, workers, wmask, trees, tmask, step,
         my_wids, my_bids) = S._canonical(observation)
        n_now = int(np.sum(wmask[p]))
        seq = state.seq_for(n_now, step)
        W, B = S.featurize(carbon, cash, bases, workers, wmask, trees,
                           tmask, p, step, seq=seq)
        actions = {}
        if my_wids:
            logits = S._forward("w_", W[:len(my_wids)]) / TEMP
            logits = logits - logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            for j, wid in enumerate(my_wids):
                a = int(rng.choice(5, p=probs[j]))
                if a in MOVE:
                    actions[wid] = MOVE[a]
        n_rec = 0
        if my_bids:
            blogits = S._forward("b_", B[:len(my_bids)])
            for i, bid in enumerate(my_bids):
                a = int(np.argmax(blogits[i]))
                if a in REC:
                    actions[bid] = REC[a]
                    n_rec += 1
        state.update(n_now, step, n_rec)
        return actions

    return act


def load_example():
    spec = importlib.util.spec_from_file_location(
        "exmod", os.path.join(HERE, "example.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.agent


def main():
    ex = load_example()
    for side in (0, 1):
        sim = FastCarbon(CFG).reset(seed=2000)
        shadow = make_sampling_agent()
        agents = [shadow, ex] if side == 0 else [ex, shadow]
        dep = 0.0
        prev_cash = 80.0
        for t in range(300):
            acts = [agents[0](sim.obs(0), CFG), agents[1](sim.obs(1), CFG)]
            sim.step(acts[0], acts[1])
        st = sim.get_state()
        ws = [w for w in st["workers"].values() if w["player"] == side]
        ts = [t for t in st["trees"].values() if t["player"] == side]
        print(f"T={TEMP} side {side}: shadow cash {st['cash'][side]:.1f} "
              f"vs example {st['cash'][1 - side]:.1f}; end workers "
              f"{len(ws)} trees {len(ts)} "
              f"cargo {sum(w['carbon'] for w in ws):.0f}")


if __name__ == "__main__":
    main()
