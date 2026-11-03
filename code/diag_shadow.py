"""diag_shadow -- localize the shadow's live-play failure.

A) npz val samples -> bc_features.featurize -> shadow numpy forward:
   accuracy must reproduce the torch val metrics (~worker 0.554).
B) Same samples round-tripped through shadow_alpha._canonical (obs
   dicts synthesized from the npz arrays) -> forward: must match A
   exactly if the live parser is faithful.
C) One live game vs example with a coarse behavior trace.

Run: venv/bin/python diag_shadow.py
"""

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from bc_features import featurize  # noqa: E402
from train_bc import VAL_FILES  # noqa: E402
import shadow_alpha as S  # noqa: E402


def build_rows(d, use_canonical):
    n = len(d["step"])
    is_val = np.array([f in VAL_FILES for f in d["game_file"]])
    rows = []
    seq_game, seq_prev_step, seq_prev_n = None, None, 0
    seq_prev_rec, seq_last_rec_step = 0, None
    for i in range(n):
        if not is_val[i]:
            # still advance seq state so val rows see the same history
            p0 = int(d["player"][i])
            n_now0 = int(np.sum(d["workers_mask"][i][p0]))
            if d["game_file"][i] != seq_game or \
                    int(d["step"][i]) != (seq_prev_step or -1) + 1:
                seq_last_rec_step = None
            seq_game = d["game_file"][i]
            seq_prev_step = int(d["step"][i])
            seq_prev_n = n_now0
            seq_prev_rec = int(np.sum(d["base_actions"][i] > 0))
            if seq_prev_rec > 0:
                seq_last_rec_step = int(d["step"][i])
            continue
        p = int(d["player"][i])
        step_i = int(d["step"][i])
        n_now = int(np.sum(d["workers_mask"][i][p]))
        if d["game_file"][i] == seq_game and step_i == seq_prev_step + 1:
            lost = max(0, seq_prev_n + seq_prev_rec - n_now)
            since = 60 if seq_last_rec_step is None \
                else step_i - seq_last_rec_step
            seq = {"prev_n": float(seq_prev_n), "lost": float(lost),
                   "since": float(since)}
        else:
            seq = {"prev_n": float(n_now), "lost": 0.0, "since": 60.0}
            seq_last_rec_step = None
        if use_canonical:
            obs = synth_obs(d, i)
            (pp, carbon, cash, bases, workers, wmask, trees, tmask,
             step2, _w, _b) = S._canonical(obs)
            assert pp == p and step2 == step_i
            W, B = featurize(carbon, cash, bases, workers, wmask, trees,
                             tmask, pp, step2, seq=seq)
        else:
            W, B = featurize(d["carbon"][i], d["cash"][i], d["bases"][i],
                             d["workers"][i], d["workers_mask"][i],
                             d["trees"][i], d["trees_mask"][i],
                             p, step_i, seq=seq)
        rows.append((W, B, d["workers_mask"][i][p],
                     d["worker_actions"][i], d["base_actions"][i],
                     d["workers"][i][p][:, 2]))
        seq_game = d["game_file"][i]
        seq_prev_step = step_i
        seq_prev_n = n_now
        seq_prev_rec = int(np.sum(d["base_actions"][i] > 0))
        if seq_prev_rec > 0:
            seq_last_rec_step = step_i
    return rows


def synth_obs(d, i):
    p = int(d["player"][i])
    players = []
    for q in range(2):
        centers = {}
        for j in range(2):
            if int(d["bases"][i][q][j]) >= 0:
                centers[f"player-{q}-center-{j}"] = int(d["bases"][i][q][j])
        ws = {}
        for j in range(10):
            if d["workers_mask"][i][q][j]:
                pos, cargo, typ = d["workers"][i][q][j]
                ws[f"player-{q}-worker-{j}"] = [
                    int(pos), float(cargo),
                    "COLLECTOR" if int(typ) == 0 else "PLANTER"]
        players.append([float(d["cash"][i][q]), centers, ws, {}])
    per_owner = {0: 0, 1: 0}
    for j in range(30):
        if d["trees_mask"][i][j]:
            pos, age, owner = d["trees"][i][j]
            owner = int(owner)
            tid = f"player-{owner}-tree-{per_owner[owner]}"
            per_owner[owner] += 1
            players[owner][3][tid] = [int(pos), float(age)]
    return {"player": p, "step": int(d["step"][i]),
            "carbon": [float(x) for x in d["carbon"][i]],
            "players": players, "trees": {},
            "remainingOverageTime": 0}


def score(rows, tag):
    correct = tot = 0
    per_type = {0: [0, 0], 1: [0, 0]}
    bcorrect = btot = 0
    rec = {1: [0, 0], 2: [0, 0]}
    for W, B, mask, wa, ba, wtype in rows:
        k = int(np.sum(mask))
        if k:
            pred = S._forward("w_", W[:k]).argmax(1)
            lab = wa[:k]
            correct += int(np.sum(pred == lab))
            tot += k
            for t in (0, 1):
                m = wtype[:k] == t
                per_type[t][0] += int(np.sum(pred[m] == lab[m]))
                per_type[t][1] += int(np.sum(m))
        bpred = S._forward("b_", B).argmax(1)
        bcorrect += int(np.sum(bpred == ba))
        btot += len(ba)
        for c in (1, 2):
            m = ba == c
            rec[c][0] += int(np.sum(bpred[m] == c))
            rec[c][1] += int(np.sum(m))
    print(f"{tag}: worker acc {correct / tot:.4f} "
          f"(collector {per_type[0][0] / per_type[0][1]:.4f}, "
          f"planter {per_type[1][0] / per_type[1][1]:.4f})  "
          f"base acc {bcorrect / btot:.4f}  "
          f"RECC rec {rec[1][0] / max(rec[1][1], 1):.3f}  "
          f"RECP rec {rec[2][0] / max(rec[2][1], 1):.3f}")


def main():
    _z = np.load(os.path.join(HERE, "bc_data", "claude_family_fixed.npz"),
                 allow_pickle=True)
    d = {k: _z[k] for k in _z.files}
    score(build_rows(d, False), "A direct-featurize")
    score(build_rows(d, True), "B canonical-roundtrip")

    # C) behavior trace vs example
    sys.path.insert(0, os.path.join(HERE, "env_src",
                                    "carbon_challenge-master"))
    from fastsim import FastCarbon
    from eval_v0_5 import CFG
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ex", os.path.join(HERE, "example.py"))
    ex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex)
    sim = FastCarbon(CFG).reset(seed=2000)
    shadow = S.make_agent()
    from collections import Counter
    for t in range(300):
        acts = [shadow(sim.obs(0), CFG), ex.agent(sim.obs(1), CFG)]
        sim.step(acts[0], acts[1])
        if t % 30 == 29 or t == 0:
            st = sim.get_state()
            ws = [w for w in st["workers"].values() if w["player"] == 0]
            print(f"step {t + 1}: cash0 {st['cash'][0]:.0f} "
                  f"workers {len(ws)} "
                  f"cargo {sum(w['carbon'] for w in ws):.0f} "
                  f"acts {Counter(acts[0].values())}")


if __name__ == "__main__":
    main()
