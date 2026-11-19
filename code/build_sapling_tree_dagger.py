"""Teach the frozen Sapling V3 actor on its own full-game trajectories.

Example: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/build_sapling_tree_dagger.py /tmp/tree-v4 --workers 3
"""
import argparse
import base64
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import zlib

import numpy as np

import cnet2_feat as F
from build_sapling_cash_dagger import PUBLIC, OPPONENTS, TEACHER, digest, module
from cnet2_build import FIELDS
from cnet2_np import Policy
from eval_v53 import CFG, _extinct
from fastsim import FastCarbon

ROOT = Path(__file__).resolve().parents[1]
STUDENT = ROOT / "data/weights/sapling_1200k/cash_v3_agent.py"
SEEDS = range(620124200, 620124224)
MASKS = ("planter_mask", "return_mask", "defense_mask", "recruit_mask")
DTYPES = {k: v[0] for k, v in FIELDS.items()}
DTYPES.update(dict.fromkeys((*MASKS, "econ_mask"), np.bool_))
DTYPES["execution_y"] = np.int8


def init():
    global teacher, student, rivals
    if digest(STUDENT) != "4cd4c1fcf40f05217ecba2a59095ba40286c5f96017d2b15285c942a3b0f5348":
        raise ValueError("unexpected V3 actor bytes")
    if hashlib.md5(TEACHER.read_bytes()).hexdigest() != "819250bfc82632786063dabd46a616bd":
        raise ValueError("unexpected Echo4 pack bytes")
    source = module(TEACHER, "tree_teacher")
    raw = zlib.decompress(base64.b64decode(source._B1))
    weights = source._dequantize(raw, source._Q1)
    weights["meta"] = np.array(source._M1)
    teacher = Policy(weights)
    student = module(STUDENT, "tree_behavior")
    rivals = [module(ROOT / "code" / f, "tree_rival_" + str(i)) for i, f in enumerate(OPPONENTS)]


def sample(x, feat, y, actual, t, gid):
    prefix = F.Prefix(x, feat.aux)
    masks = {k: np.zeros(F.OWN, bool) for k in MASKS}
    uf = np.zeros((F.OWN, F.UF), np.float16)
    mf = np.zeros((F.OWN, 5, F.MF), np.float16)
    legal = np.zeros((F.OWN, 5), bool)
    enemies = x["tok_pos"][12:22][x["tok_mask"][12:22]]
    for slot in np.flatnonzero(x["tok_mask"][:F.OWN]):
        uf[slot], mf[slot], legal[slot] = prefix.feats(slot)
        if not legal[slot, y[slot]]:
            raise ValueError("teacher action illegal under its own prefix")
        prefix.advance(slot, int(y[slot]))
        if legal[slot].sum() < 2:
            continue
        role = int(x["tok_kind"][slot])
        pos = int(x["tok_pos"][slot])
        masks["planter_mask"][slot] = role == 1
        masks["return_mask"][slot] = role == 0 and x["tok_cargo"][slot] >= 30
        masks["defense_mask"][slot] = role < 2 and (F.DIST[pos, enemies] <= 2).any()
        masks["recruit_mask"][slot] = role == 2
    row = {k: x[k] for k in PUBLIC}
    row.update(masks)
    row.update(uf=uf, mf=mf, legal=legal, y=y, execution_y=feat.labels(actual),
               econ_mask=np.logical_or.reduce(list(masks.values())), step=t, gid=gid, teacher=2,
               value=np.nan, carbon=np.zeros(F.N, np.uint8))
    return row


def play(job):
    i, out = job
    seed = SEEDS[i]
    random.seed(seed)
    np.random.seed(seed)
    seat = i % 2
    rival_id = i // 2 % 3
    rival = rivals[rival_id].make_agent() if rival_id < 2 else None
    rival_feat = F.Feat(1 - seat) if rival is None else None
    sim = FastCarbon(CFG).reset(seed=seed)
    feat = F.Feat(seat)
    rows = {k: [] for k in DTYPES}
    gained = [0, 0]
    captured = [0, 0]
    absorption = [0., 0.]
    for t in range(299):
        obs = sim.obs(seat)
        x = feat.observe(obs)
        actual = student.agent(obs, sim.config)
        # Dense opening and final countdown. Systematic midgame sample includes
        # all phases; always advance the student's observed feature history.
        if t < 40 or t >= 290 or t % 3 == i % 3:
            y = teacher.act(x, feat.aux, teacher=2)[0]
            row = sample(x, feat, y, actual, t, i)
            for k, dtype in DTYPES.items():
                rows[k].append(np.asarray(row[k], dtype=dtype))
        opp_obs = sim.obs(1 - seat)
        if rival is None:
            xr = rival_feat.observe(opp_obs)
            command = rival_feat.to_actions(teacher.act(xr, rival_feat.aux, teacher=2)[0])
        else:
            command = rival(opp_obs, sim.config)
        before = {tree.pos: tree.player for tree in sim.trees.values()}
        sim.step(*((actual, command) if seat == 0 else (command, actual)))
        for tid, tree in sim.trees.items():
            old = before.get(tree.pos)
            if old is None:
                gained[tree.player] += 1
            elif old != tree.player:
                captured[tree.player] += 1
            absorption[tree.player] += sim.trees_dict[tid][1]
        state = sim.get_state()
        if any(_extinct(state, p) for p in (0, 1)):
            break
    shard = out / f"tree-{seed}.npz"
    arrays = {k: np.stack(v) for k, v in rows.items()}
    np.savez_compressed(shard, **arrays)
    conflict = arrays["y"] != arrays["execution_y"]
    counts = {k: {"slots": int(arrays[k].sum()), "teacher_differs": int((arrays[k] & conflict).sum())}
              for k in (*MASKS, "econ_mask")}
    return dict(id=f"student-v3-{seed}", seed=seed, seat=seat, split="train" if i < 18 else "val",
                opponent=OPPONENTS[rival_id] if rival_id < 2 else "Echo4 bare style 2",
                rows=len(arrays["step"]), transitions=t+1, shard=shard.name, sha256=digest(shard),
                masks=counts, occupied_new_tree_cells_by_seat=gained,
                occupied_tree_cells_swapping_owner_by_seat=captured,
                tree_absorption_by_seat=[round(a, 5) for a in absorption],
                final_cash_by_seat=list(sim.cash))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    args = p.parse_args()
    if any(k.startswith("CNET2_") for k in os.environ):
        p.error("unset CNET2_* overrides")
    args.out.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(args.workers, initializer=init) as pool:
        games = []
        for game in pool.map(play, [(i, args.out) for i in range(len(SEEDS))]):
            games.append(game)
            print(game["id"], game["rows"], game["masks"], flush=True)
    files = [Path(__file__), STUDENT, TEACHER] + [ROOT / "code" / f for f in
             ("build_sapling_cash_dagger.py", "cnet2_feat.py", "cnet2_np.py", "cnet2_build.py",
              "eval_v53.py", "fastsim.py", *OPPONENTS)]
    manifest = dict(schema="sapling-elite-bc/1", style_slot=2, games=games,
                    files={str(f.relative_to(ROOT)): digest(f) for f in files},
                    label="Echo4 style-2 greedy quantized actor on continuous public history, no search/guard",
                    behavior="cash V3 quantized pack; only its own choices and opponent drive the real game",
                    sampling="t<40 or t>=290 or t%3==game_index%3, stopping at extinction or transition 299",
                    seed_rule="first 18 maps train, next 6 holdout; consecutive two-seat blocks versus raid, v0.5, Echo4; distinct seeds per seat",
                    role_filters="minimum 2 legal decisions; own planter; own collector cargo>=30; own worker within toroidal L1<=2 of any observed opposing worker; own base",
                    value_target="action only: value NaN, unused carbon placeholder all zeros")
    path = args.out / "manifest.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


if __name__ == "__main__":
    main()
