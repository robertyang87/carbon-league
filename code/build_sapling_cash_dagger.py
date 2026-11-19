"""Collect Echo4 actor labels on Sapling V2's actual first-120-turn trajectories.

Only the student's action drives the game. The teacher consumes the same observed
history. The labels, legality and decoder features share the teacher's prefix.
"""
import argparse
import base64
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import zlib

import numpy as np

import cnet2_feat as F
from cnet2_build import FIELDS
from cnet2_np import Policy
from eval_v53 import CFG, _extinct
from fastsim import FastCarbon

ROOT = Path(__file__).resolve().parents[1]
STUDENT = ROOT / "data/weights/sapling_1200k/top5_h2h_v2_agent.py"
TEACHER = ROOT / "code/packs/echo4_int8.py"
OPPONENTS = ("policy_raid_v0_1.py", "policy_v0_5.py")
SEEDS = (*range(620113400, 620113420), *range(620113500, 620113508))
PUBLIC = ("maps", "scal", "tok_pos", "tok_kind", "tok_mask", "tok_cargo",
          "hist_cell", "hist_cargo", "hist_move")
DTYPE = {k: v[0] for k, v in FIELDS.items()}
DTYPE.update(econ_mask=np.bool_, execution_y=np.int8)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def start_worker():
    global ghost, actor, rivals
    teacher = module(TEACHER, "cash_v3_echo4")
    if hashlib.md5(TEACHER.read_bytes()).hexdigest() != "819250bfc82632786063dabd46a616bd":
        raise ValueError("Echo4 source mismatch")
    w = teacher._dequantize(zlib.decompress(base64.b64decode(teacher._B1)), teacher._Q1)
    w["meta"] = np.array(teacher._M1)
    ghost = Policy(w)
    if digest(STUDENT) != "dc20c132a80424e54c3a61fac7c3c31eca13c79119073165b6693b27cf07ce29":
        raise ValueError("Sapling V2 source mismatch")
    actor = module(STUDENT, "cash_v3_student")
    rivals = [module(ROOT / "code" / f, "cash_v3_opp_" + str(i)) for i, f in enumerate(OPPONENTS)]


def play(job):
    i, out = job
    seed = SEEDS[i]
    seat = i % 2
    rival_id = (i // 2) % 2
    random.seed(seed)
    np.random.seed(seed)
    pressure = i >= 20
    rival = None if pressure else rivals[rival_id].make_agent()
    rival_feat = F.Feat(1 - seat) if pressure else None
    sim = FastCarbon(CFG).reset(seed=seed)
    feat = F.Feat(seat)
    rows = {k: [] for k in DTYPE}
    count = dict(choices=0, economic=0, economic_changed=0, loaded=0, broke=0)
    steps = 0
    for t in range(120):
        obs = sim.obs(seat)
        x = feat.observe(obs)
        teacher_y = ghost.act(x, feat.aux, teacher=2)[0]
        physical = actor.agent(obs, sim.config)
        executed_y = feat.labels(physical)
        prefix = F.Prefix(x, feat.aux)
        uf = np.zeros((F.OWN, F.UF), np.float32)
        mf = np.zeros((F.OWN, 5, F.MF), np.float32)
        legal = np.zeros((F.OWN, 5), bool)
        mask = np.zeros(F.OWN, bool)
        own = obs["players"][seat]
        cash = float(own[0])
        n_col = sum(w[2] == "COLLECTOR" for w in own[2].values())
        n_plan = len(own[2]) - n_col
        for s in range(F.OWN):
            if not x["tok_mask"][s]:
                continue
            uf[s], mf[s], legal[s] = prefix.feats(s)
            if not legal[s, teacher_y[s]]:
                raise ValueError("ghost produced a prefix-illegal decision")
            prefix.advance(s, int(teacher_y[s]))
            role = int(x["tok_kind"][s])
            cargo = float(x["tok_cargo"][s])
            mask[s] = (role == 0 and (cargo >= 30 or (cash <= 60 and cargo == 0))
                       or role == 2 and cash <= 60 and (n_col < 2 or n_plan == 0))
            mask[s] &= legal[s].sum() >= 2
            count["choices"] += int(legal[s].sum() >= 2)
            count["loaded"] += int(role == 0 and cargo >= 30)
        count["broke"] += int(cash < 30)
        count["economic"] += int(mask.sum())
        count["economic_changed"] += int((mask & (teacher_y != executed_y)).sum())
        values = {k: x[k] for k in PUBLIC}
        # These two fields are loader scaffolding for an action-only experiment.
        # Full carbon is confined to its own auxiliary target; 120-step windows
        # have no terminal utility, whose missing target is represented by NaN.
        carbon = F.q8(np.asarray(sim.carbon)[F.CANON if seat else F.GRID] / 100.)
        values.update(uf=uf, mf=mf, legal=legal, y=teacher_y, execution_y=executed_y,
                      econ_mask=mask, step=t, gid=i, teacher=2, value=np.nan, carbon=carbon)
        for k, dtype in DTYPE.items():
            rows[k].append(np.asarray(values[k], dtype=dtype))
        rival_obs = sim.obs(1-seat)
        if pressure:
            rival_x = rival_feat.observe(rival_obs)
            rival_y = ghost.act(rival_x, rival_feat.aux, teacher=2)[0]
            opponent_action = rival_feat.to_actions(rival_y)
        else:
            opponent_action = rival(rival_obs, sim.config)
        acts = (physical, opponent_action) if seat == 0 else (opponent_action, physical)
        sim.step(*acts)
        steps += 1
        post = sim.get_state()
        if any(_extinct(post, p) for p in (0, 1)):
            break
    shard = out / f"cash-{seed}.npz"
    np.savez_compressed(shard, **{k: np.stack(v) for k, v in rows.items()})
    return dict(id=f"student-v2-{seed}", seed=seed, seat=seat, split="train" if i < 16 or 20 <= i < 26 else "val",
                model="Echo4 quantized actor, style 2", opponent="Echo4 bare actor (independent opponent seat history)" if pressure else OPPONENTS[rival_id], rows=steps,
                shard=shard.name, sha256=digest(shard), **count)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=2)
    a = p.parse_args()
    if any(k.startswith("CNET2_") for k in os.environ):
        p.error("clear CNET2_* build or feature overrides")
    a.out.mkdir(parents=True, exist_ok=False)
    jobs = [(i, a.out) for i in range(len(SEEDS))]
    with ProcessPoolExecutor(a.workers, initializer=start_worker) as pool:
        games = []
        for game in pool.map(play, jobs):
            games.append(game)
            print(game["id"], game["split"], game["rows"], game["economic_changed"], flush=True)
    publish(a.out, games)


def publish(out, games):
    if [g["seed"] for g in games] != list(SEEDS):
        raise ValueError("missing or reordered planned map")
    source = [Path(__file__), TEACHER, STUDENT] + [ROOT / "code" / n for n in
             ("fastsim.py", "cnet2_feat.py", "cnet2_np.py", "eval_v53.py", *OPPONENTS)]
    report = dict(schema="sapling-elite-bc/1", style_slot=2, games=games,
                  files={str(p.relative_to(ROOT)): digest(p) for p in source},
                  economic_mask="public pre-action: collector with cargo >=30 or empty with cash <=60, "
                                "or base with cash <=60 and <2 collectors or no planter; 2+ legal commands under teacher prefix",
                  label="Echo4 style-2 greedy int8 actor on actual observation history; no search/guard",
                  behavior="frozen Sapling top5_h2h_v2 int8 pack", max_transitions=120,
                  value_target="unavailable (NaN), action-only dataset", seed_rule="620113400..419: first 16 train, last 4 holdout, two scripted opponents. 620113500..507: first 6 train, last 2 holdout, strong Echo4 actor. Seats alternate, maps fixed before labels/exams.")
    manifest = out / "manifest.json"
    stage = manifest.with_suffix(".tmp")
    stage.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    stage.replace(manifest)


if __name__ == "__main__":
    main()
