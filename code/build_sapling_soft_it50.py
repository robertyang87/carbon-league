"""Generate natural-frequency it50 conditionals on executed V3 or it50 prefixes.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/build_sapling_soft_it50.py /tmp/soft-it50
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
from build_sapling_cash_dagger import PUBLIC, OPPONENTS, TEACHER as ECHO4, digest, module
from cnet2_build import FIELDS
from cnet2_np import NEG, Policy
from eval_v53 import CFG, _extinct
from fastsim import FastCarbon

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "data/weights/sapling_1200k/cash_v3_agent.py"
IT50 = ROOT / "code/packs/ppou2_m_it50.py"
SEEDS = range(620176000, 620176024)
DTYPES = {k: v[0] for k, v in FIELDS.items()}
DTYPES.update(teacher_probs=np.float32, teacher_argmax=np.int8, soft_mask=np.bool_)


def actor(source, name):
    pack = module(source, name)
    # The explicit feature and decoder imports for both networks must match
    # the frozen package; otherwise previously computed conditionals may differ.
    for f in ("cnet2_feat", "cnet2_np"):
        if digest(ROOT / "code" / (f+".py")) != hashlib.sha256(pack._SRC[f].encode()).hexdigest():
            raise ValueError("embedded vs. local module mismatch: " + f)
    weights = pack._dequantize(zlib.decompress(base64.b64decode(pack._B1)), pack._Q1)
    weights["meta"] = np.array(pack._M1)
    return Policy(weights)


def init():
    global teacher, student, strong, rivals
    if digest(V3) != "4cd4c1fcf40f05217ecba2a59095ba40286c5f96017d2b15285c942a3b0f5348":
        raise ValueError("unexpected V3 bytes")
    if digest(IT50) != "57e2557665636a1bbc92ae46a4c33ee9dac609efcffefa1c30df1233746c2211":
        raise ValueError("unexpected it50 bytes")
    if hashlib.md5(ECHO4.read_bytes()).hexdigest() != "819250bfc82632786063dabd46a616bd":
        raise ValueError("unexpected Echo4 bytes")
    teacher = actor(IT50, "soft_it50_actor")
    strong = actor(ECHO4, "soft_it50_echo4")
    student = module(V3, "soft_it50_v3")
    rivals = [module(ROOT / "code" / f, "soft_it50_rival_" + str(i)) for i, f in enumerate(OPPONENTS)]


def sample(x, aux, executed, scores, t, gid):
    prefix = F.Prefix(x, aux)
    uf = np.zeros((F.OWN, F.UF), np.float16)
    mf = np.zeros((F.OWN, 5, F.MF), np.float16)
    legal = np.zeros((F.OWN, 5), bool)
    prob = np.zeros_like(scores, dtype=np.float32)
    best = np.full(F.OWN, -1, np.int8)
    active = x["tok_mask"][:F.OWN]
    if not np.array_equal(active, executed >= 0):
        raise ValueError("execution does not identify all own units")
    for s in np.flatnonzero(active):
        uf[s], mf[s], legal[s] = prefix.feats(s)
        allowed = legal[s]
        if not allowed[executed[s]] or not np.array_equal(scores[s] != NEG, allowed):
            raise ValueError("executed command, teacher logits or prefix mask disagree")
        v = scores[s, allowed].astype(np.float64)
        if not np.isfinite(v).all():
            raise ValueError("non-finite teacher logit")
        q = np.exp(v - v.max())
        prob[s, allowed] = q / q.sum()
        best[s] = int(np.argmax(scores[s]))
        prefix.advance(s, int(executed[s]))
    if not np.allclose(prob.sum(-1), active, atol=1.e-6) or np.any(prob[~legal] != 0):
        raise ValueError("teacher probability escaped its legal support")
    row = {k: x[k] for k in PUBLIC}
    row.update(y=executed, legal=legal, uf=uf, mf=mf, teacher_probs=prob, teacher_argmax=best,
               soft_mask=legal.sum(-1)>=2, step=t, gid=gid, teacher=2,
               value=np.nan, carbon=np.zeros(F.N, np.uint8))
    return row


def play(job):
    i, out = job
    seed = SEEDS[i]
    random.seed(seed)
    np.random.seed(seed)
    seat = i % 2
    rival_id = i // 2 % 3
    is_teacher = bool((i % 2) ^ ((i // 6) % 2) ^ ((i // 2) % 2))
    rival = rivals[rival_id].make_agent() if rival_id < 2 else None
    other_feat = F.Feat(1-seat) if rival is None else None
    feat = F.Feat(seat)
    sim = FastCarbon(CFG).reset(seed=seed)
    rows = {k: [] for k in DTYPES}
    for t in range(299):
        obs = sim.obs(seat)
        x = feat.observe(obs)
        if is_teacher:
            actions, logits, _ = teacher.act(x, feat.aux, teacher=2)
            actual = feat.to_actions(actions)
        else:
            actual = student.agent(obs, sim.config)
            actions = feat.labels(actual)
        # Fixed phase-offset systematic sampling across all game stages.
        # Every sampled root retains every available unit decision exactly once.
        if t % 3 == i % 3:
            if not is_teacher:
                forced, logits, _ = teacher.act(x, feat.aux, teacher=2, forced=actions)
                if not np.array_equal(forced, actions):
                    raise ValueError("teacher did not consume the actual prefix")
            if not np.array_equal(feat.labels(actual), actions):
                raise ValueError("command mapping differs from executed labels")
            data = sample(x, feat.aux, actions, logits, t, i)
            for k, dtype in DTYPES.items():
                rows[k].append(np.asarray(data[k], dtype=dtype))
        opp_obs = sim.obs(1-seat)
        if rival is None:
            other = other_feat.observe(opp_obs)
            command = other_feat.to_actions(strong.act(other, other_feat.aux, teacher=2)[0])
        else:
            command = rival(opp_obs, sim.config)
        sim.step(*((actual, command) if seat == 0 else (command, actual)))
        if any(_extinct(sim.get_state(), j) for j in (0, 1)):
            break
    arrays = {k: np.stack(v) for k,v in rows.items()}
    shard = out / f"soft-{seed}.npz"
    np.savez_compressed(shard, **arrays)
    mask = arrays["soft_mask"]
    kind = arrays["tok_kind"][:, :F.OWN]
    return dict(id=f"soft-it50-{seed}", seed=seed, seat=seat, split="train" if i<18 else "val",
                behavior="it50 bare style 2" if is_teacher else "Sapling cash V3",
                opponent=OPPONENTS[rival_id] if rival_id < 2 else "Echo4 bare style 2",
                model="it50 bare style 2", rows=len(arrays["step"]), transitions=t+1,
                shard=shard.name, sha256=digest(shard),
                slots_by_role=[int((mask & (kind==r)).sum()) for r in (0, 1, 2)],
                slots=int(mask.sum()), differing=int((mask & (arrays["y"]!=arrays["teacher_argmax"])).sum()),
                roots_by_stage=[int(((arrays["step"]>=lo)&(arrays["step"]<hi)).sum())
                                for lo, hi in ((0,100),(100,200),(200,299))])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--workers", type=int, choices=(1, 2), default=2)
    a = p.parse_args()
    if any(k.startswith("CNET2_") for k in os.environ):
        p.error("unset CNET2_* overrides")
    a.out.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(a.workers, initializer=init) as pool:
        games = []
        for game in pool.map(play, [(i,a.out) for i in range(len(SEEDS))]):
            games.append(game)
            print(game["seed"], game["rows"], game["slots_by_role"], flush=True)
    files = [Path(__file__), IT50, V3, ECHO4] + [ROOT / "code" / f for f in
             ("build_sapling_cash_dagger.py", "cnet2_feat.py", "cnet2_np.py", "cnet2_build.py",
              "eval_v53.py", "fastsim.py", *OPPONENTS)]
    m = dict(schema="sapling-soft-it50/1", style_slot=2, games=games,
             files={str(f.relative_to(ROOT)):digest(f) for f in files},
             sampling="t % 3 == game_index % 3, stopping at the first extinction or at 299; no event or role sampling",
             label="T=1 softmax over prefix-legal it50 bare style-2 logits, conditioned on y (the actual executed joint command), not teacher argmax",
             soft_mask="every own unit with >=2 legal actions, no role weights; inactive probabilities zero, forced slots one-hot",
             split_rule="620176000..017 train, 620176018..023 val. Two distinct maps per seat pair, evenly stratified by opponent. No exam sources.",
             value_target="unavailable (NaN); unused carbon target all zeros")
    path = a.out / "manifest.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(m, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


if __name__ == "__main__":
    main()
