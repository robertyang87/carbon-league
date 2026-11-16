#!/usr/bin/env python3
"""Part 3 refit step 1 (strategy-group ruling 2026-11-15 13:34, sec.3
disposition (iii)): re-extract z for the Part 2 held-out roots through the
DEPLOYED leaf's own computation path -- the A base pack's bytes
(echo3_search_pack.py, ea20a271): pack-embedded cnet2_feat (2a45b08b),
pack-embedded cnet2_np (1433b5ac, via the pack search module's encode_b),
pack decoded int8 weights (payload md5 444cc54c). Everything else mirrors
Part 2's fit_z.py pipeline item by item: same held-out root set (ds
filename order [32:64], steps [200,299), both seats, 6,336 roots), same
seat-oriented realized margins from the ds meta scores.

Per root (mirrors test_leaf.py T2, the wiring-check caliber that produced
construction receipt sec.3's dz values):
  root obs assembled as in gate_driver.tasks_from_replays (step/players/
  trees from the seat-0 entry, carbon = the seat's own entry, player=seat);
  sim = pack fastsim (61859708) load_observation(seat-0 full observation)
  (cont_driver precedent);
  fm = Feat(seat); fm.observe(root obs)          # agent Feat at the root
  fc = feat_copy(fm); xm = fc.observe(sim.obs(seat))
  ctx = encode_b(pol, stack_x([xm]), [T=2])[1][0]
  z_pack = ctx @ pol.w["value.weightT"][:, 0] + pol.w["value.bias"][0]

Pairing: rows are emitted in heldout_z.npz row order, joined on
(game_key, seat, step); every margin is cross-checked against the
held-out manifest's recorded scores.

--validate: boundary cases (steps 297/298/299) -- z_pack must reproduce
part3/data/leaf_tests.json's z_pack values (T2's own construction:
sim built from the case obs, fm observes the case obs).
"""
import argparse
import base64
import gzip
import hashlib
import importlib.util
import json
import os
import sys
import zlib

import numpy as np

BASE = "/home/hatch/workspace/katago-probe2"
PACK = os.path.join(BASE, "modules", "echo3_search_pack.py")
PACK_MD5 = "ea20a2714209e1a58b3e435b41d943bb"
PAYLOAD_MD5 = "444cc54c0d88da5656ff4c7eef1ec8eb"
HELDOUT_Z = os.path.join(BASE, "part2b", "data", "heldout_z.npz")
HELDOUT_Z_MD5 = "17f26aa926f9a686e34ab6fba3bc331e"
MANIFEST = os.path.join(BASE, "part2", "data", "heldout_manifest.json")
MANIFEST_MD5 = "de0ab6d83c2f3b05f09a360aa51930dd"
REPLAYS = os.path.join(BASE, "part3", "work", "refit", "replays")
SHARDS = os.path.join(BASE, "part3", "work", "refit", "shards")
OUT = os.path.join(BASE, "part3", "data", "heldout_pack_z.npz")
BOUNDARY = os.path.join(BASE, "boundary_cases.jsonl")
LEAF_TESTS = os.path.join(BASE, "part3", "data", "leaf_tests.json")
TEACHER = 2


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_bytes(b):
    return hashlib.md5(b).hexdigest()


def load_pack(path):
    spec = importlib.util.spec_from_file_location("pack_refit", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    raw = zlib.decompress(base64.b64decode(m._B1))
    assert md5_bytes(raw) == PAYLOAD_MD5, "decoded payload md5 drift"
    w = m._dequantize(raw, m._Q1)
    w["meta"] = np.array(m._M1)
    return m, w


def cells(fn):
    return {n: c.cell_contents for n, c in
            zip(fn.__code__.co_freevars, fn.__closure__ or ())}


def grab(agent_fn, *names):
    seen, stack, owner = set(), [agent_fn], None
    while stack:
        fn = stack.pop()
        if id(fn) in seen or not hasattr(fn, "__code__"):
            continue
        seen.add(id(fn))
        c = cells(fn)
        if "leaf" in c:
            owner = c
            break
        stack.extend(v for v in c.values() if hasattr(v, "__code__"))
    assert owner is not None, "leaf not found in closure graph"
    lc = cells(owner["leaf"])
    return {n: (lc[n] if n in lc else owner[n]) for n in names}


class PackZ:
    def __init__(self):
        assert md5_file(PACK) == PACK_MD5, "base A pack md5 drift"
        self.mod, w = load_pack(PACK)
        agent = self.mod._S.make_search_agent(self.mod.CONF, weights=w)
        g = grab(agent, "pol", "T")
        self.pol = g["pol"]
        assert g["T"] == TEACHER, g["T"]
        self.feat_copy = getattr(self.mod._S, "feat_copy")
        self.encode_b = getattr(self.mod._S, "encode_b")
        self.stack_x = getattr(self.mod._S, "stack_x")
        self.Feat = self.mod._P["cnet2_feat"].Feat
        self.FastCarbon = self.mod._P["fastsim"].FastCarbon

    def z(self, fm, sim, seat):
        fc = self.feat_copy(fm)
        xm = fc.observe(sim.obs(seat))
        ctx = self.encode_b(self.pol, self.stack_x([xm]),
                            np.array([TEACHER]))[1][0]
        return float(ctx @ self.pol.w["value.weightT"][:, 0]
                     + self.pol.w["value.bias"][0])

    def z_root(self, root_obs, full_obs, seat):
        sim = self.FastCarbon()
        sim.load_observation(full_obs)
        fm = self.Feat(seat)
        fm.observe(root_obs)
        return self.z(fm, sim, seat)


def validate(pz):
    cases = [json.loads(l) for l in open(BOUNDARY)]
    case = {(c["step"], c["seat"]): c for c in cases}
    ref = json.load(open(LEAF_TESTS))
    worst = 0.0
    for step in (297, 298, 299):
        for me in (0, 1):
            c = case[(step, me)]
            obs = c["obs"]
            full = {"step": obs["step"], "players": obs["players"],
                    "trees": obs.get("trees") or {},
                    "full_carbon": [float(v) for v in obs["carbon"]],
                    "carbon": obs["carbon"], "remainingOverageTime": 60}
            sim = pz.FastCarbon()
            sim.load_observation(full)
            fm = pz.Feat(me)
            fm.observe(obs)
            z = pz.z(fm, sim, me)
            want = ref[f"z_step{step}_me{me}"]["z_pack"]
            dz = abs(z - want)
            worst = max(worst, dz)
            print(f"validate step {step} me {me}: z_pack {z:.10f} "
                  f"vs receipt {want:.10f}  |diff| {dz:.3e}")
            assert dz < 1e-6, (step, me, z, want)
    print(f"validation OK: worst |diff| {worst:.3e} < 1e-6 "
          f"(construction receipt sec.3 z_pack reproduced)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--games", type=int, default=None,
                    help="only the first N held-out games (smoke)")
    ap.add_argument("--mod", type=int, default=1)
    ap.add_argument("--rem", type=int, default=0,
                    help="process games with index %% mod == rem "
                         "(disjoint parallel shards)")
    a = ap.parse_args()

    pz = PackZ()
    if a.validate:
        validate(pz)
        return

    assert md5_file(HELDOUT_Z) == HELDOUT_Z_MD5, "heldout_z.npz md5 drift"
    assert md5_file(MANIFEST) == MANIFEST_MD5, "manifest md5 drift"
    hz = np.load(HELDOUT_Z, allow_pickle=True)
    man = {g["key"]: g for g in json.load(open(MANIFEST))}
    keys = [str(x) for x in hz["game_key"]]
    seat = np.asarray(hz["seat"]).astype(int)
    step = np.asarray(hz["step"]).astype(int)
    z_fp = np.asarray(hz["z"], dtype=np.float64)
    margin = np.asarray(hz["margin"], dtype=np.float64)
    n = len(keys)
    assert n == 6336, n

    # margin cross-check against manifest scores, every root
    for i in range(n):
        sc = man[keys[i]]["scores"]
        assert abs(margin[i] - (sc[seat[i]] - sc[1 - seat[i]])) < 1e-9, i
    print("margin cross-check vs manifest scores: 6336/6336 OK", flush=True)

    by_game = {}
    for i in range(n):
        by_game.setdefault(keys[i], []).append(i)
    game_order = list(by_game)
    if a.games:
        game_order = game_order[:a.games]

    # Sharded per game (owner rule 2026-11-15 13:41: staged, externally
    # readable progress + breakpoint resume). Progress = shard count in
    # SHARDS / total games; a rerun skips games whose shard is present
    # with the expected root count and re-derives the rest.
    os.makedirs(SHARDS, exist_ok=True)
    z_pack = np.full(n, np.nan)
    for gi, gk in enumerate(game_order):
        if gi % a.mod != a.rem:
            continue
        rows = by_game[gk]
        shard = os.path.join(SHARDS, gk + ".npz")
        if os.path.exists(shard):
            sd = np.load(shard)
            if len(sd["idx"]) == len(rows):
                z_pack[np.asarray(sd["idx"])] = sd["z_pack"]
                print(f"game {gi + 1}/{len(game_order)} {gk[:8]} "
                      f"resumed from shard ({len(rows)} roots)",
                      flush=True)
                continue
        g = man[gk]
        rp = os.path.join(REPLAYS, gk)
        assert md5_file(rp) == g["md5"], f"replay md5 drift: {gk}"
        d = json.load(gzip.open(rp, "rt"))
        steps = d["steps"]
        gz = []
        for i in rows:
            t, s = int(step[i]), int(seat[i])
            o0 = steps[t][0]["observation"]
            ep = steps[t][s]["observation"]
            assert int(o0["step"]) == t
            root_obs = {"step": int(o0["step"]), "players": o0["players"],
                        "carbon": ep["carbon"], "player": s,
                        "trees": o0.get("trees") or {},
                        "remainingOverageTime": ep.get(
                            "remainingOverageTime",
                            o0.get("remainingOverageTime"))}
            gz.append(pz.z_root(root_obs, o0, s))
        z_pack[rows] = gz
        np.savez(shard, idx=np.asarray(rows), z_pack=np.asarray(gz))
        print(f"game {gi + 1}/{len(game_order)} {gk[:8]} done "
              f"({len(rows)} roots), shard saved", flush=True)

    if a.games:
        print("smoke only; not saving")
        return
    if a.mod != 1:
        print("parallel shard worker done; assembly is a separate "
              "--mod 1 run once all shards exist")
        return
    assert np.isfinite(z_pack).all()
    np.savez(OUT, z_pack=z_pack, z_fp=z_fp, margin=margin,
             game_key=np.asarray(keys), seat=seat, step=step)
    print("saved", OUT, "md5", md5_file(OUT))


if __name__ == "__main__":
    main()
