"""Validate a half-turn in the official rule engine and the original feature pipeline."""
from pathlib import Path
import hashlib
import json
import os
import random
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ENGINE = REPO / "code/env_src/carbon_challenge-master"
sys.path[:0] = [str(ENGINE), str(HERE.parent / "opponent_policy")]
os.sched_setaffinity(0, {0})
for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CARBON_DEBUG"] = "1"
import numpy as np
import torch
from original_policy import PROTOCOL, module, original, tape
from zerosum_env import make
from zerosum_env.envs.carbon import idgen
from zerosum_env.envs.carbon.helpers import Board
from rotate import rotate_cell, rotate_features, rotate_players, rotate_observation, rotate_orders, encode_rotated, WORKER_INVERSE


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def differences(a, b, path=()):
    if isinstance(a, dict):
        if a.keys() != b.keys():
            yield (path, "keys", sorted(a), sorted(b))
        else:
            for key in a:
                yield from differences(a[key], b[key], (*path, key))
    elif isinstance(a, (tuple, list)):
        if len(a) != len(b):
            yield (path, "length", len(a), len(b))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                yield from differences(x, y, (*path, i))
    elif a != b:
        yield (path, "value", a, b)


def official_turn(obs, actions, cfg, counters):
    try:
        return Board(obs, cfg, actions).next().observation
    finally:
        idgen._id_dict.clear()
        idgen._id_dict.update(counters)


def inspect_card(c, mod, times):
    base_seat, seed = c["expert_seat"], c["seed"]
    tape_data = tape(c)
    cfg = dict(tape_data["game_configuration"], randomSeed=seed)
    random.seed(seed)
    np.random.seed(seed)
    env = make("carbon", configuration=cfg)
    tracker = mod.BeliefTracker(mod._Bel, base_seat)
    prev = None
    rows = []
    for t, joint in enumerate(tape_data["actions"]):
        state = env.states[0].observation
        own = env.states[base_seat].observation
        players = state.players
        est, known = tracker.step(players, t, own.carbon)
        if t in times:
            # Only the official equivalence oracle gets the complete refereed board.
            full = dict(step=t, player=base_seat, players=players, trees=state.trees,
                        carbon=state.full_carbon, remainingOverageTime=own.remainingOverageTime)
            rot = rotate_observation(full)
            ro = [rotate_orders(a) for a in joint]
            assert rotate_observation(rot) == full and [rotate_orders(a) for a in ro] == joint
            ctr = dict(idgen._id_dict)
            oracle = official_turn(full, joint, cfg, ctr)
            transformed = official_turn(rot, ro, cfg, ctr)
            diffs = list(differences(rotate_observation(oracle), transformed))
            numeric = [abs(float(v)-float(w)) for path,kind,v,w in diffs if kind=="value"
                       and isinstance(v,(int,float)) and isinstance(w,(int,float))]
            # Source-visible history only from here. No opponent command or full_carbon enters these encodings.
            xs = []
            planar = []
            for mask in (known, np.ones(225)):
                x = mod.encode(players,t,base_seat,est,mask,prev)
                rx = encode_rotated(mod,players,t,base_seat,est,mask,prev)
                assert np.array_equal(rotate_features(x), rx) and np.array_equal(rotate_features(rx), x)
                q, rq = mod._quant(x), mod._quant(rx)
                assert np.array_equal(rotate_features(q),rq)
                xs.extend([q,rq])
                planar.append(sha(q.tobytes()))
            sc = mod._scalars(players,t,base_seat)
            assert np.array_equal(sc, mod._scalars(rotate_players(players),t,base_seat))
            scores=mod._net()(np.stack(xs),mod._SID,np.tile(sc,(4,1)))[2]
            rows.append(dict(step=t, seat=base_seat,
                             terrain=c['terrain'], seed=seed, referee_differences=len(diffs),
                             maximum_numeric_difference=max(numeric,default=0.),
                             all_differences_numeric=len(numeric)==len(diffs), first_differences=diffs[:4],
                             differences_outside_tree_absorption=sum(not(path[0]=="trees" and path[-1]==1) for path,*_ in diffs),
                             source_plane_hashes=planar, critic_public=[float(z) for z in scores[:2]],
                             critic_rollout_mask=[float(z) for z in scores[2:]],
                             base_set_half_turn_invariant=all(set(p[1].values())==set(r[1].values()) for p,r in zip(players,rot["players"]))))
        tracker.set_actions(players,joint[base_seat])
        prev=players
        env.step(joint, [])
    assert env.done and [p.reward for p in env.states]==tape_data["official_score_by_seat"]
    assert [r['step'] for r in rows]==list(times)
    return rows


def decision_rows(mod):
    directory=HERE.parent/"rollout_readout/timing"
    panel=json.loads((directory/"panel.json").read_bytes())
    inp=directory/"panel.npz"
    assert sha(inp.read_bytes())==panel["features_sha256"]
    roots=[]
    with np.load(inp,allow_pickle=False) as features:
        for r in panel["public_roots"]:
            # The final native neural batch is verifiably a full equal-depth scalar pass only
            # on planned depth 8 with an own-only, scalar-bearing batch of two survivors.
            b=r['batches'][-1]
            row={k:r[k] for k in ('terrain','seat','t','seed','depth','candidates')}
            ok=r['depth']==mod._S_D2 and b['kind']=='own' and b['scalar'] and b['rows']==2
            row['eligible']=ok
            full_batch=r["depth"]==mod._S_D2 and b["kind"]=="own" and b["scalar"] and b["rows"]==max(2, int(np.ceil(r["candidates"]*.25)))
            row["all_survivors_descriptive"]=full_batch
            if full_batch:
                x=features[b['key']].astype(np.float32)*np.float32(.01)
                s=features[b['key']+'s']
                native=mod._net()(x,b['sid'],s)[2].astype(float)
                rotated=mod._net()(rotate_features(x),b['sid'],s)[2].astype(float)
                avg=(native+rotated)/2
                def choose(z):
                    best=int(np.argmax(z))
                    return best if z[best]-z[0]>mod._S_EPS else 0
                a,best=choose(native),choose(avg)
                row.update(native=native.tolist(),rotated=rotated.tolist(),mean=avg.tolist(),
                           original_best=a, averaged_best=best, selection_changed=a!=best,
                           inputs_equal_between_choices=bool(np.array_equal(x[a],x[best]) and np.array_equal(s[a],s[best])),
                           old_margin_between_choices=float(native[a]-native[best]),
                           new_margin_between_choices=float(avg[best]-avg[a]))
                if ok:
                    row.update(raw_pair_order_changed=bool((native[0]>native[1])!=(avg[0]>avg[1])),
                               margin_rule_changed=bool((native[1]-native[0]>mod._S_EPS)!=(avg[1]-avg[0]>mod._S_EPS)))
            roots.append(row)
    return roots


def main(destination):
    plan=json.loads((HERE/'geometry_protocol.json').read_text())
    raw=original()
    assert sha(raw)==plan['source']
    mod=module(raw)
    torch.set_num_threads(1)
    for i in range(225):
        for k,p in enumerate(WORKER_INVERSE):
            assert rotate_cell(mod.move(i,k))==mod.move(rotate_cell(i),p)
    training=[g for g in PROTOCOL['games'] if g['split']=='train']
    terrain=sorted({g['terrain'] for g in training})[:2]
    runs=[c for c in training if c['terrain'] in terrain]
    assert len(runs)==4
    rows=[item for c in sorted(runs,key=lambda c:(c['terrain'],c['expert_seat']))
          for item in inspect_card(c,mod,plan['steps'])]
    candidates=decision_rows(mod)
    out=dict(protocol_sha256=sha((HERE/'geometry_protocol.json').read_bytes()), source_sha256=sha(raw),
             rotation_source_sha256=sha((HERE/'rotate.py').read_bytes()),official_engine={p:sha((ENGINE/p).read_bytes()) for p in ('zerosum_env/envs/carbon/helpers.py','zerosum_env/envs/carbon/carbon.py','zerosum_env/envs/carbon/idgen.py')},
             directional_identities=225*5,root_rows=rows,captured_original_last_batches=candidates)
    destination.write_text(json.dumps(out,allow_nan=False,indent=2)+'\n')


if __name__=="__main__":
    main(Path(sys.argv[1]))
