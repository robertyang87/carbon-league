"""Fingerprint 851c's public observation history at curriculum handoff; test historical forced outputs offline."""
from collections import Counter
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
LEAGUE=HERE.parents[1]
ROOT=next(p for p in HERE.parents if (p/'code/env_src').is_dir())
PACK=LEAGUE/'natural_original/candidate_sampled.py'
SOURCE=LEAGUE.parents[1]/'direct_original/games/candidate/raw_621758346_1.json.gz'
SHA='851c901aa561ce80bf04d03e7c2a8e7876bfa2f24dc6c57ffe22b501979320ee'
SOURCE_SHA='7a5967383c78324d248f4b228541d7eb30f3ac3d39bf18e74f81a011a78c4631'
sys.path.insert(0,str(LEAGUE))
import collect

def need(condition,message):
    if not condition:raise ValueError(message)

def sha(b):return hashlib.sha256(b).hexdigest()

def state_hash(state):
    """Hash structured numpy buffers with dimensions, dtype and unambiguous nested values. No Python object addresses."""
    def tagged(tag,payload):
        return hashlib.sha256(tag+len(payload).to_bytes(8,'big')+payload).digest()
    def node(x):
        if x is None:return tagged(b'null',b'')
        if isinstance(x,np.generic):return node(x.item())
        if isinstance(x,bool):return tagged(b'bool',bytes([int(x)]))
        if isinstance(x,int):return tagged(b'int',str(x).encode())
        if isinstance(x,float):
            need(math.isfinite(x),'Nonfinite feature history scalar')
            return tagged(b'float',struct.pack('>d',x))
        if isinstance(x,str):return tagged(b'string',x.encode())
        if isinstance(x,np.ndarray):
            need(not x.dtype.hasobject,'Object array in feature history')
            a=np.ascontiguousarray(x)
            return tagged(b'array',node(a.dtype.str)+node(tuple(a.shape))+a.tobytes())
        if isinstance(x,dict):return tagged(b'dict',b''.join(sorted(node(k)+node(v) for k,v in x.items())))
        if isinstance(x,(set,frozenset)):return tagged(b'set',b''.join(sorted(map(node,x))))
        if isinstance(x,(list,tuple)):return tagged(b'list' if isinstance(x,list) else b'tuple',b''.join(map(node,x)))
        raise TypeError(f"Unexpected history type {type(x)}")
    return node(state).hex()

@lru_cache(maxsize=16)
def package_sha(path):return sha(Path(path).read_bytes())

def snapshot(player):
    """Feature history only. Use an additional generator-state digest when common random numbers are required."""
    mod=player.mod
    need(package_sha(mod.__file__)==SHA,'Unexpected curriculum student package')
    feat=mod._FEAT
    need(feat is not None,'Student has not yet observed its opening')
    return state_hash(vars(feat))

def behavior_rng_snapshot(player):
    rng=player.mod._behavior_rng
    need(rng is not None,'Student natural sampler is not initialized')
    return state_hash(rng.bit_generator.state)

def check():
    need(package_sha(PACK)==SHA and package_sha(SOURCE)==SOURCE_SHA,'Unexpected signed fixtures')
    raw=json.loads(gzip.decompress(SOURCE.read_bytes()))
    side=0
    need(raw['candidate_seat']==1 and raw['examiner_md5']=='57f9a594cce4f2ec9e3573d8d9d26ca1','Wrong physical original opponent')
    cfg=raw['game_configuration'];env=collect.begin(raw['seed'],cfg)
    ordinary=collect.Player(PACK);forced=collect.Player(PACK)
    # Control explicitly substitutes the external physical team's labels at the *output* of
    # a complete normal decode. Both sides consume the same natural RNG draws while their outputs differ.
    decoder=forced.mod._mixed_actions;box={};captures={}
    def replace_output(x,aux):
        own=decoder(x,aux)
        if box['t']>=240:return own
        captures['native']=forced.mod._FEAT.to_actions(own)
        return forced.mod._FEAT.labels(box['command'])
    forced.mod._mixed_actions=replace_output
    standalone=ordinary.modules['cnet2_feat'].Feat(side)
    chain=hashlib.sha256();rows=[];delta_count=0;rng_changes=[];fork=None;cold=None
    for t,orders in enumerate(raw['actions'][:raw['steps']]):
        public=collect.public(env,side)
        need(public['step']==t and not env.done,'Noncontiguous historical public input')
        box.update(t=t,command=orders[side])
        prev_rng=behavior_rng_snapshot(ordinary) if t else None
        ordinary_order=ordinary.act(public,cfg);forced_order=forced.act(public,cfg)
        if t==0:
            # Native initialization samples OS entropy independently. For this engineering comparison only,
            # couple the otherwise unmodified RNGs after opening. Step 0 has no stochastic decision.
            ordinary.mod._behavior_rng.bit_generator.state=np.random.default_rng(341690).bit_generator.state
            forced.mod._behavior_rng.bit_generator.state=np.random.default_rng(341690).bit_generator.state
        independent=standalone.observe(public)
        need(independent is not None,'Independent observation was not consumed')
        sf=snapshot(ordinary);sr=behavior_rng_snapshot(ordinary)
        need(sf==snapshot(forced)==state_hash(vars(standalone)),'History depends on overridden returned order')
        need(sr==behavior_rng_snapshot(forced) and state_hash(ordinary.mod._COUNTS)==state_hash(forced.mod._COUNTS),
             'Choice sampler or diagnostics lost alignment')
        if prev_rng and prev_rng!=sr:rng_changes.append(t)
        if t<240:
            physical={u:a for u,a in orders[side].items() if a not in ('NONE',None)}
            need(captures['native']==ordinary_order and forced_order==physical,'Explicit external action or coupled native decode mismatched')
            changed=ordinary_order!=forced_order;delta_count+=changed
        else:
            changed=False;need(ordinary_order==forced_order,'Post-prefix public student order changed')
        if t==239:fork=dict(last_observed_step=t,feature_sha256=sf,behavior_rng_sha256=sr)
        if t==240:
            fresh=ordinary.modules['cnet2_feat'].Feat(side);fresh.observe(public)
            cold=dict(complete=sf,cold_features=state_hash(vars(fresh)),same=(sf==state_hash(vars(fresh))))
            need(not cold['same'],'Discarding the public prefix failed to change the feature history')
        entry=dict(t=t,features=sf,rng=sr,prefix_order_differed=changed)
        chain.update(collect.canonical_bytes(entry));rows.append(entry)
        env.step(orders)
    need(env.done and [s.reward for s in env.states]==raw['official_score_by_seat'] and delta_count>0 and fork is not None,'Official historical replay differs')
    return dict(student_sha256=SHA,source_path=SOURCE.relative_to(ROOT).as_posix(),source_sha256=SOURCE_SHA,
                seed=raw['seed'],observer_seat=side,historical_override_actor='archived original Master3s2 in physical seat 0',
                source_scores=raw['official_score_by_seat'],source_policy_invocations=0,prefix_observations=240,
                prefix_disagree_with_historical_actual=delta_count,persistent_feature_equality_turns=len(rows),
                verified_distinct_posthandoff_student_steps=len(rows)-240,prefix_sampling_rng_updates=[i for i in rng_changes if i<240],
                all_sampling_rng_update_steps=rng_changes,common_handoff=fork,cold_negative_control=cold,
                feature_attributes=sorted(vars(standalone)),trace_sha256=chain.hexdigest(),per_turn=rows,
                scope="All factual public observations and environment transitions are archived; this test checks hidden-state/decoder mechanics, not intervention outcomes or opponent strength. The ordinary public agent was exercised twice per turn. Its recurrent decoder runs within the current joint action only. Last workers, trees and move histories update from next public observations; no previous returned action is fed into the next decode. The private natural sampler persists independently of this feature hash; a fork must separately declare whether its future random stream is shared or independent.")


if __name__=='__main__':
    r=check()
    path=HERE/'student_history.json';payload=json.dumps(r,sort_keys=True,allow_nan=False,separators=(',',':'))+'\n'
    if path.exists():need(path.read_text()==payload,'Existing offline control differs')
    else:path.write_text(payload)
    print({k:r[k] for k in ('seed','prefix_disagree_with_historical_actual','persistent_feature_equality_turns','verified_distinct_posthandoff_student_steps','common_handoff')})
