"""Prepare coarse full-game samples with paired one-turn economic labels and independently signed multi-turn roots."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
from actions import _jrc_context,_jrc_repr,_jrc_options,JRC_ROLES
D=Path(__file__).resolve().parent;ACTOR=D.parent;BASE=ACTOR.parent;ROOT=BASE.parents[1];PARENT=ROOT/'data/weights/sapling_1200k/terminal_return_fit_v1.py'
sys.path.insert(0,str(ROOT/'code'))
from fastsim import FastCarbon


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact(x):
    return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def load_parent():
    assert sha(PARENT)==read(D/'protocol.json')['parent_sha256']
    spec=importlib.util.spec_from_file_location('jrc_parent',PARENT);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    src=PARENT.read_text();selector=src[src.rindex('\ndef _mixed_actions('):]
    assert selector.count('return baseline')==3 and selector.count('return candidate')==1
    selector=selector.replace('return baseline','return _jrc_capture(x,aux,enc,baseline)').replace('return candidate','return _jrc_capture(x,aux,enc,candidate)')
    def capture(x,aux,enc,labels):
        mod._jrc_root=(x,aux,enc[1][1],labels)
        return labels
    mod._jrc_capture=capture;exec(compile(selector,'jrc-parent-capture','exec'),mod.__dict__);return mod


def value(sim,s):
    cargo=[0.,0.];units=[0,0];trees=[0,0]
    for w in sim.workers.values():units[w.player]+=1;cargo[w.player]+=w.carbon
    for q in sim.trees.values():trees[q.player]+=1
    j=1-s
    return np.array([sim.cash[s],sim.cash[j],cargo[s]-cargo[j],units[s]-units[j],trees[s]-trees[j]],dtype=np.float64)


def exclusive_key(seed,s,t,orders):
    return hashlib.sha256(compact([seed,s,t,orders])).digest()


def sampled(ctx,raw,obs,feat,x,aux,y,t):
    F,live,roles,*_=ctx
    base=_jrc_repr(ctx,obs,raw['game_configuration'],feat,x,aux,y)
    assert base is not None
    out=[(y,base[0],'parent')]
    for role in (0,1,2):
        options=[]
        for slot in live[roles==role]:
            for action in range(3 if role==2 else 5):
                if action==y[slot]:continue
                lab=y.copy();lab[slot]=action
                options.append((exclusive_key(raw['seed'],obs['player'],t,feat.to_actions(lab)),lab))
        n=0
        for _,lab in sorted(options,key=lambda v:v[0]):
            result=_jrc_repr(ctx,obs,raw['game_configuration'],feat,x,aux,lab)
            if result is not None and result[1].issubset(base[1]):
                out.append((lab,result[0],JRC_ROLES[role]));n+=1
                if n==2:break
    return out


def extract_aux(registry,shard,shards):
    m=load_parent(); rows=[]; counts=[]
    for gi,g in enumerate(registry['games']):
        if gi%shards!=shard:continue
        path=ROOT/g['source'];r=read(path);assert sha(path)==g['sha256'] and r['initial_terrain_u8_sha256']==g['terrain']
        assert r['pack_sha256']==sha(PARENT) and r['examiner_md5']=='57f9a594cce4f2ec9e3573d8d9d26ca1' and not r['errors']
        cfg=r['game_configuration'];s=g['seat'];sim=FastCarbon(cfg).reset(seed=g['seed']);branch=FastCarbon(cfg);n=0
        for t,both in enumerate(r['actions']):
            obs=sim.obs(s); selected=t>=12 and ((t-12)%16==0 or t==298)
            if t==0 or selected:
                parent=m.agent(obs,cfg);assert parent==both[s],(g['seed'],s,t)
            else:m._FEAT.observe(obs)
            if selected:
                x,aux,e,y=m._jrc_root;ctx=_jrc_context(obs,cfg,m._FEAT,x,aux,e)
                opts=sampled(ctx,r,obs,m._FEAT,x,aux,y,t)
                st=sim.get_state(); sim.step(*both);base_val=value(sim,s)
                for lab,vec,role in opts[1:]:
                    commands=m._FEAT.to_actions(lab);b=list(both);b[s]=commands
                    branch.set_state(st);branch.step(*b);delta=value(branch,s)-base_val
                    rows.append((gi,t,role,opts[0][1],vec,delta));n+=1
            else:sim.step(*both)
        counts.append((g['seed'],s,n));print(shard,g['seed'],s,n,flush=True)
    np.savez_compressed(D/f'aux_{shard}.npz',map_idx=[r[0] for r in rows],t=[r[1] for r in rows],role=[r[2] for r in rows],
        b=np.stack([r[3] for r in rows]),a=np.stack([r[4] for r in rows]),target=np.stack([r[5] for r in rows]))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('shard',type=int);ap.add_argument('--shards',type=int,default=2);args=ap.parse_args()
    extract_aux(read(D/'source_registry.json'),args.shard,args.shards)
