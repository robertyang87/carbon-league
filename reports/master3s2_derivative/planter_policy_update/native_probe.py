"""Trace the registered held roots through v2 and the two fitted own-planter heads."""
import argparse
import collections
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROTOCOL_SHA = '24142ebf41f4b97e486589de916435b98ce92b3cd9271650ab6e6805a24d77c8'
HEAD_SHA = '845e8f23c8c69ca310b9a41cfb42fecf72663a9480b98ed1b0dcaff3ff09421b'
MODES = ('original', 'bc', 'weighted')
MIRROR = ('original', 'bc', 'weighted', 'weighted', 'bc', 'original')
PAIRS = (('original','weighted'), ('bc','weighted'), ('original','bc'))
WORK_FIELDS = ('candidates','depth','rollouts','sorted_depths','native_aborted','single')
sha = lambda data: hashlib.sha256(data).hexdigest()


def read_public(frame, seat):
    shared, per = frame[0]['observation'], frame[seat]['observation']
    assert per['player']==seat
    return dict(step=shared['step'],player=seat,players=shared['players'],trees=shared['trees'],
                carbon=per['carbon'],remainingOverageTime=per['remainingOverageTime'])


def choose(card, frames):
    terrain, seat = card['terrain'], card['seat']
    eligible=[t for t in range(24,min(299,len(frames)-1))
              if any(w[2]=='PLANTER' for w in frames[t][0]['observation']['players'][seat][2].values())]
    token=lambda t:sha(f'planter-native/{terrain}/{t}'.encode())
    return len(eligible), sorted(eligible,key=token)[:2]


def snapshots(card, reference):
    raw=(ROOT/card['path']).read_bytes(); assert sha(raw)==card['sha256']
    replay=json.loads(gzip.decompress(raw)); frames=replay['steps']; seat=card['seat']
    n, picked=choose(card,frames); prev=None; tracker=reference.BeliefTracker(reference._Bel,seat)
    cfg={k:v for k,v in replay['configuration'].items() if k!='randomSeed'}
    for t in range(max(picked,default=-1)+1):
        obs=read_public(frames[t],seat); assert obs['step']==t
        if t in picked:
            yield n,obs,cfg,copy.deepcopy(tracker),prev
        tracker.step(obs['players'],t,obs['carbon']); tracker.set_actions(obs['players'],frames[t+1][seat]['action'] or {})
        prev=obs['players']


def numeric(m,heads,obs,tr,prev):
    import numpy as np
    # Reference native forward instrumented to expose hf. It is never used to issue an order.
    seat,t=obs['player'],obs['step']; ps=obs['players']; cells=[v[0] for v in ps[seat][2].values() if v[2]=='PLANTER']
    est,known=copy.deepcopy(tr).step(ps,t,obs['carbon']); ref=m['encoder']; x=ref._quant(ref.encode(ps,t,seat,est,known,prev))
    h,lw,lb=ref._POLICY.forward(x,ref._SID); normal_lw,normal_lb=m['original']._POLICY.forward(x,ref._SID)
    assert np.array_equal(lw,normal_lw) and np.array_equal(lb,normal_lb)
    untouched=np.ones(225,bool); untouched[cells]=False; result={}
    for mode in ('bc','weighted'):
        plugin=m[mode]._POLICY; sid=m[mode]._SID
        off_lw,off_lb=plugin.forward(x,sid); on,base_logits=plugin.forward(x,sid,cells)
        assert np.array_equal(off_lw,lw) and np.array_equal(off_lb,lb) and np.array_equal(base_logits,lb)
        assert np.array_equal(on[untouched],lw[untouched]); expected=lw[cells]+1.25*np.tanh((h[cells]@heads[mode+'_W']+heads[mode+'_b'])/1.25)
        err=float(np.max(np.abs(on[cells]-expected))); assert np.isfinite(on).all() and err<2e-5
        result[mode]=dict(max_forward_abs_error=err,max_delta=float(np.abs(on[cells]-lw[cells]).max()),
                          planter_argmax_changed=int((on[cells].argmax(1)!=lw[cells].argmax(1)).sum()))
    return dict(planter_cells=cells,frozen_features_checked=True,per_mode=result)


def pair(rows,left,right):
    a,b=rows[left]; x,y=rows[right]; four=(a,b,x,y)
    def equals(seq,fields):
        return all(all(r[k]==seq[0][k] for r in seq[1:]) for k in fields)
    stable=all(equals(p,('issued','act0','menu','opponent_samples')) for p in ((a,b),(x,y)))
    traced=all(r['depth']>0 and len(r['opponent_samples'])==1 and r['menu'] for r in four)
    matched=traced and equals(four,WORK_FIELDS+('opponent_samples',))
    joint=a['issued']!=x['issued']; a0=a['act0']!=x['act0']; menu=a['menu']!=x['menu']
    return dict(stable=stable,matched_work_and_enemy=bool(matched),a0_changed=a0,menu_changed=menu,
                issued_changed=joint,attributed=bool(stable and matched and joint and (a0 or menu)))


def iterate(core):
    os.sched_setaffinity(0,{core}); os.environ['CARBON_DEBUG']='1'
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
    import numpy as np
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    sys.path.insert(0,str(HERE.parent/'opponent_policy/search_probe'))
    from full_search_check import root_call
    from package import build_all, PARENT
    protocol=(HERE/'protocol.json').read_bytes(); assert sha(protocol)==PROTOCOL_SHA
    identity=build_all(HERE/'head.npz',HERE); assert identity['head_sha256']==HEAD_SHA
    parent=PARENT.read_bytes(); site=b'        return lw.T, lb.T'; assert parent.count(site)==1
    blobs={'original':parent,'bc':(HERE/'bc.py').read_bytes(),'weighted':(HERE/'weighted.py').read_bytes(),
           'encoder':parent.replace(site,b'        return hf.T, lw.T, lb.T')}
    mods={}
    for name,blob in blobs.items():
        mod=types.ModuleType('planter_public_'+name); exec(compile(blob,name,'exec'),mod.__dict__); mods[name]=mod
    for mode in MODES:
        m=mods[mode]; assert m._S_ABORT==.58 and m._S_D2==8 and m._S_K==32 and m._TEMP==1 and m._SEED is None
        assert m._net() is not None
    cost=[sum(mods[n]._S_COST[i] for n in MODES)/len(MODES) for i in range(4)]
    assert all(np.isfinite(v) and v>0 for v in cost)
    with np.load(HERE/'head.npz',allow_pickle=False) as z: heads={k:z[k] for k in z.files}
    roots=[]; held=[c for c in json.loads(protocol)['tickets'] if c['split']=='held']
    for card in held:
        for n,obs,cfg,tracker,prev in snapshots(card,mods['original']):
            t=obs['step']; seed=int(sha(f"planter-root-rng/{card['terrain']}/{t}".encode())[:8],16)
            root={k:card[k] for k in ('terrain','seat','match','version','opponent','path','sha256')}
            root.update(step=t,eligible_on_terrain=n,selection_sha256=sha(f"planter-native/{card['terrain']}/{t}".encode()),
                        common_rng_seed=seed,numeric=numeric(mods,heads,obs,tracker,prev))
            for tier,k,d in (('deployment',32,8),('restricted',4,4)):
                calls={mode:[] for mode in MODES}; pp=(len(roots)%3)
                first=MODES[pp:]+MODES[:pp]; order=first+first[::-1]
                for tag in order:
                    m=mods[tag]; m._S_K=k; m._S_D2=d
                    calls[tag].append(root_call(m,tracker,prev,obs,cfg,seed,cost))
                root[tier]=dict(k_cap=k,depth_cap=d,order=list(order),calls=calls,
                                pairs={a+'__'+b:pair(calls,a,b) for a,b in PAIRS})
            roots.append(root); print(card['match'],t,len(roots),flush=True)
    totals={}
    for tier in ('deployment','restricted'):
        totals[tier]={}
        for a,b in PAIRS:
            tag=a+'__'+b
            totals[tier][tag]=dict(collections.Counter({field:sum(row[tier]['pairs'][tag][field] for row in roots)
                                                        for field in roots[0][tier]['pairs'][tag]}))
    out=dict(schema='own-root-native-planter-readout/1',protocol_sha256=PROTOCOL_SHA,head_sha256=HEAD_SHA,identities=identity,
             verifier_sha256=sha(Path(__file__).read_bytes()),native_clock_seconds=.58,initial_native_costs=cost,
             root_count=len(roots),held_map_count=len(held),cpu=core,torch_version=torch.__version__,
             scope='Historical public observations; current public history only. Each sampled decision repeated twice per package with identical private root RNG. Deployment tier is unmodified native search. Restricted secondary diagnostic uses K cap 4 and horizon 4 on the same live 0.58s clock. Equal work requires equal achieved K/H/rollouts/per-arm depth multiset, abort and shared opponent order across all four repeats. Changes in either menu or A0 can transmit the effect. Neither tier replays the original ladder RNG nor measures terminal benefit.',
             totals=totals,roots=roots)
    dest=HERE/'native.json.gz'; raw=(json.dumps(out,sort_keys=True,allow_nan=False,separators=(',',':'))+'\n').encode()
    tmp=dest.with_suffix('.tmp'); tmp.write_bytes(gzip.compress(raw,mtime=0)); tmp.replace(dest)
    print(json.dumps(totals),sha(dest.read_bytes()),flush=True)


def matched_first(core):
    """Replay the first preordered deployment response at the exact work paid by its parent."""
    os.sched_setaffinity(0,{core}); os.environ['CARBON_DEBUG']='1'
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
    import ast
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    from package import PARENT
    sys.path.insert(0,str(HERE.parent/'opponent_policy/search_probe'))
    from full_search_check import root_call
    source_file=HERE.parent/'endgame_cash_projection/response_check/run.py'
    tree=ast.parse(source_file.read_text()); helpers=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('need','control_fn')],type_ignores=[])
    space={'ast':ast}; exec(compile(helpers,str(source_file),'exec'),space); control_fn=space['control_fn']
    prior_raw=(HERE/'native.json.gz').read_bytes(); native=json.loads(gzip.decompress(prior_raw)); assert native['head_sha256']==HEAD_SHA
    protocol=json.loads((HERE/'protocol.json').read_bytes()); cards={r['terrain']:r for r in protocol['tickets'] if r['split']=='held'}
    sequence={key:i for i,key in enumerate(cards)}
    possibilities=sorted(native['roots'],key=lambda r:(sequence[r['terrain']],r['selection_sha256']))
    eligible=[r for r in possibilities if r['deployment']['pairs']['original__weighted']['issued_changed']]
    out=dict(scope='First original/weighted live-readout command change, sorted by held ticket order and registered root hash. Common native parent workload on the archived board; clock interception diagnostic, no deadline certification.',base_receipt_sha256=sha(prior_raw),control_source_sha256=sha(source_file.read_bytes()),verifier_sha256=sha(Path(__file__).read_bytes()),eligible=len(eligible),all_roots=len(possibilities),cpu=core)
    if eligible:
        row=eligible[0]; card=cards[row['terrain']]; step=row['step']; mods={}
        for mode in MODES:
            src=(PARENT if mode=='original' else HERE/(mode+'.py')).read_bytes()
            m=types.ModuleType('planter_work_'+mode); exec(compile(src,mode,'exec'),m.__dict__)
            assert m._net() is not None; m._search=control_fn(m,src); mods[mode]=m
        _,obs,cfg,tracker,prev=next(item for item in snapshots(card,mods['original']) if item[1]['step']==step)
        first=row['deployment']['calls']['original'][0]; keys=('candidates','depth','rollouts'); k,h,r=[first[z] for z in keys]
        calls={mode:[] for mode in MODES}; stable_cost=native['initial_native_costs']; rng=row['common_rng_seed']
        for mode in MIRROR:
            m=mods[mode]; m._response_cap=k; m._response_depth=h; m._S_MMAX=r
            calls[mode].append(root_call(m,tracker,prev,obs,cfg,rng,stable_cost))
        matching=all(tuple(run[z] for z in keys)==(k,h,r) for runs in calls.values() for run in runs)
        restored=all(first[z]==call[z] for call in calls['original'] for z in ('issued','act0','menu','depth_by_arm','opponent_samples','leaf_scores'))
        out.update(terrain=row['terrain'],match=row['match'],step=step,seat=row['seat'],common_rng_seed=rng,khr=[k,h,r],actual_work_matches=matching,parent_native_recovered=restored,calls=calls,pairs={a+'__'+b:pair(calls,a,b) for a,b in PAIRS},agrees_with_live={n:calls[n][0]['issued']==row['deployment']['calls'][n][0]['issued'] for n in MODES})
    dest=HERE/'native_matched.json.gz'; raw=(json.dumps(out,sort_keys=True,allow_nan=False,separators=(',',':'))+'\n').encode()
    tmp=dest.with_suffix('.tmp'); tmp.write_bytes(gzip.compress(raw,mtime=0)); tmp.replace(dest)
    print({k:v for k,v in out.items() if k!='calls'},flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cpu',type=int,required=True); ap.add_argument('--match-first',action='store_true')
    args=ap.parse_args(); (matched_first if args.match_first else iterate)(args.cpu)
