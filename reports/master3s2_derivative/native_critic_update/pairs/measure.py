"""Score two native critics on exactly the same historical H8 leaf tensors."""
import base64
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
DER=HERE.parents[1]
SOURCE=DER/'root_rotation_policy/terminal'
PARENT=DER/'advantage_policy_update/v2_bc.py'
PARENT_SHA='cd230d40a08f7dea07e261202b865f951b88b3b030febc9828627aa5f47c7d2a'
PAIRS_SHA='2f9308d124b31ec142938f53e97b2b03c910d2d5d5f22a4a49a3cb32ca1d2639'
sha=lambda raw: hashlib.sha256(raw).hexdigest()

def read(path):
    raw=path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix=='.gz' else raw)

def freeze():
    assert sha(PARENT.read_bytes())==PARENT_SHA and sha((SOURCE/'results.json').read_bytes())==PAIRS_SHA
    cards=[]
    for r in read(SOURCE/'results.json')['records']:
        keys={k:r[k] for k in ('group','seed','seat','step','terrain')}
        src=DER/'action_rank_data'/r['group']/('_'.join(str(r[k]) for k in ('seed','seat','step'))+'_leaf.npz')
        if r['group'] in ('train','held'):
            assert src.is_file()
            root=Path(str(src).replace('_leaf.npz','_root.json.gz'))
            keys.update(status='eligible',input=str(src.relative_to(DER)),root=str(root.relative_to(DER)),
                        input_sha=sha(src.read_bytes()),root_sha=sha(root.read_bytes()))
        else:
            keys['status']='no archived native two-arm H8 tensors linked to this exact registered order/trajectory'
        cards.append(keys)
    assert len(cards)==25 and sum(x['status']=='eligible' for x in cards)==11
    return dict(parent_sha=PARENT_SHA,source_pairs_sha=PAIRS_SHA,labels_exposed=True,
        calibration='V estimates the expectation of native terminal(margin)=clip(margin/3000,−2.5,2.5)+0.5*sign(margin). Compare each branch and each B−A against that target. Also report 3000 times the pair error for pairs of same observed outcome sign and unclipped margins only, where the WDL constants cancel; an approximate scale on this subset, not an independently trained margin regression.',
        comparison='Exactly 11 known pairs with signed H8 tensors, archived arms A/B in native rank order. Same PARENT torso, X/scalars; no new search, suffix or unobserved candidates. Use each critic winner on each pair, exact ties retain A. 10,000 equal-map bootstrap draws seed 183450; show both all candidate pairs, same-input pairs, & WDL diagnostic eligibility. All 25 source maps are reserved from this training and validation per training split.',
        cards=cards)

def run(weights):
    p=HERE/'protocol.json';lock=read(p)
    assert lock==freeze()
    training=read(HERE.parent/'protocol.json')
    ids={r['seed'] for r in lock['cards']}; maps={r['terrain'] for r in lock['cards']}
    assert ids<=set(training['excluded_seeds']) and maps<=set(training['excluded_terrain'])
    assert all(t['seed'] not in ids and t['terrain'] not in maps for t in training['tickets'])
    model_spec=importlib.util.spec_from_file_location('pair_v4',PARENT);m=importlib.util.module_from_spec(model_spec);model_spec.loader.exec_module(m)
    import torch;torch.set_num_threads(1)
    def load_head(buf):
        with np.load(io.BytesIO(buf),allow_pickle=False) as z:
            arrays={k:z[k] for k in z.files}
        assert all(x.dtype==np.float16 and np.isfinite(x).all() for x in arrays.values())
        return {k:np.asarray(v,dtype=np.float32) for k,v in arrays.items()}
    parent_head=load_head(base64.b64decode(m._VALUE));heads=[parent_head]
    if weights:
        import ast
        blob=(HERE.parent/'candidate.py').read_bytes()
        values=[n.value.value for n in ast.parse(blob).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_VALUE' for t in n.targets)]
        assert len(values)==1 and base64.b64decode(values[0])==weights.read_bytes()
        child=load_head(weights.read_bytes());assert child.keys()==parent_head.keys() and all(a.shape==child[k].shape for k,a in parent_head.items())
        assert any(not np.array_equal(v,child[k]) for k,v in parent_head.items())
        heads.append(child)
    torch_nets=[m._TorchNet(m._POLICY,v) for v in heads]
    winos=[m._WinoNet(net) for net in torch_nets]
    archived={(r['seed'],r['seat'],r['step']):r for r in read(SOURCE/'results.json')['records']}
    rows=[]
    for c in lock['cards']:
        if c['status']!='eligible': continue
        key=tuple(c[k] for k in ('seed','seat','step'));r=archived[key];meta=read(DER/c['root'])
        assert [a['order'] for a in meta['arms']]==r['options']
        with np.load(DER/c['input'],allow_pickle=False) as d:
            assert np.all(d['depth']==8) and not np.any(d['exact_terminal'])
            X=d['X'][:,0].astype(np.float32)*np.float32(.01);S=d['scalars'][:,0]
            native=d['critic'][:,0].copy()
        assert [sha(x.tobytes()) for x in X]==[a['leaves'][0]['x_float32_sha256'] for a in meta['arms']]
        torch_vs=[net(X,m._SID,S)[2] for net in torch_nets]
        win_vs=[net(X,m._SID,S)[2] for net in winos]
        error=np.max(np.abs(win_vs[0]-native))
        assert np.allclose(win_vs[0],native,rtol=1e-4,atol=1e-4),(key,native,win_vs[0])
        assert all(np.allclose(a,b,rtol=1e-4,atol=1e-4) for a,b in zip(torch_vs,win_vs))
        margins=r['margins'];truth=[m._terminal(v) for v in margins];pred=[v.tolist() for v in win_vs]
        picks=[int(v[1]>v[0]) for v in pred]
        rows.append(dict(**{k:c[k] for k in ('group','seed','seat','step','terrain')},margins=margins,
                         target=truth,values=pred,choose=picks,original_value=native.tolist(),same_X=np.array_equal(X[0],X[1]),
                         same_scalars=np.array_equal(S[0],S[1]),parent_native_error=float(error),
                         torch_wino_error=max(float(np.abs(a-b).max()) for a,b in zip(torch_vs,win_vs))))
    assert len({r['terrain'] for r in rows})==len(rows)==11
    return dict(protocol_sha=sha(p.read_bytes()),weights_sha=sha(weights.read_bytes()) if weights else None,rows=rows)

def measure(rows):
    n_net=len(rows[0]['choose'])
    def est(arr):
        a=np.asarray(arr);bootstrap=np.random.default_rng(183450).choice(a,(10000,len(a)),replace=True).mean(1)
        return dict(mean=float(a.mean()),percentile95=np.quantile(bootstrap,[.025,.975]).tolist())
    fit=[]
    for idx in range(n_net):
        arm_error=[];pair_error=[];correct=0
        for r in rows:
            pred=np.asarray(r['values'][idx]);actual=np.asarray(r['target'])
            arm_error.append(np.abs(actual-pred).mean())
            pair_error.append(abs(float((actual[1]-actual[0])-(pred[1]-pred[0]))))
            correct+=r['choose'][idx]==int(r['margins'][1]>r['margins'][0])
        subset=[j for j,r in enumerate(rows) if np.sign(r['margins'][0])==np.sign(r['margins'][1]) and max(map(abs,r['margins']))<7500]
        fit.append(dict(branch_native_target_MAE=est(arm_error),pair_B_minus_A_native_target_MAE=est(pair_error),
                        pair_margin_equivalent_MAE=est([3000*pair_error[j] for j in subset]) if subset else None,
                        margin_subset_n=len(subset),higher_margin_chosen=correct))
    stats=dict(n_maps=len(rows),exact_same_X_and_scalars=sum(r['same_X'] and r['same_scalars'] for r in rows),
          both_arms_same_WDL=sum(np.sign(r['margins'][0])==np.sign(r['margins'][1]) for r in rows),models=fit)
    if n_net==2:
        change=[float(r['margins'][r['choose'][1]]-r['margins'][r['choose'][0]]) for r in rows]
        same=[r for r in rows if r['same_X'] and r['same_scalars']]
        different=[r for r in rows if not (r['same_X'] and r['same_scalars'])]
        stats['equal_input_numerical_switches']=sum(r['choose'][0]!=r['choose'][1] for r in same)
        stats['holding_identical_inputs_at_A']=dict(chosen_margin_change=est([0. if r['same_X'] and r['same_scalars'] else v for r,v in zip(rows,change)]),
            higher_margin_chosen=[sum((0 if r['same_X'] and r['same_scalars'] else r['choose'][i])==int(r['margins'][1]>r['margins'][0]) for r in rows) for i in range(2)])
        stats['different_input']={'n':len(different),'higher_margin_chosen':[sum(r['choose'][j]==int(r['margins'][1]>r['margins'][0]) for r in different) for j in range(2)],
                                  'switches':sum(r['choose'][0]!=r['choose'][1] for r in different)}
        stats.update(action_switches=sum(r['choose'][0]!=r['choose'][1] for r in rows),favorable=sum(v>0 for v in change),
                     unfavorable=sum(v<0 for v in change),official_picked_margin_change=est(change),
                     WDL_switches=sum(np.sign(r['margins'][r['choose'][0]])!=np.sign(r['margins'][r['choose'][1]]) for r in rows))
    return stats

if __name__=='__main__':
    if sys.argv[1]=='freeze':
        (HERE/'protocol.json').write_text(json.dumps(freeze(),indent=2,allow_nan=False)+'\n')
    else:
        w=Path(sys.argv[1]) if sys.argv[1]!='parent' else None;out=run(w)
        out['summary']={group:measure([r for r in out['rows'] if group=='all' or r['group']==group]) for group in ('all','train','held')}
        path=HERE/('score.json' if w else 'parent.json')
        path.write_text(json.dumps(out,indent=2,allow_nan=False,default=lambda a:a.item())+'\n')
