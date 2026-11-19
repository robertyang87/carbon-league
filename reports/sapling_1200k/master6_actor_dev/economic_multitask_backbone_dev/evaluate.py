"""Compare deployed policies on disjoint maps and names, with retained rows and a train-only financial null."""
from collections import Counter,defaultdict
import argparse
import gzip,json,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from fit import HERE,BROAD,bc,sha,targets,write
import standalone_serving as solo

def summarize(rows):
    games=[];by_role=defaultdict(Counter)
    for card,episode in rows:
        count=Counter();roles=defaultdict(Counter)
        for r in episode['roots']:
            pc=r['arms']['parent']['actual_command'];cc=r['arms']['candidate']['actual_command']
            count['roots']+=1;count['joint_changed']+=pc!=cc
            for z in r['slots'].values():
                a=z['outputs']['parent'];b=z['outputs']['candidate'];truth=z['demo_canonical']
                if min(len(a['legal']),len(b['legal']))<=1:continue
                inc=Counter(eligible=1,bc_correct=int(a['actual']==truth),econ_correct=int(b['actual']==truth),changed=int(a['actual']!=b['actual']))
                count.update(inc);roles[z['role']].update(inc);by_role[z['role']].update(inc)
        if not count['eligible']:raise ValueError('no movable slots')
        games.append(dict(id=card['id'],opponent=card['opponent'],seat=card['seat'],split=card['split'],counts=dict(count),
                    delta=(count['econ_correct']-count['bc_correct'])/count['eligible']))
    res={}
    for split in ('val','holdout','registration_holdout'):
        subset=[g for g in games if g['split']==split];tot=sum((Counter(g['counts']) for g in subset),Counter())
        names=sorted(set(g['opponent'] for g in subset))
        byname={k:float(np.mean([g['delta'] for g in subset if g['opponent']==k])) for k in names}
        res[split]=dict(maps=len(subset),opponents=len(names),**tot,
            pooled_delta=(tot['econ_correct']-tot['bc_correct'])/tot['eligible'],
            map_equal_delta=float(np.mean([g['delta'] for g in subset])),equal_name_delta=float(np.mean(list(byname.values()))),
            by_name=byname)
    return dict(primary=res,all_roles={k:dict(v) for k,v in by_role.items()},per_map=games)

def eval_arms(pure):
    ctl=BROAD/('standalone.py' if pure else 'candidate.py');tr=HERE/('standalone.py' if pure else 'candidate.py')
    if pure and not tr.exists():bc.common.pack.main(str(HERE/'candidate.npz'),str(tr),2)
    if pure:
        arms={k:bc.imported(path,'economic_solo_'+k) for k,path in [('parent',ctl),('candidate',tr)]}
        if not arms['parent']._OPTS==arms['candidate']._OPTS or arms['parent']._SRC_SHA256!=arms['candidate']._SRC_SHA256:raise ValueError('standalone contract differs')
        feat=sys.modules['cnet2_feat']
        def step(card,steps):return solo.game(arms,card,steps,feat)
    else:
        arms={k:bc.probe.package(path,'economic_mix_'+k) for k,path in [('parent',ctl),('candidate',tr)]}
        def step(card,steps):return bc.probe.run_game(arms,card,steps)
    panel=bc.assemble();trn=[g for g in panel if g['split']=='train']
    fresh=bc.load(BROAD/'opponent_holdout/panel.json')
    added=[dict(g,opponent_new_to_ancestor=True) for g in fresh['games']]
    if {g['opponent'] for g in trn}&{g['opponent'] for g in added}:raise ValueError('unseen opponents overlap')
    selected=[g for g in panel if g['split'] in ('val','holdout')]+added
    for key in ('id','terrain_sha256','seed'):
        if len(set(g[key] for g in trn+selected))!=len(trn+selected):raise ValueError('held maps overlap')
    steps=bc.load(BROAD/'protocol.json')['eval_steps'];rows=[]
    for i,g in enumerate(selected):
        d=step(g,steps);rows.append((g,d))
        if i%10==9:print('held deployment',pure,i+1,flush=True)
    pref='standalone' if pure else 'composite'
    raw=gzip.compress(json.dumps([r for c,r in rows],separators=(',',':'),allow_nan=False).encode(),mtime=0)
    trace=HERE/(pref+'_action_evidence.json.gz');trace.write_bytes(raw)
    write(pref+'_comparison.json',dict(control=sha(ctl),treatment=sha(tr),trace=sha(trace),**summarize(rows)))

@torch.inference_mode()
def aux_validation():
    torch.set_num_threads(1);net,_=bc.common.model(HERE/'candidate.npz');net.eval()
    rows=[g for g in bc.assemble() if g['split']!='train']
    with np.load(HERE/'offline_head.npz',allow_pickle=False) as h:
        w,b,med=(torch.from_numpy(h[k]) for k in ('w','b','train_phase_median'))
    outputs={}
    for split in ('val','holdout'):
        cards=[g for g in rows if g['split']==split];d=bc.ref.load(cards,bc.SHARDS)
        y=targets(d);p=[]
        for start in range(0,len(y),16):
            bb=bc.common.batch(d,np.arange(start,min(start+16,len(y))))
            _,ctx,_=net.encode(bb);p.append(ctx@w.T+b)
        predicted=torch.cat(p);phase=np.minimum(3,d['step']//75)
        model_loss=F.smooth_l1_loss(predicted,y,reduction='none');base_loss=F.smooth_l1_loss(med[phase.astype(int)],y,reduction='none')
        outputs[split]=dict(maps=len(cards),roots=len(y),aux_huber=float(model_loss.mean()),phase_median_huber=float(base_loss.mean()),
                            by_head_aux=model_loss.mean(0).tolist(),by_head_median=base_loss.mean(0).tolist())
    write('future_economy_heldout.json',outputs)

def finish():
    ours=bc.load(HERE/'fit.json');control=bc.load(BROAD/'fit.json')
    if (control['epoch_order_hash']!=ours['sample_order_sha256'] or control['train_roots']!=ours['roots'] or
       control['updates']!=ours['steps'] or control['init_sha256']!=ours['parent']):raise ValueError('unmatched control')
    aux=bc.load(HERE/'future_economy_heldout.json')['val'];act=bc.load(HERE/'composite_comparison.json')['primary']
    with np.load(HERE/'candidate.npz',allow_pickle=False) as a,np.load(BROAD/'candidate.npz',allow_pickle=False) as b:
        diffs={k:int(np.count_nonzero(a[k]!=b[k])) for k in a.files if k!='meta'}
    gate=bool(act['val']['pooled_delta']>0 and act['registration_holdout']['pooled_delta']>0 and aux['aux_huber']<aux['phase_median_huber'])
    write('decision.json',dict(native_gate=gate,delta_weights_vs_control=sum(diffs.values()),weight_difference_by_tensor=diffs,
                     val_pooled_actions=act['val']['pooled_delta'],held_opponent_pooled_actions=act['registration_holdout']['pooled_delta'],
                     validation_economy_huber=aux['aux_huber'],training_phase_null_huber=aux['phase_median_huber'],
                     held_opponent_equal_name_actions=act['registration_holdout']['equal_name_delta'],control_fit_sha256=sha(BROAD/'fit.json')))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['aux','composite','standalone','finish']);a=p.parse_args()
    if a.mode=='aux':aux_validation()
    elif a.mode=='finish':finish()
    else:eval_arms(a.mode=='standalone')
