"""Train the attached team decision layer on disjoint maps and only then evaluate complete signed actions."""
from collections import Counter,defaultdict
from copy import deepcopy
import gzip,json,hashlib
import numpy as np
import torch
from torch import nn
from data import D,read,sha,compact


class Economic(nn.Module):
    def __init__(self,n):
        super().__init__(); self.l1=nn.Linear(n,48);self.l2=nn.Linear(48,16);self.aux=nn.Linear(16,5)
    def forward(self,x):
        h1=torch.tanh(self.l1(x));h2=torch.tanh(self.l2(h1));aux=self.aux(h2)
        return torch.cat([h1,h2,aux],dim=-1)


def unit(x):
    return torch.from_numpy(x.astype(np.float32))


def statistics(pred,y):
    return dict(n=len(y),mae=np.abs(pred-y).mean(0).tolist(),constant_zero_mae=np.abs(y).mean(0).tolist(),
                rmse=np.sqrt(((pred-y)**2).mean(0)).tolist(),constant_zero_rmse=np.sqrt((y**2).mean(0)).tolist(),
                realized_changes=(y!=0).sum(0).tolist(),sign_accuracy_given_realized_change=[float(np.mean((pred[y[:,j]!=0,j]>0)==(y[y[:,j]!=0,j]>0))) if np.any(y[:,j]) else None for j in range(y.shape[1])])


def objective_rows(rows,score):
    out=[]
    for r in rows:
        b,a=r['parent'],r['alternate'];lo,hi=r['menu'];signed_diff=score[a]-score[b]
        ix=lo+int(np.argmax(score[lo:hi])); ix=b if score[ix]-score[b]<100 else ix
        signed_alt=bool(signed_diff>=100)
        match='parent' if ix==b else ('alternative' if ix==a else 'unlabeled')
        actual=float(r['delta'][0]-r['delta'][1])
        out.append(dict(seed=r['seed'],terrain=r['terrain'],seat=r['seat'],t=r['t'],role=r['role'],family=r['family'],case=r['case'],split=r['split'],
                        native_alt_in_menu=r['in_menu'],signed_prediction=float(signed_diff),signed_choose_alternate=signed_alt,
                        official_alt_minus_parent=actual,own_alt_minus_parent=r['delta'][0],rival_alt_minus_parent=r['delta'][1],
                        chosen_joint=match,selected_role=r['menu_roles'][ix-lo],chosen_command=r['menu_commands'][ix-lo],actual_known_delta=actual if match=='alternative' else (0. if match=='parent' else None)))
    return out


def report(rows):
    if not rows:return {'roots':0}
    m={r['terrain'] for r in rows}
    means=[]; bymap={}
    for terrain in sorted(m):
        rr=[r for r in rows if r['terrain']==terrain];v=float(np.mean([r['official_alt_minus_parent'] if r['signed_choose_alternate'] else 0 for r in rr]));means.append(v);bymap[terrain]=v
    z=[r for r in rows if r['signed_choose_alternate']];confirmed=[r for r in rows if r['chosen_joint']=='alternative']
    k=[r for r in rows if r['actual_known_delta'] is not None];winning=[r for r in confirmed if r['actual_known_delta']>0]
    rng=np.random.default_rng(79103);v=np.asarray(means);ci=np.quantile(v[rng.integers(0,len(v),(2000,len(v)))].mean(1),[.025,.975]).tolist()
    return dict(roots=len(rows),terrains=len(m),roles=dict(Counter(r['role'] for r in rows)),offered_alternate_in_menu=sum(r['native_alt_in_menu'] for r in rows),
                signed_changed=len(z),signed_changed_roles=dict(Counter(r['role'] for r in z)),signed_changed_maps=len({r['terrain'] for r in z}),
                signed_map_mean_margin_delta=float(v.mean()),signed_per_map=bymap,signed_map_bootstrap_ci95=ci,
                signed_winner_accuracy=float(np.mean([(r['signed_prediction']>0)==(r['official_alt_minus_parent']>0) for r in rows if r['official_alt_minus_parent']!=0])),
                menu_decision_identity=dict(Counter(r['chosen_joint'] for r in rows)),menu_matched_changed_roles=dict(Counter(r['role'] for r in confirmed)),
                menu_matched_changed_winning_maps=len({r['terrain'] for r in winning}),menu_matched_changed_winning_roles=sorted({r['role'] for r in winning}),
                known_menu_cases=len(k),known_menu_subset_mean_margin_delta=float(np.mean([r['actual_known_delta'] for r in k])) if k else None,
                menu_true_signed_gain_compare_prohibits_unmatched=True)


def main(aux_only=False):
    torch.manual_seed(70431);torch.set_num_threads(1)
    reg=read(D/'source_registry.json');sg=np.array([r['split']=='train' for r in reg['games']]); chunks=[np.load(D/f'aux_{i}.npz') for i in (0,1)]
    get=lambda k:np.concatenate([z[k] for z in chunks]); ii=get('map_idx');tr=sg[ii];b=get('b');a=get('a');y=get('target')
    mu=np.concatenate([a[tr],b[tr]]).mean(0);sd=np.maximum(.05,np.concatenate([a[tr],b[tr]]).std(0))
    norm=lambda z:unit(np.clip((z-mu)/sd,-8,8))
    xx,xb=norm(a),norm(b); sy=np.maximum([1,1,1,.2,.2],y[tr].std(0)).astype('f'); yy=unit(y/sy)
    model=Economic(len(mu));control=deepcopy(model); opt=torch.optim.Adam(model.parameters(),lr=.003)
    rng=np.random.default_rng(70431);losses=[]
    checkpoints=D/'aux_weights.npz'; prior=D/'aux_training.json'
    identities={f:sha(D/f) for f in ('aux_0.npz','aux_1.npz','source_registry.json','protocol.json','actions.py')}
    if checkpoints.exists() and prior.exists():
        receipt=read(prior)
        if receipt['sources']!=identities or receipt['weights_sha256']!=sha(checkpoints):
            raise ValueError('auxiliary source identity changed')
        losses=receipt['training_losses']
        with np.load(checkpoints) as archive:model.load_state_dict({k:torch.from_numpy(archive[k]) for k in archive.files})
    else:
        for epoch in range(12):
            ids=rng.permutation(np.flatnonzero(tr));loss=0.
            for ix in np.array_split(ids,max(1,(len(ids)+127)//128)):
                z=model(xx[ix])[:,-5:]-model(xb[ix])[:,-5:];ls=((z-yy[ix])**2).mean();opt.zero_grad();ls.backward();opt.step();loss+=float(ls.detach())*len(ix)
            losses.append(loss/len(ids))
        np.savez_compressed(checkpoints,**{k:v.detach().numpy() for k,v in model.state_dict().items()})
        with torch.no_grad():hp=(model(xx[~tr])[:,-5:]-model(xb[~tr])[:,-5:]).numpy()*sy
        prior.write_text(json.dumps(dict(weights_sha256=sha(checkpoints),sources=identities,training_losses=losses,
            training_rows=int(tr.sum()),held_rows=int((~tr).sum()),held=statistics(hp,y[~tr])),indent=2)+'\n')
    with torch.no_grad():pred=(model(xx)[:,-5:]-model(xb)[:,-5:]).numpy()*sy
    if aux_only:return
    n=read(D/'native_inputs.json.gz');nf=np.load(D/'native_inputs.npz')['features'];rr=n['rows']
    nt=np.asarray([r['split']=='train' for r in rr]);nb=np.asarray([r['parent'] for r in rr]);na=np.asarray([r['alternate'] for r in rr]);ty=np.array([r['delta'] for r in rr])
    assert not ({r['terrain'] for r in rr if r['split']=='held'} & {r['terrain'] for r in reg['games'] if r['split']=='train'})
    model_reports={}; stored=None;actual=[]
    for name,net in [('trained',model),('no_aux_gradient_control',control)]:
        with torch.no_grad():enc=net(norm(nf)).numpy().astype('d')
        dx=enc[na]-enc[nb]; scale=np.maximum(dx[nt].std(0),.05)
        des=dx[nt]/scale; ridge=np.linalg.solve(des.T@des+60*np.eye(des.shape[1]),des.T@ty[nt])
        score=(enc/scale)@ridge; margin=score[:,0]-score[:,1]
        rows=objective_rows(rr,margin);actual+= [{**row,'model':name} for row in rows]
        model_reports[name]={s:report([r for r in rows if r['split']==s]) for s in ('train','held')}
        model_reports[name]['held_by_role']={s:report([r for r in rows if r['split']=='held' and r['role']==s]) for s in ['base','collector','planter','mixed']}
        if name=='trained':
            stored=dict(mu=mu,sd=sd,term_scale=scale,term=ridge,sy=sy)
            for k,v in net.state_dict().items():stored[k.replace('.','_')]=v.detach().numpy()
    np.savez_compressed(D/'weights.npz',**stored)
    ro=get('role'); tt=get('t')
    def binrows(mask):return {'branches':int(mask.sum()),'roles':dict(Counter(ro[mask])),'sampled_turns':sorted(int(i) for i in set(tt[mask])),
                              'terrain_count':len({reg['games'][i]['terrain'] for i in ii[mask]})}
    h=model_reports['trained']['held']
    gates=dict(at_least_10_held_roots=h['roots']>=10,two_held_maps=h['terrains']>=2,three_changes=h['signed_changed']>=3,
               two_roles=len(h['signed_changed_roles'])>=2,positive_paired_map_margin=h['signed_map_mean_margin_delta']>0,
               production_complete_winning_arm_two_maps=h['menu_matched_changed_winning_maps']>=2,
               production_complete_winning_arm_two_roles=len(h['menu_matched_changed_winning_roles'])>=2)
    out=dict(weights_sha256=sha(D/'weights.npz'),protocol_sha256=sha(D/'protocol.json'),source_registry_sha256=sha(D/'source_registry.json'),
             signed_sha256=sha(D/'signed_roots.json.gz'),train=binrows(tr),held=binrows(~tr),aux_train_loss=losses,
             aux_held=statistics(pred[~tr],y[~tr]),aux_targets=['own cash','rival cash','carried differential','live worker differential','tree differential'],
             new_trainable_nn_parameters=sum(p.numel() for p in model.parameters()),terminal_ridge_parameters=int(ridge.size),
             signed_excluded=n['excluded'],native=model_reports,gates=gates,decision='ELIGIBLE' if all(gates.values()) else 'STOP')
    (D/'verdict.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    (D/'decisions.json.gz').write_bytes(gzip.compress(compact(actual),mtime=0))
    print(out['decision'],json.dumps(gates),repr(h),flush=True)


if __name__=='__main__':
    import sys
    main('--aux-only' in sys.argv)
