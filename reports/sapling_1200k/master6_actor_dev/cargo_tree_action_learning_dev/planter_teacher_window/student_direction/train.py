"""Optimize a direction-only neural readout and insert it into the actual mixed autoregressive policy."""
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from dataset import HERE,PARENT,RULE,read,write,sha
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

RUNTIME='''
def _direction_scores(combined, hidden, move):
    ctx = np.tanh(hidden @ _DIR_H + _DIR_B)
    directional = np.tanh(move[1:] @ _DIR_M + _DIR_A) @ ctx
    old = combined[1:].astype(np.float64)
    new = old + directional * .5
    peak_old, peak_new = old.max(), new.max()
    old_norm = peak_old + np.log(np.exp(old-peak_old).sum())
    new_norm = peak_new + np.log(np.exp(new-peak_new).sum())
    return np.concatenate(([float(combined[0])], new + old_norm-new_norm))

def _direction_team(x, aux, enc, enabled):
    F = sys.modules["cnet2_feat"]
    base = _return_decode(x, aux, enc, False, enabled)
    conditional = _return_decode(x, aux, enc, True, enabled)
    protected = (np.arange(F.OWN)>=10) | (x["tok_kind"][:F.OWN]==1)
    if np.any(base[protected] != conditional[protected]):
        return base
    if np.any(base != conditional) and _return_conflicts(x,conditional)-_return_conflicts(x,base):
        return base
    return conditional

def _mixed_actions(x, aux):
    enc = []
    for policy in _NETS:
        tokens, context = policy.encode(x, _HYBRID["style"])
        enc.append((policy,tokens,context,policy.lin(context,"h0")))
    original = _direction_team(x,aux,enc,False)
    if not np.any(x["tok_mask"][:10] & (x["tok_kind"][:10]==1)):
        return original
    modified = _direction_team(x,aux,enc,True)
    if _return_conflicts(x,modified)-_return_conflicts(x,original):
        return original
    return modified
'''


def bundle(path,arrays):
    src=PARENT.read_text()
    start=src.index('def _return_decode(')
    end=src.index('\ndef _mixed_actions(',start)
    prefix=src[:start]
    before=src[start:end]
    assert before.count('def _return_decode(x, aux, encoded, apply_residual):')==1
    before=before.replace('def _return_decode(x, aux, encoded, apply_residual):','def _return_decode(x, aux, encoded, apply_residual, direction=False):')
    marker='        if not apply_residual:'
    assert before.count(marker)==1
    before=before.replace(marker,'''        if direction and slot < 10 and x["tok_kind"][slot] == 1:
            assert np.all(legal)
            directional = _direction_scores(combined, states[1][3], move)
            if action != 0:
                action = int(1 + np.argmax(directional[1:]))
'''+marker)
    weights='\n'.join('_DIR_'+k+' = np.array('+repr(v.tolist())+',dtype=np.float32)'for k,v in arrays.items())
    path.write_text(prefix+before+'\n'+weights+'\n'+RUNTIME)


def main():
    assert sha(PARENT)==RULE['parent']
    rows=read(HERE/'rows.json.gz')
    f=np.load(HERE/'features.npz')
    h,m,z=[torch.from_numpy(f[k]).float() for k in ('h','m','z')]
    labels=torch.tensor([int(r['label'])-1 for r in rows],dtype=torch.long)
    assert labels.min()==0 and labels.max()==3
    y=np.array([r["return24"] for r in rows])
    train=np.array([i for i,r in enumerate(rows) if r["split"]=="train"])
    # Role-matched roots of unequal workforce size do not count bigger teams repeatedly.
    base_w=np.array([1/r["nplanters"] for r in rows])
    multi=np.ones(len(rows))
    for src in {r["source"] for r in rows}:
        ids=[i for i in train if rows[i]["source"]==src]
        med=np.median(y[ids]);scale=max(1.,1.4826*np.median(np.abs(y[ids]-med)))
        all_ids=[i for i,r in enumerate(rows) if r["source"]==src]
        multi[all_ids]=1+.75*np.tanh((y[all_ids]-med)/scale)
    arms={}
    for arm in ("control","candidate"):
        torch.manual_seed(94131)
        ws=nn.ParameterDict({"H":nn.Parameter(torch.zeros(96,12)), "B":nn.Parameter(torch.zeros(12)),
            "M":nn.Parameter(torch.randn(15,12)*.13), "A":nn.Parameter(torch.randn(4,12)*.13)})
        opt=torch.optim.Adam(ws.parameters(),lr=.002)
        weights=torch.tensor(base_w*(multi if arm=="candidate" else 1),dtype=torch.float32)
        train_losses=[]
        for epoch in range(12):
            epoch_loss=[]
            order=np.random.default_rng(94131+epoch).permutation(train)
            for batch in np.array_split(order, math.ceil(len(train)/96)):
                ctx=torch.tanh(h[batch]@ws["H"]+ws["B"])
                moves=torch.tanh(m[batch,1:]@ws["M"]+ws["A"])
                delta=.5*(moves*ctx[:,None,:]).sum(-1)
                original=z[batch,1:].log_softmax(-1)
                pred=(z[batch,1:]+delta).log_softmax(-1)
                ce=-pred[torch.arange(len(batch)),labels[batch]]
                loss=(ce*weights[batch]).sum()/weights[batch].sum()+.5*(original.exp()*(original-pred)).sum(1).mean()
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(ws.parameters(),2,error_if_nonfinite=True);opt.step()
                epoch_loss.append(float(loss.detach()))
            train_losses.append(float(np.mean(epoch_loss)))
        arrays={name:ws[name].detach().numpy() for name in ("H","B","M","A")}
        np.savez(HERE/f"{arm}.npz",**arrays)
        file=HERE/f"{arm}.py";bundle(file,arrays)
        arms[arm]=dict(package=sha(file),bytes=file.stat().st_size,weights=sha(HERE/f"{arm}.npz"),fit_loss=train_losses)
    write(HERE/"fit.json",dict(protocol=sha(HERE/"protocol.json"),parent=sha(PARENT),dataset=sha(HERE/"data.json"),
        trained_parameters=sum(x.size for x in arrays.values()),nominal_move_loss="conditional NLL",arms=arms))


if __name__=="__main__":main()
