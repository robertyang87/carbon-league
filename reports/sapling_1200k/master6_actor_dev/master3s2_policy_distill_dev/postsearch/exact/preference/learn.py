"""Reference-relative pairwise distillation on search-replaced units only."""
from pathlib import Path
import sys,os,hashlib,tempfile,json,importlib.util
import numpy as np,torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import train as exact
base=exact.base;shared=exact.shared
def run():
    assert os.sched_getaffinity(0)=={3} and os.getpriority(os.PRIO_PROCESS,0)==19
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    p=exact.dataset.read(HERE/'protocol.json');d,cards=exact.load()
    chosen=d['src_act'].clip(0).astype(np.int64);pre=d['pre_act'].clip(0).astype(np.int64)
    pair=d['eligible']&d['unit_changed']&np.take_along_axis(d['legal'],pre[...,None],axis=-1)[...,0]
    idx=exact.draws(dict(d,eligible=pair),cards,p);net,meta=base.model(shared.PARENT,shared.PARENT_HASH)
    anchor,_=base.model(shared.PARENT,shared.PARENT_HASH);anchor.eval()
    torch.manual_seed(p['seed']);params=[v for k,v in net.named_parameters() if v.requires_grad_(not k.startswith(shared.NOT_POLICY)).requires_grad]
    opt=torch.optim.AdamW(params,lr=p['lr'],weight_decay=.0001);loss=[];net.train()
    for ix in idx:
        b=base.batch(d,ix);g=net(b)[0]
        with torch.no_grad():ref=anchor(b)[0]
        po=torch.from_numpy(chosen[ix])[...,None];ne=torch.from_numpy(pre[ix])[...,None]
        def gap(v):return (v.gather(-1,po)-v.gather(-1,ne)).squeeze(-1)
        keep=torch.from_numpy(pair[ix]);same=torch.from_numpy(d['eligible'][ix]&~d['unit_changed'][ix])
        prefer=-torch.nn.functional.logsigmoid((gap(g)-gap(ref))[keep]).mean()
        kl=(ref.softmax(-1)*(ref.log_softmax(-1)-g.log_softmax(-1))).sum(-1)[same].mean()
        value=prefer+3*kl;opt.zero_grad(set_to_none=True);value.backward();torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True)
        opt.step();loss.append([float(prefer.detach()),float(kl.detach())])
    net.eval();output=HERE/'preference.npz';base.trainer.export(net,str(output),dict(meta,fit='native search replacement paired preference'))
    with tempfile.TemporaryDirectory() as td:
        staged=Path(td)/output.name;staged.write_bytes(output.read_bytes());pkg=Path(td)/'preference.py';base.pack.main(str(staged),str(pkg),teacher=2);(HERE/pkg.name).write_bytes(pkg.read_bytes())
    h=exact.dataset.sha
    shared.dump(HERE/'fit.json',dict(source=h(Path(__file__)),protocol=h(HERE/'protocol.json'),manifest=h(HERE.parent/'manifest.json'),parent=shared.PARENT_HASH,
                 draws=idx.tolist(),counts={'pair_examples':int(pair[idx].sum()),'roots':idx.size,'distinct_roots':len(np.unique(idx))},loss=loss,pack=h(HERE/'preference.py'),weights=h(output)))
if __name__=='__main__':run()
