"""One supervised, regularized update of our root prediction of the original examiner."""
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from original_policy import HERE,PROTOCOL,sha,need,dump,build

def softmax(z):
    logp=z-logsumexp(z,axis=1)[:,None]
    return logp,np.exp(logp)

def materialize(work):
    result=[]
    for c in PROTOCOL["games"]:
        with np.load(work/(c["sha256"]+".npz"),allow_pickle=False) as data:
            need(str(data["source"])==c["sha256"] and int(data["seat"])==c["candidate_seat"],"Wrong source row")
            result.append((c,{k:data[k] for k in data if k!="source"}))
    return result

def readout(train,base):
    key=str(int(base));rows=[]
    for c,d in train:
        x,z,y=[d[key+k] for k in ("x","z","y")]
        if len(y): rows.append((x,z,y,np.full(len(x),1./len(x))))
    x,z,y,m=(np.concatenate(c) for c in zip(*rows));m/=m.sum()
    rms=np.maximum(np.sqrt(m@(x*x)),.25).astype(np.float16).astype(np.float32)
    x=np.column_stack([np.tanh(x/rms),np.ones(len(x))]).astype(np.float64)
    z=z.astype(np.float64)
    prior_log,prior=softmax(z);ids=np.arange(len(y))
    def f(s):
        w=s.reshape(x.shape[1],z.shape[1]);logit=x@w+z
        lp,P=softmax(logit)
        loss=m@(-lp[ids,y]+np.sum(prior*(prior_log-lp),axis=1))+.005*s@s
        d=2*P-prior;d[ids,y]-=1
        return loss,(x.T@(d*m[:,None])).ravel()+.01*s
    opt=minimize(f,np.zeros(x.shape[1]*z.shape[1]),jac=True,method="L-BFGS-B",options={"maxiter":50,"ftol":1e-12})
    for halves in range(18):
        head=(opt.x.reshape(x.shape[1],z.shape[1])*2.**(-halves)).astype(np.float16)
        raw=x@head.astype(np.float64)
        lp,P=softmax(z+raw)
        kl=float(m@np.sum(prior*(prior_log-lp),axis=1));pct=float(np.quantile(abs(raw),.99))
        if kl<=.03 and pct<=1.25: break
    else: raise ValueError("Cannot satisfy update bounds")
    prefix="b" if base else "w"
    arrays={prefix+"w":head[:-1],prefix+"b":head[-1],prefix+"scale":rms.astype(np.float16)}
    return arrays,dict(training_slots=len(y),iterations=int(opt.nit),status=int(opt.status),halvings=halves,
         mean_KL=kl,abs_delta_p99=pct,nonzero=int(np.count_nonzero(head)),new_parameters=int(head.size))

def audit(card,data,params):
    out={}
    for label,role,typ in (("worker",None,0),("collector",0,0),("planter",1,0),("base",2,1)):
        k=str(typ);name="b" if typ else "w"
        keep=np.ones(len(data[k+"kind"]),bool) if role is None else data[k+"kind"]==role
        if not np.any(keep):continue
        x,z,y=[data[k+suffix][keep] for suffix in ("x","z","y")]
        new=z+(np.tanh(x/params[name+"scale"].astype(np.float32))@params[name+"w"].astype(np.float32)+params[name+"b"].astype(np.float32))
        old_lp,old_p=softmax(z.astype(np.float64));new_lp,new_p=softmax(new.astype(np.float64));idx=np.arange(len(y))
        metrics=dict(n=len(y),old_nll=float(-old_lp[idx,y].mean()),new_nll=float(-new_lp[idx,y].mean()),
            old_modal_hits=int((z.argmax(1)==y).sum()),new_modal_hits=int((new.argmax(1)==y).sum()),
            changed_modal=int((new.argmax(1)!=z.argmax(1)).sum()),tv=float(np.mean(.5*abs(new_p-old_p).sum(1))))
        if not typ:
            coll=data[k+"own_target"][keep]
            realized=coll[idx,y]
            metrics.update(actual_move_to_same_cell_count=int(realized.sum()),steps_with_collision_opportunity=int((coll.sum(1)>0).sum()),
              old_collision_brier=float(np.mean((realized-np.sum(old_p*coll,axis=1))**2)),
              new_collision_brier=float(np.mean((realized-np.sum(new_p*coll,axis=1))**2)))
        out[label]=metrics
    return dict(terrain=card["terrain"],seat=card["candidate_seat"],seed=card["seed"],roles=out)

def main(work):
    allgames=materialize(work);fit=[r for r in allgames if r[0]["split"]=="train"];held=[r for r in allgames if r[0]["split"]=="val"]
    params={};details={}
    for base in (False,True):
        head,score=readout(fit,base);params.update(head);details["base" if base else "worker"]=score
    np.savez(HERE/"weights.npz",**params)
    candidate=build(params);path=HERE/"candidate.py"
    need(not path.exists() or path.read_bytes()==candidate,"Release conflict")
    path.write_bytes(candidate)
    result=[audit(c,x,params) for c,x in held]
    def mean(rows,k):
        vals=[r["roles"][k]["new_nll"]-r["roles"][k]["old_nll"] for r in rows if k in r["roles"]]
        return float(np.mean(vals))
    role={k:mean(result,k) for k in ("worker","collector","planter","base")}
    def pooled(r):return (mean(r,"worker")+mean(r,"base"))*.5
    bymap={s:pooled([r for r in result if r["terrain"]==s]) for s in sorted({r["terrain"] for r in result})}
    byseat={str(s):pooled([r for r in result if r["seat"]==s]) for s in (0,1)}
    hold_pass=bool(role["worker"]<=0 and role["base"]<=0 and max(byseat.values())<=0 and
        pooled(result)<0 and sum(x<0 for x in bymap.values())>=3)
    dump(HERE/"held.json",result)
    summary=dict(source=PROTOCOL["source_master_sha256"],protocol=sha((HERE/"protocol.json").read_bytes()),
        candidate_sha256=sha(candidate),head_sha256=sha((HERE/"weights.npz").read_bytes()),source_maps=len(bymap),
        fitting_maps=len({c["terrain"] for c,_ in fit}),archive_count=len(allgames),held_sha=sha((HERE/"held.json").read_bytes()),
        fit=details,role_nll_change=role,per_map_nll_change=bymap,per_seat_nll_change=byseat,observational_gate=hold_pass,
        bytes=len(candidate))
    dump(HERE/"fit.json",summary)
    print(summary,flush=True)
if __name__=="__main__":main(Path(sys.argv[1]))
