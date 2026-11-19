"""Report equal-independent-map predictive diagnostics, with repeated observations explicitly counted."""
import gzip,json,sys
from collections import Counter
from pathlib import Path
import numpy as np
from protocol import HERE,sha
from train import load
from data import generate,write
import head

def read_weights(name):
    return json.loads(gzip.decompress((HERE/name).read_bytes()))
def mean(a):return float(np.mean(a)) if len(a) else None
def predictions(w,xx):
    rows=[head.from_features(w,r) for r in xx]
    return dict(cash=np.array([r["expected_cash"] for r in rows]),
                realized=np.array([[r["cash_by_deadline"][str(h)] for h in head.HORIZONS[:-1]] for r in rows]),
                causes=np.array([[r["outcome_fraction"][k] for k in head.CATEGORIES] for r in rows]),
                time=np.array([r["expected_time_if_cash"] or 0. for r in rows]))
def metrics(rows,y,cargo,extra,old,mask):
    n=len(head.CATEGORIES);m=len(head.HORIZONS)
    truth=y[:,:2].sum(1)*cargo
    bins=y[:,n:n+m]; curve=np.cumsum(bins,axis=1)[:,:-1]*cargo[:,None]
    mom=np.sum(y[:,n+m:],axis=1)
    hit=truth>1e-8
    dtime=np.divide(mom,bins.sum(1),out=np.zeros_like(mom),where=bins.sum(1)>0)
    def filt(v):return v[mask]
    risk=extra[:,0]>1e-7
    # A partly taxed lot may retain only a fraction of principal at +10; report both counts and fractions.
    return dict(roots=int(sum(mask)),positive_future_cash_rows=int(sum(mask&hit)),
        distinct_10step_still_at_risk_rows=int(sum(mask&risk)),
        at_risk_at10_eventual_cash_fraction_per_unit=float(sum((cargo*extra[:,1])[mask])/sum((cargo*extra[:,0])[mask])) if sum((cargo*extra[:,0])[mask]) else None,
        actual_mean_terminal_cash=mean(filt(truth)),predicted_mean_terminal_cash=mean(filt(rows["cash"])),
        eventual_cash_mae=mean(filt(abs(rows["cash"]-truth))),
        terminal_recovery_fraction_mae=mean(filt(abs(rows["cash"]/cargo-truth/cargo))),
        old10_as_terminal_mae=mean(filt(abs(old-truth))),old10_as_terminal_mean_bias=mean(filt(old-truth)),
        old10_mae_against_own_original_10step_target=mean(filt(abs(old-cargo*extra[:,2]))),
        new10_mae_against_all_own_10step=mean(filt(abs(rows["realized"][:,3]-curve[:,3]))),
        old10_mae_against_all_own_10step=mean(filt(abs(old-curve[:,3]))),
        cumulative_cash_mae={str(h):mean(filt(abs(rows["realized"][:,i]-curve[:,i]))) for i,h in enumerate(head.HORIZONS[:-1])},
        cash_weighted_time_mae_steps=float(sum((truth*abs(rows["time"]-dtime))[mask])/sum(truth[mask])) if sum(truth[mask]) else None,
        monetary_loss_event_rows=int(sum(mask & (y[:,2:5].sum(1)>1e-7))),
        cause_calibration={k:dict(actual_fraction=mean(filt(y[:,i])),predicted_fraction=mean(filt(rows["causes"][:,i])),mae=mean(filt(abs(y[:,i]-rows["causes"][:,i])))) for i,k in enumerate(head.CATEGORIES)})
def run():
    generate("held_maps")
    x,xa,y,ext,old,keys,ids,seats=load("held_maps"); w=read_weights("weights.json.gz");base=read_weights("distance_control.json.gz")
    models={"own_action":predictions(w,xa),"state_only":predictions(w,x),"distance_control":predictions(base,x)}
    result={"protocol_sha256":sha(HERE/"protocol.json"),"weights_sha256":sha(HERE/"weights.json.gz"),"scores":{},"root_examples":[]}
    seeds=np.unique(ids)
    for name,estimate in models.items():
        by_map={str(i):metrics(estimate,y,x[:,0],ext,old,ids==i) for i in seeds}
        macro={}
        for k,v in next(iter(by_map.values())).items():
            if isinstance(v,(float,int)) and k not in ("roots","positive_future_cash_rows","monetary_loss_event_rows","distinct_10step_still_at_risk_rows"):
                macro[k]=mean([vs[k] for vs in by_map.values() if vs[k] is not None])
        result["scores"][name]=dict(by_map=by_map,map_macro=macro)
    for seed,s,t,u in ((621474120,0,195,"player-0-worker-44"),(621474120,1,242,"player-1-worker-28"),(621474121,1,82,"player-1-worker-16")):
        matches=np.where((ids==seed)&(seats==s)&(keys[:,0]==str(t))&(keys[:,1]==u))[0];j,=matches
        outcome=head.from_features(w,xa[j])
        result["root_examples"].append(dict(seed=seed,seat=s,t=t,uid=u,cargo=x[j,0],action=keys[j,2],actual_eventual_cash=y[j,:2].sum()*x[j,0],
            actual_ten_step_cash=y[j,len(head.CATEGORIES):len(head.CATEGORIES)+4].sum()*x[j,0],
            actual_cash_time=float(sum(y[j,len(head.CATEGORIES)+len(head.HORIZONS):])/sum(y[j,:2])),
            old_ten_step_cash=old[j],new=outcome))
    write(HERE/"predictive_evaluation.json",result)
if __name__=="__main__":run()
