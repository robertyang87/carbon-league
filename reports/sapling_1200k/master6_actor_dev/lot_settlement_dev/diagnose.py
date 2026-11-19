"""Post-fit accuracy of 'all lots come home' and concentration of actual failures. No refitting."""
import gzip,json
import numpy as np
from train import load
from evaluate import read_weights,predictions
from data import write
from protocol import HERE
import head

def run():
    x,xa,y,e,old,k,ids,ss=load("held_maps")
    model=predictions(read_weights("weights.json.gz"),xa)
    data={}
    for seed in sorted(set(ids)):
        m=ids==seed; positive=y[:,0]+y[:,1]; full=x[:,0]; actual=positive*full
        # Duplicated principal over time is intentional; these are prediction observations.
        failure=(positive<.99)
        heavy=(full>=100.)
        one={}
        for label,mask in [("all",m),("principal_at_least_100",m&heavy),("under_99pct_recovery",m&failure),("recover_at_least_99pct",m&~failure)]:
            ix=np.flatnonzero(mask)
            if not len(ix):continue
            one[label]=dict(n=len(ix),actual_mean_cash=float(np.mean(actual[ix])),assume_full_MAE=float(np.mean(full[ix]-actual[ix])),
                           model_MAE=float(np.mean(np.abs(model["cash"][ix]-actual[ix]))),
                           actual_unrecovered_fraction=float(np.mean(1-positive[ix])),model_unrecovered_fraction=float(np.mean(1-model["cash"][ix]/full[ix])))
        risk=1-model["cash"][m]/full[m];truth=1-positive[m];size=max(1,int(np.ceil(len(risk)*.1)))
        worst=np.argsort(-risk,kind="stable")[:size]
        one["highest_predicted_risk_decile"]=dict(n=size,actual_mean_unrecovered_fraction=float(np.mean(truth[worst])),
            predicted_mean_unrecovered_fraction=float(np.mean(risk[worst])),fraction_true_unrecovered_principal_in_bin=float(sum(truth[worst]*full[m][worst])/sum(truth*full[m])))
        data[str(seed)]=one
    from statistics import mean
    macro={k:{field:mean(d[k][field] for d in data.values() if k in d) for field in next(iter(data.values()))[k] if field!="n"} for k in next(iter(data.values()))}
    write(HERE/"full_recovery_diagnostic.json",dict(scope="Post-fit descriptive baseline; fixed predictor, same four held maps; per-decision amounts repeat the same trip across turns; not incremental team income",per_map=data,macro=macro))
if __name__=="__main__":run()
