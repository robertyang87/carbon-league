"""Observational state strata for same-ID five-step deposits, never favorable-action labels."""
import json
import numpy as np
from fit import load,equal
from model import HERE,FEATURES,Forecast


def run():
    d=load()
    with (HERE/'sources.json').open() as f:s=json.load(f)
    source=s['native']+s['ladder'];card=d['game']
    split=np.array([v['split'] for v in source])[card];terrain=np.array([v['terrain'] for v in source])[card]
    x=d['x'];idx={k:i for i,k in enumerate(FEATURES)};p=Forecast().arrays(x,d['prior'],d['group'])[1][:,1]
    # Hold cargo scale and distance range fixed. The thresholds are descriptive, never used to select training rows or commands.
    base=(x[:,idx['cargo']]>=50)&(x[:,idx['own_base_distance']]>=2)&(x[:,idx['own_base_distance']]<=4)&(x[:,idx['observed_3']]>0)
    out={}
    for label,panel in [('native_holdout',np.isin(split,['held','recent_evaluation'])),('ladder',split=='ladder')]:
        records=[]
        for advancing in (False,True):
            for safe in (False,True):
                keep=base&panel&((x[:,idx['home_progress_3']]>0)==advancing)&((x[:,idx['safe_shortest_first_steps']]>0)==safe)
                m=int(sum(keep));row=dict(n=m,maps=int(len(set(terrain[keep]))),approached_last3=advancing,has_unblocked_next_shortest_square=safe)
                if m:
                    w=equal(terrain[keep]);y=d['bank'][keep,1]
                    row.update(mean_forecast=float(np.average(p[keep],weights=w)),actual_frequency=float(np.average(y>0,weights=w)),
                               mean_same_id_cash_deposited_by_h5=float(np.average(y,weights=w)),root_mean_cargo=float(np.average(x[keep,idx['cargo']],weights=w)))
                records.append(row)
        out[label]=records
    (HERE/'return_context.json').write_text(json.dumps(dict(filter='cargo>=50 home distance 2..4 prior three-step track exists',statistics='map equal within each row, overlapping worker-timepoints; each row conditional on actual simultaneous actions',strata=out),indent=2)+'\n')


if __name__=='__main__':run()
