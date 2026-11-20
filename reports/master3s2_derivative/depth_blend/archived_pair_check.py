"""Apply the frozen selector to the already archived official two-action branches."""
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
src=HERE.parent/'multidepth_value/one_coefficient/estimate.py'
spec=importlib.util.spec_from_file_location('verified_legacy_pairs',src)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def replay(split):
    rows,missing,reference=m.load_split(split)
    settled=[]
    for r in rows:
        value=0.75*r['d8']+0.25*r['d4']
        i=int(value<-1e-8)
        settled.append(dict(seed=r['seed'],seat=r['seat'],step=r['step'],margins=r['margins'],
                          d4=r['d4'],d8=r['d8'],blended=value,chosen=i,
                          delta=r['margins'][i]-r['margins'][0]))
    return dict(verified=len(settled),missing=missing,features_sha256=reference,
                mean_gain=sum(r['delta'] for r in settled)/len(settled),rows=settled)


if __name__=='__main__':
    output={name:replay(name) for name in ('train','held')}
    (HERE/'archived_pairs.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
