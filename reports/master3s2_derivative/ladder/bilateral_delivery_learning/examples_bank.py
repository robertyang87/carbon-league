"""Inspect already-exposed published decisions. Hypothetical futures are explicitly qualified."""
import gzip
import json
import numpy as np
from model import predict
from register import H,read,dump,sha
from grade import held_report
from ryan_example import real_examples
from v1_example import corrected_example,alphacarbon_279

def main():
    model=read(H/'bank_weights.json')
    examples=[corrected_example(r) for r in real_examples()]+[alphacarbon_279()]
    results=[]
    for r in examples:
        p,s,t=r['players'],r['seat'],r['step']
        out={k:v for k,v in r.items() if k not in ('players','alternatives','branches')}
        out['predictions10']=predict(p,s,t,r['order'],model)
        out['alternatives']=[dict(**a,predictions10=predict(p,s,t,a['order'],model)) for a in r.get('alternatives',r.get('branches',[]))]
        results.append(out)
    dump(H/'bank_examples.json',dict(weights=sha((H/'bank_weights.json').read_bytes()),rows=results))
    old=json.loads(gzip.decompress((H/'held_predictions.json.gz').read_bytes()))['rows']
    new=json.loads(gzip.decompress((H/'bank_held_predictions.json.gz').read_bytes()))['rows']
    lookup={(r['match'],r['step']):r for r in new};assert len(lookup)==len(old)
    paired=[dict(r,target=lookup[r['match'],r['step']]['target']) for r in old]
    changed=[(r['match'],r['step'],r['target'],o['target']) for o,r in zip(old,paired) if any(abs(v-u)>1e-7 for v,u in zip(r['target'],o['target']))]
    digest=dump(H/'bank_regrade.json',dict(changed_narrow_to_inclusive=changed,old_network_on_bank_receipts=held_report(paired)))
    print('subset',len(changed),digest)
if __name__=='__main__':main()
