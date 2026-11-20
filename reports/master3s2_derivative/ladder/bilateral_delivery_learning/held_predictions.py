"""Export the registered actual payments and all frozen pre-action predictions."""
from pathlib import Path
import gzip
import json
from held import rows, read_bytes, sha, SETTLEMENT_SHA
from model import predict, closest

HERE = Path(__file__).resolve().parent
WEIGHTS = 'd219ebf4c1a11b87321fb15b1267955e4eb9b6782c20265c861071461c7791b0'

def save_predictions():
    weights = json.loads(read_bytes(HERE/'weights.json', WEIGHTS))
    result = []
    for r in rows():
        predictions = predict(r['players'],r['seat'],r['step'],r['order'],weights)
        item = {k:r[k] for k in ('match','seed','terrain','source_arm','seat','step','target')}
        foe=r['players'][1-r['seat']]
        loads=sorted((cargo for pos,cargo,kind in foe[2].values()
                      if kind=='COLLECTOR' and cargo>0 and closest(pos,foe[1].values())<=5),reverse=True)
        item['multi_enemy']=len(loads)>=2
        item['top2_enemy_cargo']=loads[:2]
        item['predictions'] = {k:list(map(float,v)) for k,v in predictions.items()}
        result.append(item)
    bundle = {'weights_sha256':WEIGHTS,'held_source_sha256':SETTLEMENT_SHA,
              'protocol_sha256':sha((HERE/'protocol.json').read_bytes()),'rows':result}
    compressed = gzip.compress(json.dumps(bundle,ensure_ascii=False,sort_keys=True,
                               separators=(',',':'),allow_nan=False).encode()+b'\n',mtime=0)
    out=HERE/'held_predictions.json.gz'
    if out.exists() and out.read_bytes()!=compressed:raise ValueError('Different export already present')
    if not out.exists():
        tmp=out.with_suffix('.tmp');tmp.write_bytes(compressed);tmp.replace(out)
    print(len(result),sha(compressed),[k for k in result[0]['predictions']])
    return out

if __name__=='__main__': save_predictions()
