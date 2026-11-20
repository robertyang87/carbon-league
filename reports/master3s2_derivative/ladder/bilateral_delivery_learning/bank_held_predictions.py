"""Run the separately frozen inclusive bank-credit predictors on the same own public requests."""
import gzip
import json
from pathlib import Path
import zipfile
from held import HOSTED, SETTLEMENT_SHA, RUN, STEPS, sha, read_bytes, require
from held_bank_labels import NARROW
from model import predict

HERE=Path(__file__).resolve().parent
WEIGHTS='b2c90f462344aecf50134c82c565648b39b2d4078d092da29be1f45f232e45cc'

def save_predictions():
    w=json.loads(read_bytes(HERE/'bank_weights.json',WEIGHTS))
    src=json.loads(read_bytes(HOSTED/'settlement.json',SETTLEMENT_SHA))
    v0=json.loads(gzip.decompress(read_bytes(HERE/'held_predictions.json.gz',NARROW)))
    labels=(HERE/'held_bank_labels.json.gz').read_bytes()
    lb=json.loads(gzip.decompress(labels));require(lb['source_settlement_sha256']==SETTLEMENT_SHA and lb['v0_predictions_sha256']==NARROW,'Inclusive labels differ')
    lookup={(r['match'],r['step']):r for r in lb['rows']};narrow={(r['match'],r['step']):r for r in v0['rows']};out=[]
    for ticket in src['archives']:
        seed,seat=ticket['seed'],ticket['seat'];path=HOSTED/(str(ticket['id'])+'.zip')
        require(sha(path.read_bytes())==ticket['sha256'],'Hosted ZIP differs')
        with zipfile.ZipFile(path) as arc:
            for arm in ('challenger','parent'):
                match=f'{RUN}:{seed}:{seat}:{arm}'
                views=gzip.decompress(arc.read(arm+'/observations.jsonl.gz')).splitlines()
                tape=arc.read(arm+'/turns.jsonl').splitlines()
                for t in STEPS:
                    row=json.loads(views[t]);record=json.loads(tape[t]);view=row['observations'][seat]
                    require(row['step']==record['step']==view['step']==t and view['player']==seat and
                            row['public_observation_sha256'][seat]==record['public_observation_sha256'][seat], 'Changed request')
                    label=lookup[match,t];previous=narrow[match,t]
                    require(previous['target']==label['components']['own_delivery'],'V0 target differs')
                    result=predict(view['players'],seat,t,record['calls'][seat]['response']['action'],w)
                    item={key:previous[key] for key in ('match','seed','terrain','source_arm','seat','step','multi_enemy','top2_enemy_cargo')}
                    item.update(target=label['all_bank_target'], components=label['components'], predictions=result)
                    out.append(item)
    require(len(out)==len(lookup)==len(narrow)==768,'Incomplete held rows')
    bundle={'weights_sha256':WEIGHTS,'held_source_sha256':SETTLEMENT_SHA,'bank_labels_sha256':sha(labels),
            'v0_predictions_sha256':NARROW,'protocol_sha256':sha((HERE/'bank_protocol.json').read_bytes()),'rows':out}
    compressed=gzip.compress((json.dumps(bundle,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode(),mtime=0);dst=HERE/'bank_held_predictions.json.gz'
    if dst.exists() and dst.read_bytes()!=compressed:raise ValueError('Different inclusive predictions already present')
    if not dst.exists():
        tmp=dst.with_suffix('.tmp');tmp.write_bytes(compressed);tmp.replace(dst)
    print(len(out),sha(compressed));return dst
if __name__=='__main__':save_predictions()
