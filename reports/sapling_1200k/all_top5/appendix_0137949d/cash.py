"""Write a seat-specific cash ledger for every official transition, including actual worker-to-bank receipts."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
from prepare import HERE
from economy_extract import reconstruct,KEYS


def main(cache,directory=HERE):
    games=[]
    for g in json.loads((directory/'manifest.json').read_bytes())['games']:
        f=cache/g['path'];raw=f.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=g['sha256']:raise ValueError('replay digest')
        e=json.loads(gzip.decompress(raw));fs=e['steps'];econ,loot,banks,taxes=reconstruct(fs,e['configuration']);turns=[]
        for j,entries in enumerate(econ):
            before=fs[j][0]['observation']['players'];after=fs[j+1][0]['observation']['players'];before_cash=[s[0] for s in before];after_cash=[s[0] for s in after]
            for s in (0,1):
                if abs(sum(entries[s])-(after_cash[s]-before_cash[s]))>1.e-5:raise ValueError('unreconciled transition')
            turns.append([j+1,after_cash,[[round(v,7) for v in side] for side in entries],[[s,u,round(v,7)] for s,u,v in banks[j]],
                          [round(v,7) for v in loot[j]],[round(v,7) for v in taxes[j]]])
        games.append(dict(id=g['id'],replay_sha256=g['sha256'],opening_cash=[s[0] for s in fs[0][0]['observation']['players']],turns=turns))
    result=dict(physical_seats=[0,1],turn_schema=['post_action_t','post_cash','joint_cash_components_by_seat','own_delivery_seat_worker_amount','captured_enemy_cargo_by_beneficiary_not_yet_cash','enemy_cargo_tax_by_tree_beneficiary_included_in_tree_cash'],components=KEYS,games=games)
    data=json.dumps(result,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode();(directory/'turn_cash.json.gz').write_bytes(gzip.compress(data,mtime=0))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=Path('/workspace/sapling-all-top5-20261119'));p.add_argument('--manifest',type=Path,default=HERE/'manifest.json');a=p.parse_args();main(a.cache,a.manifest.parent)
