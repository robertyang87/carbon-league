"""Incremental verified defense receipts. One atomic cache per game, with byte hashes."""
import argparse
from hashlib import sha256
import gzip
import json
import os
from pathlib import Path
import tempfile
import time
import fcntl
from defense import audit,HERE

VERSION=sha256((HERE/'defense.py').read_bytes()).hexdigest()
def compact(x):return json.dumps(x,separators=(',',':'),allow_nan=False,ensure_ascii=False).encode()
def atomic(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:f.write(data)
    os.replace(f.name,path)

def receipt(item,source,out):
    blob=source.read_bytes();h=sha256(blob).hexdigest()
    if item.get('sha256') and h!=item['sha256']:raise ValueError('source hash mismatch')
    path=out/(item['id']+'.json.gz')
    if path.exists():
        old=json.loads(gzip.decompress(path.read_bytes()))
        if old['sha256']==h and old['algorithm']==VERSION and old['id']==item['id']:
            return old
    g=json.loads(gzip.decompress(blob))
    if list(g['rewards'])!=item['score_by_seat'] or (item.get('seed') is not None and item['seed']!=g['configuration']['randomSeed']):
        raise ValueError('score or seed mismatch')
    if not all(frame['status'] in ('DONE','INVALID','TIMEOUT','ERROR') for frame in g['steps'][-1]):
        raise ValueError('nonterminal tape')
    data=audit(g)
    data.update(id=item['id'],side_names=item['seat_names'],score=item['score_by_seat'],cohort=item['cohort'],
                sha256=h,algorithm=VERSION,seed=g['configuration']['randomSeed'],terminal=g['statuses'])
    atomic(path,gzip.compress(compact(data),compresslevel=3,mtime=0))
    return data


def compare_exact(folder,out):
    exact=json.loads((HERE.parent/'latest5_intercepts.json').read_bytes());man=json.loads((HERE.parent/'latest5_inventory.json').read_bytes())
    lookup={g['id']:g for g in man['games']};checks=0;maxdiff=0.
    keys=('ticks','mixed_collision_squares','loaded_decisions','loaded_stay_under_threat','loaded_stay_under_threat_with_safe_move',
          'chose_geometrically_safe_move','all_enemy_base_donations','enemy_base_refund_at_contact','acquired_collector',
          'acquired_tree','acquired_base','acquisitions_all_methods','mixed_cargo_left_on_ground','mixed_cargo_destroyed_by_cell_cap',
          'collector_stolen_banked','collector_stolen_tax_paid_to_enemy_trees','collector_stolen_opponent_worker','collector_stolen_opponent_cash','collector_stolen_carried_at_terminal')
    for row in exact['games']:
        item=lookup[row['id']]
        compare=receipt(item,folder/item['path'],out)
        if len(compare['lots'])!=len(row['lots']):raise ValueError('lot count mismatch')
        for s in (0,1):
            for k in keys:
                diff=abs(compare['seat_totals'][s].get(k,0)-row['seat_totals'][s].get(k,0))
                if diff>.00001:raise ValueError((row['id'],s,k,diff))
                maxdiff=max(maxdiff,diff);checks+=1
        def lot_key(e):return e['t'],e['victim']
        gold={lot_key(e):e for e in row['lots']}
        for e in compare['lots']:
            ref=gold[lot_key(e)]
            if abs(ref['mass']-e['mass'])>1e-8 or any(abs(ref['fate'].get(k,0)-e['fate'].get(k,0))>1e-6 for k in set(ref['fate'])|set(e['fate'])):
                raise ValueError((row['id'],'lot fate differs',lot_key(e)))
    atomic(HERE/'defense_crosscheck.json',compact({'against':'../latest5_intercepts.json','cases':len(exact['games']),'per_seat_field_checks':checks,
                                                  'all_positive_lots':sum(len(g['lots']) for g in exact['games']),'max_field_delta':maxdiff,'algorithm':VERSION}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--crosscheck',type=Path)
    p.add_argument('--manifest',type=Path,default=HERE/'manifest.json');p.add_argument('--watch',action='store_true');a=p.parse_args()
    out=a.cache/'defense';out.mkdir(exist_ok=True,parents=True)
    if a.crosscheck:
        compare_exact(a.crosscheck,out);return
    checked=set();errors={}
    with (a.cache/'defense.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            man=json.loads(a.manifest.read_bytes())
            for item in man['games']:
                gid=item['id']
                if gid in checked or gid in errors:continue
                source=a.cache/item['path'];receipt_file=a.cache/'receipts'/(gid+'.json')
                if not (source.exists() and receipt_file.exists()):continue
                proof=json.loads(receipt_file.read_bytes())
                if not proof.get('frame_verified'):continue
                try:
                    receipt(dict(item,sha256=proof['sha256'],seed=proof['seed']),source,out)
                    checked.add(gid)
                except (ValueError,KeyError,TypeError) as exc:
                    errors[gid]=repr(exc)
                if (len(checked)+len(errors))%100==0:
                    print('validated',len(checked),'errors',len(errors),flush=True)
            atomic(a.cache/'defense_scan.json',compact({'n':len(checked),'errors':errors,'targets':len(man['games']),'algorithm':VERSION}))
            if not a.watch or len(checked)+len(errors)==len(man['games']) or man['state'] in ('complete','partial'):
                print(len(checked),len(man['games']),list(errors.items())[:8],flush=True);return
            time.sleep(3)



if __name__=='__main__':main()
