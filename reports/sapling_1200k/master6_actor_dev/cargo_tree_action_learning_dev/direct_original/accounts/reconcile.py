"""Replay the eight predeclared games locally and split normal deliveries from tree trading and tax."""
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXAM = HERE.parent
ACTOR = EXAM.parents[1]
ROOT = ACTOR.parents[2]
sys.path.insert(0, str(ACTOR / 'deployed_mix_policy_gradient_dev/full_episode/direct_original/economy'))
from analyze import process, read, sha

def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2) + '\n')
    temp.replace(path)

def deposits(events, seat, start, end):
    return [e for e in events if e['kind']=='own_delivery' and e['value']>0 and
            e['recipient']==seat and start<=e['t']<end]

def bank_summary(events):
    vals = [e['value'] for e in events]
    return {'n':len(vals),'sum':sum(vals),'mean':sum(vals)/len(vals) if vals else None,
            'large_n':sum(v>=200 for v in vals),'large_sum':sum(v for v in vals if v>=200)}

def one(complete, binding):
    c = read(complete)
    seat, arm, seed = c['seat'], c['arm'], c['seed']
    folder = EXAM/'games'/arm
    raw = folder/f'raw_{seed}_{seat}.json.gz'
    rpt = folder/f'receipt_{seed}_{seat}.json'
    receipt = read(rpt); o = receipt['official']; g = receipt['game']; tape = read(raw)
    assert (sha(raw)==c['raw_sha256']==o['transcript_sha256']==g['raw_sha256']
            and sha(rpt)==c['receipt_sha256'] and o['statuses']==['DONE','DONE']
            and receipt['binding']==binding and tape['pack_sha256']==binding[arm]
            and tape['candidate_seat']==o['seat']==seat and tape['seed']==o['seed']==seed
            and tape['official_score_by_seat']==o['official_reward_by_seat']
            and tape['initial_terrain_u8_sha256']==o['map_sha256'] and not tape['errors'])
    card={'path':str(raw.relative_to(ROOT)),'seed':seed,'seat':seat,'terrain':o['map_sha256']}
    rr, _, events, _, _ = process(card,events=True)
    sides={}
    for name,i in [('own',seat),('opponent',1-seat)]:
        loc=0 if name=='own' else 1
        rank='candidate' if name=='own' else 'opponent'
        full = deposits(events,i,0,299)
        by_due =[dict(t=e['t'],worker=e['ident'][0],recipient_seat=i,amount=e['value']) for e in full]
        checks=g['final'][rank]
        assert len(full)==checks['positive_delivery_events']
        assert math.isclose(sum(e['value'] for e in full),checks['own_positive_delivery_carbon'],abs_tol=1e-6)
        assert math.isclose(rr['cash_income'][loc]['tree_absorption'],checks['cumulative_tree_absorption_income'],abs_tol=1e-6)
        sides[name]={'normal_deliveries':bank_summary(full),'late_t200_298':bank_summary(deposits(events,i,200,299)),
                     'mid_soil_t100_199':rr['phases']['100'][loc]['tree_ground'],
                     'mid_enemy_tax_t100_199':rr['phases']['100'][loc]['tree_tax'],
                     'mid_total_absorption_t100_199':rr['phases']['100'][loc]['tree_absorption'],
                     'all_receipts':by_due,'terminal_score':rr['scores_own_enemy'][loc],'extinct':rr['extinct'][loc]}
    return {**card,'arm':arm,'scores_by_physical_seat':o['official_reward_by_seat'],'statuses':o['statuses'],'steps':rr['steps'],
            'result':o['result'],'raw_sha256':sha(raw),'receipt_sha256':sha(rpt),'complete_sha256':sha(complete),'sides':sides}

def main():
    protocol, binding = read(EXAM/'protocol.json'),read(EXAM/'binding.json')
    games={}
    for seed in protocol['seeds']:
        for seat in (0,1):
            for arm in ('parent','candidate'):
                stem=f'{arm}_{seed}_{seat}'
                done=EXAM/f'complete_{stem}.json'; dest=HERE/f'{stem}.json'
                if not done.exists():
                    continue
                if dest.exists():
                    record=read(dest)
                    assert record['complete_sha256']==sha(done) and record['raw_sha256']==sha(ROOT/record['path'])
                else:
                    record=one(done,binding)
                    save(dest,record)
                games[seed,seat,arm]=record
    print(f'{len(games)}/8 locally reconciled',flush=True)
    pairs=[]
    for seed in protocol['seeds']:
        for seat in (0,1):
            if any((seed,seat,arm) not in games for arm in ('parent','candidate')): continue
            parent,new=[games[seed,seat,arm] for arm in ('parent','candidate')]
            assert parent['terrain']==new['terrain']
            p={}
            for who in ('own','opponent'):
                a,b=[v['sides'][who] for v in (parent,new)]
                x={k:b[k]-a[k] for k in ('mid_soil_t100_199','terminal_score')}
                for segment in ('normal_deliveries','late_t200_298'):
                    x[segment]={k:b[segment][k]-a[segment][k] for k in ('n','mean','sum','large_n','large_sum')}
                p[who]=x
            p['margin']=p['own']['terminal_score']-p['opponent']['terminal_score']
            pairs.append({'seed':seed,'seat':seat,'delta':p})
    save(HERE/'summary.json',{'n':len(games),'pairs':pairs,'protocol_sha256':sha(EXAM/'protocol.json'),
                              'implementation_sha256':sha(Path(__file__)),'clock':'All intervals use t just before logged action. Positive ordinary home deliveries; soil excludes enemy cargo tax.'})

if __name__=='__main__':
    main()
