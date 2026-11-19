"""Reconcile both cash accounts; compare the exact original-examiner historical cohort and current holdout profiles."""
from pathlib import Path
import sys,collections,gzip,hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent;EXAM=HERE.parent;EXP=EXAM.parent;ACTOR=EXP.parents[1];ROOT=ACTOR.parents[2];BASE=ACTOR.parent
sys.path.insert(0,str(BASE))
from turning_point_ledger import FastCarbon,CashEvents
from fastsim import DEFAULT_CONFIG
def read(p):return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,v):
    p=HERE/n;b=json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if p.suffix=='.gz':b=gzip.compress(b,mtime=0)
    p.write_bytes(b)
def close(a,b):return all(abs(x-y)<.011 for x,y in zip(a,b))
def dist(x,y,n=15):
    dy,dx=abs(x//n-y//n),abs(x%n-y%n);return min(dy,n-dy)+min(dx,n-dx)
def index():
    target=read(EXAM/'games/parent/raw_621758343_0.json.gz')
    active={arm:read(EXAM/'games'/arm/'raw_621758343_0.json.gz')['pack_sha256'] for arm in ('candidate','parent')}
    rows=[];seen=set();receipts={}
    for evidence in sorted(ACTOR.rglob('official_replay.json')):
        for series in read(evidence).get('runs',{}).values():
            for official in series.get('games',[]):
                receipts.setdefault(official.get('transcript_sha256'),(evidence,official))

    for path in sorted(ACTOR.rglob('raw_*_[01].json.gz')):
        g=read(path)
        if (g.get('pack_sha256') not in active.values() or g.get('examiner_md5')!=target['examiner_md5'] or g.get('configuration')!=target['configuration']
                or g.get('game_configuration')!=target['game_configuration'] or g.get('errors')):continue
        receipt=path.with_name(path.name.replace('raw_','receipt_').replace('.json.gz','.json'))
        h=sha(path)
        if receipt.exists():
            r=read(receipt);o=r['official']
            if r['game']['raw_sha256']!=h:raise ValueError("wrong signed raw file")
        elif h in receipts:
            receipt,o=receipts[h]
        else:continue
        if (o.get('transcript_sha256')!=h or o['statuses']!=['DONE','DONE'] or not close(o['official_reward_by_seat'],g['official_score_by_seat'])
                or o['map_sha256']!=g['initial_terrain_u8_sha256']):raise ValueError('receipt mismatch')
        # The same basename on a different baseline may be searched; double-count only distinct arm/seed/seat once.
        tag=(g['pack_sha256'],g['seed'],g['candidate_seat'])
        if tag in seen:raise ValueError(('duplicate evidence',tag))
        seen.add(tag)
        rows.append(dict(path=str(path.relative_to(ROOT)),sha256=h,receipt_sha256=sha(receipt),receipt_path=str(receipt.relative_to(ROOT)),seed=g['seed'],seat=g['candidate_seat'],pack=g['pack_sha256'],terrain=o['map_sha256'],
                         source='direct343' if g['seed']==621758343 else 'historic_original',arm=next(a for a,v in active.items() if v==g['pack_sha256'])))
    for p in sorted((EXP/'games/holdout').glob('*.json.gz')):
        g=read(p);x=g['identity'];arm=x['arm']
        if arm not in ('parent','selected'):continue
        rows.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),seed=x['seed'],seat=x['seat'],pack=x['pack'],terrain=x['terrain'],source='self_owned_holdout',arm='candidate' if arm=='selected' else 'parent'))
    assert sum(r['source']=='direct343' for r in rows)==4
    return rows

def process(card,events=False):
    raw=read(ROOT/card['path']);s=card['seat'];seed=card['seed']
    cfg=raw.get('game_configuration',DEFAULT_CONFIG);sim=FastCarbon(cfg).reset(seed=seed)
    total=[collections.Counter(),collections.Counter()];per=[];hold=[];observed=[];flags=[]
    if hashlib.sha256(bytes(map(int,sim.carbon))).hexdigest()!=card['terrain']:raise ValueError('opening')
    for t,order in enumerate(raw['actions']):
        cash0=sim.cash.copy();snap=[sim.obs(j) for j in (0,1)]
        if events:observed.append(snap)
        for owner in (0,1):
            ps=snap[owner]['players'];own=ps[owner];opp=ps[1-owner]
            for uid,(pos,load,role) in own[2].items():
                if role!='COLLECTOR':continue
                h=min(dist(pos,a) for a in own[1].values());near=min((dist(pos,w[0]) for w in opp[2].values() if w[2]=='COLLECTOR' and w[1]<load),default=99)
                if load>=100:
                    flags.append((t,owner,uid,float(load),h,near))
        sim.cash=CashEvents(sim.cash);sim.step(*order)
        perstep=[collections.Counter(),collections.Counter()]
        for kind,owner,v,*args in sim.cash.events:
            perstep[owner][kind]+=v;total[owner][kind]+=v
            if kind=='tree_absorption':
                perstep[owner]['tree_ground']+=v-args[2];total[owner]['tree_ground']+=v-args[2]
                perstep[owner]['tree_tax']+=args[2];total[owner]['tree_tax']+=args[2]
            if events:hold.append(dict(t=t,kind=kind,recipient=owner,value=v,ident=args))
        if not close([cash0[k]+sum(v for e,v in perstep[k].items() if e not in ('tree_ground','tree_tax')) for k in (0,1)],sim.cash):raise ValueError('cash write mismatch')
        per.append(dict(t=t,balances=list(sim.cash),accounts=perstep,trees=[len(ps) for ps in sim.player_trees.values()],worker_sizes=[len(sim.player_workers[k]) for k in (0,1)]))
    outcomes=raw.get('official_score_by_seat') or raw['official']['scores']
    extinct=raw.get('extinct_by_seat') or raw['official']['extinct_by_seat']
    reconstructed=[sim.step_count-cfg['episodeSteps']-1 if extinct[k] else cash0[k] if any(extinct) else sim.cash[k] for k in (0,1)]
    if not close(reconstructed,outcomes):raise ValueError((card['path'],reconstructed,outcomes))
    for k in (0,1):total[k]['terminal_adjustment']=outcomes[k]-sim.cash[k]
    roles=[s,1-s]
    phase={}
    for lo,hi in ((0,100),(100,200),(200,299)):
        sums=[collections.Counter(),collections.Counter()]
        for p in per[lo:hi]:
            for k in (0,1):sums[k].update(p['accounts'][k])
        phase[str(lo)]=[sums[k] for k in roles]
    cargo=[]
    for k in roles:
        subset=[q for q in flags if q[1]==k]
        cargo.append(dict(laden=len(subset),near_home=sum(q[4]<=3 for q in subset),enemy_can_contest=sum(q[5]<=1 for q in subset),mean_home=sum(q[4] for q in subset)/max(len(subset),1)))
    row=dict(**card,scores_own_enemy=[outcomes[k] for k in roles],extinct=[extinct[k] for k in roles],cash_income=[total[k] for k in roles],
             phases=phase,cargo=cargo,tree_turns=[sum(q['trees'][k] for q in per) for k in roles],steps=len(per))
    return row,per,hold,observed,raw['actions']
def main():
    rows=index();summ=[]
    for c in rows:
        v,*_=process(c);summ.append(v)
    write('cohort_accounts.json.gz',summ)
    for sel in ('direct343','historic_original','self_owned_holdout'):
        rr=[v for v in summ if v['source']==sel];print(sel,len(rr),len({v['seed'] for v in rr}))
        for g in rr if sel=='direct343' else ():
            print(g['arm'],g['seat'],{k:round(g['cash_income'][0].get(k,0)-g['cash_income'][1].get(k,0)) for k in ('own_delivery','tree_absorption','tree_ground','tree_tax','plant','seize','hire_collector','hire_planter')})
    print('done')
if __name__=='__main__':main()
