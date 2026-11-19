"""Audit actual base choices, worker birth, subsequent deposits and held-tree cash on complete frozen v1 tapes."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
DEV=HERE.parent
BASE=DEV.parent
ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from turning_point_ledger import CashEvents,FastCarbon,compact
V1="9c54df17eba7cfecf6ac2698eccbbadc2607665e16ab613925188131ce5514e0"
EXAM="57f9a594cce4f2ec9e3573d8d9d26ca1"
def H(b):return hashlib.sha256(b).hexdigest()
def read(p):
    b=p.read_bytes()
    return json.loads(gzip.decompress(b) if p.suffix=='.gz' else b)
def stage(t):
    return next(tag for stop,tag in ((60,'0-59'),(150,'60-149'),(220,'150-219'),(260,'220-259'),(280,'260-279'),(290,'280-289'),(300,'290-298')) if t<stop)
def dist(a,b):
    u,v=abs(a//15-b//15),abs(a%15-b%15)
    return min(u,15-u)+min(v,15-v)
def sources():
    fit=DEV/'terminal_return_fit';rows={}
    for directory in (fit/'dev_learned',fit/'replication/learned'):
        ledger=read(directory/'results.json')
        official=read(directory/'official_replay.json')['runs'][directory.name]
        if ledger['header']['pack_sha256']!=V1 or ledger['header']['examiner_md5']!=EXAM or ledger['status']!='complete' or official['pack_sha256']!=V1:
            raise ValueError('input is not the complete original opponent pair')
        for g in ledger['games']:
            seat=g['candidate_seat'];seed=g['seed'];p=directory/g['raw_file'];sg=H(p.read_bytes());raw=read(p)
            o=next(z for z in official['games'] if (z['seat'],z['seed'])==(seat,seed))
            if sg!=g['raw_sha256'] or sg!=o['transcript_sha256'] or o['statuses']!=['DONE','DONE'] or o['official_reward_by_seat']!=g['official_score_by_seat'] or raw['pack_sha256']!=V1 or raw['examiner_md5']!=EXAM:
                raise ValueError('archive mismatch')
            source={'source':str(p.relative_to(ROOT)),'source_sha256':sg,'official_rewards':o['official_reward_by_seat'],'group':'complete'}
            rows[seed,seat]=(raw,source)
            yield raw,source
    if set(rows)!={(seed,seat) for seed in (621461220,621461321,621461322) for seat in (0,1)}:
        raise ValueError('unexpected six-game baseline')
    for key in ((621461321,0),(621461322,1)):
        original,stock=rows[key];d=DEV/'planter_option'/f'response_{key[0]}_{key[1]}'
        p=d/'live_parent.json';meta=read(p);t=meta['root']['t'];tape=d/meta['action_tape']
        if (meta['pack_sha256']!=V1 or meta['original_trajectory_sha256']!=stock['source_sha256'] or
            H(tape.read_bytes())!=meta['action_sha256'] or meta['arm']!='parent' or meta['terminal_t']!=299 or meta['statuses']!=['DONE','DONE']):
            raise ValueError('unchanged-policy branch not bound to original')
        act=original['actions'][:t]+read(tape)
        if len(act)!=299 or act[t]!=original['actions'][t]:raise ValueError('branch did not preserve source root')
        yield {**original,'actions':act},{'group':'shared_prefix_branch','root':t,'source':str(p.relative_to(ROOT)),
                                          'source_sha256':H(p.read_bytes()),'continuation_sha256':meta['action_sha256'],
                                          'original_sha256':stock['source_sha256'],'official_rewards':meta['score']}


def audit(raw,source):
    cfg=raw['game_configuration'];seat=raw['candidate_seat'];sim=FastCarbon(cfg).reset(seed=raw['seed'])
    sim.cash=CashEvents(sim.cash)
    if H(bytes(map(int,sim.carbon)))!=raw['initial_terrain_u8_sha256']:raise ValueError('terrain mismatch')
    rows=[];workers={};summary=[Counter(),Counter()];unknown=[Counter(),Counter()]
    for t,joint in enumerate(raw['actions']):
        starts=list(sim.worker_counters);n_roles=Counter(w.is_collector for w in sim.workers.values())
        prices={'C':cfg['recCollectorCost']+n_roles[True]*cfg['recCollectorIncreaseCost'],
                'P':cfg['recPlanterCost']+n_roles[False]*cfg['recPlanterIncreaseCost']}
        tree_count=sum(tr.age<cfg['treeLifespan'] for tr in sim.trees.values())
        allowed=[];expected={}
        for s in (0,1):
            cash=sim.cash[s];capacity=cfg['workerLimit']-len(sim.player_workers[s])
            me=sim.player_workers[s];roles=Counter('C' if sim.workers[u].is_collector else 'P' for u in me)
            land={sim._move_nbr[joint[s][u]][sim.workers[u].pos] if joint[s].get(u) in sim._move_nbr else sim.workers[u].pos for u in me}
            foes=[sim.workers[u] for u in sim.player_workers[1-s]]
            tree_price=cfg['plantCost'] if len(sim.player_trees[s])<=cfg['playerTreeProtectiveNumber'] else cfg['plantCost']+cfg['plantCostInflationRatio']*(cfg['plantCostInflationBase']**tree_count)
            for bid,cell in sim.bases[s]:
                cmd=joint[s].get(bid,'NONE');role={'RECCOLLECTOR':'C','RECPLANTER':'P','NONE':None}[cmd]
                choices=[r for r,p in prices.items() if p<=cash and capacity>0]
                r={'t':t,'seat':s,'base':bid,'cell':cell,'command':cmd,'choices':choices,
                   'pre_cash':round(cash,5),'remaining_capacity':capacity,'workersC':roles['C'],'workersP':roles['P'],
                   'trees':len(sim.player_trees[s]),'enemy_trees':len(sim.player_trees[1-s]),'own_landing':cell in land,
                   'opponent_in_one_step':any(dist(w.pos,cell)<=1 for w in foes),'prices':prices,
                   'tree_price':round(tree_price,5),'would_leave_one_hire':{k:cash-v>=min(prices.values()) for k,v in prices.items()},
                   'would_leave_plant_or_seize':{k:cash-v>=min(tree_price,cfg['seizeCost']) for k,v in prices.items()}}
                rows.append(r)
                if role in choices:
                    uid=f'player-{s}-worker-{starts[s]}';starts[s]+=1;price=prices[role];capacity-=1;cash-=price
                    r.update(uid=uid,paid=price,after_hire_cash=round(cash,5))
                    workers[uid]={'uid':uid,'role':role,'seat':s,'t':t,'base':bid,'price':price,
                                  'cash_after_payment_before_collections':round(cash,5),'death':None,'birth_visible_after_step':False,
                                  'bank':0.,'tree':0.,'tree_cost':0.,'events':[]}
                    expected[bid]=(role,price,uid)
                if s==seat:allowed.append(r)
        sim.cash.events.clear();sim.step(*joint)
        paid={}
        for typ,s,val,*detail in sim.cash.events:
            if typ.startswith('hire_'):
                bid=detail[0];role='C' if typ=='hire_collector' else 'P';paid[bid]=(role,-val)
            elif typ in ('own_delivery','plant','seize','tree_absorption'):
                key={'own_delivery':'bank','plant':'tree_cost','seize':'tree_cost','tree_absorption':'tree'}[typ]
                uid=sim.trees_dict[detail[0]][0] if typ=='tree_absorption' else detail[0]
                amount=-val if key=='tree_cost' else val
                summary[s][key]+=amount
                if uid not in workers:unknown[s][key]+=amount
                else:
                    w=workers[uid]
                    if w['seat']!=s:raise ValueError('cross-seat worker credit')
                    w[key]+=amount;w['events'].append((t+1,typ,round(amount,9)))
        if paid!={bid:(r,cost) for bid,(r,cost,_) in expected.items()} or list(sim.worker_counters)!=starts:
            raise ValueError('sequential legal base commands do not match actual new worker IDs/prices')
        for uid,_,_,_,_ in sim.last_collisions:
            if uid in workers:workers[uid]['death']=t+1
        for _,_,uid in expected.values():
            workers[uid]['birth_visible_after_step']=uid in sim.workers
        for uid,w in workers.items():
            if w['death'] is None and uid not in sim.workers:w['death']=t+1
    if list(sim.cash)!=source['official_rewards']:
        raise ValueError('reconstructed terminal cash differs from official full-horizon game')
    for w in workers.values():
        w['lifetime_net']=round(w['bank']+w['tree']-w['tree_cost']-w['price'],5)
        for k in ('bank','tree','tree_cost'):w[k]=round(w[k],5)
        w['paid_back_by_10']=sum((v if kind in ('own_delivery','tree_absorption') else -v) for at,kind,v in w['events'] if at<=w['t']+10)>=w['price']
    return {'source':source,'seed':raw['seed'],'seat':seat,'end':len(raw['actions']),'summary':[dict(z) for z in summary],
            'income_not_attributed_to_hires':[dict(z) for z in unknown],'base_rows':rows,'workers':workers}


def main():
    result=[audit(raw,source) for raw,source in sources()]
    blob=compact({'version':1,'source_sha256':H(Path(__file__).read_bytes()),'games':result})
    (HERE/'baseline_games.json.gz').write_bytes(gzip.compress(blob,mtime=0))
    print('replayed',len(result),'full or assembled records')

if __name__=='__main__':main()
