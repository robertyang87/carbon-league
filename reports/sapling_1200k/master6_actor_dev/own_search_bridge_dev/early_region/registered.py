"""Prioritize a publicly valuable reachable tree region at early affordable planting decisions."""
import copy
import fcntl
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'third_route'))
import run as third
import audit as original_audit
sr = third.sr

def distance(a,b):
    dr,dc=abs(a//15-b//15),abs(a%15-b%15)
    return min(dr,15-dr)+min(dc,15-dc)

def ranks(root):
    obs=root['public_at_root'];s=obs['player']
    own,enemy=obs['players'][s],obs['players'][1-s]
    enemy_trees={v[0]:(uid,v[1]) for uid,v in enemy[3].items()}
    occupied={v[0] for v in own[3].values()}|set(own[1].values())|set(enemy[1].values())
    def value(cell):
        if cell in occupied:return 0.
        adj=[z for z in range(225) if distance(z,cell)==1]
        carbon=sum(obs['carbon'][z] or 0 for z in adj if z not in occupied and z not in enemy_trees)
        earnings=carbon*.0375
        if cell in enemy_trees:
            uid,age=enemy_trees[cell]
            earnings+=2*float(obs['trees'].get(uid,[None,0])[1])*max(0,1-age/50)
        return earnings
    sites=[value(i) for i in range(225)]
    return {m:max(v/(1+distance(i,a['cell'])) for i,v in enumerate(sites) if distance(i,a['cell'])<=2)
            for m,a in root['candidate']['actions'].items()}

def choose():
    import gzip, json
    tree=third.TREE
    games={(r['seed'],r['seat']):r for r in sr.load(tree/'source_receipt.json')['games']}
    with gzip.open(tree/'opportunities.jsonl.gz','rt') as file:
        pool=[r for r in map(json.loads,file) if r['t']<=100 and r['parent_command']=='NONE' and r['actions']['NONE']['can_pay_now']]
    boards={}
    for key,game in games.items():
        raw=sr.load(sr.ROOT/game['raw_path'])
        sim=sr.FastCarbon(raw['game_configuration']).reset(seed=key[0])
        for t,pair in enumerate(raw['actions'][:101]):
            boards[key,t]=sim.obs(key[1]);sim.step(*pair)
    selected=[];seen=set()
    for seat in (0,1):
        for item in sorted(pool,key=lambda r:(r['t'],r['seed'],r['uid'])):
            seed=item['seed'];t=item['t'];key=(seed,seat)
            if item['seat']!=seat or seed in seen:continue
            scores=ranks(dict(public_at_root=boards[key,t],candidate=item))
            legal=[m for m in sr.MOVES[1:] if not item['actions'][m]['friendly_conflict']
                   and item['actions'][m]['enemy_collector_distance']>1]
            if not legal:continue
            cmd=max(legal,key=lambda m:scores[m])
            if scores[cmd]<=scores['NONE']:continue
            game=games[key];src=game['raw_path'];raw=sr.load(sr.ROOT/src)
            root=sr.route(raw,item)
            if not root or cmd==root['new']:continue
            p=sorted(tree.glob(f'case_{seed}_{seat}_*/live_protocol.json'))[0]
            q=sr.load(p)
            assert q['record']['source']==src
            q.update(root_t=t,root=dict(t=t,worker=item['uid'],old='NONE',tree='',alternative=cmd))
            q.pop('training_root_sha256',None)
            forced=copy.deepcopy(item)
            for key,v in forced['actions'].items():
                if key!=cmd:v['friendly_conflict']=True
            alt=sr.route(sr.load(sr.ROOT/q['record']['source']),forced)
            if not alt:continue
            assert alt['new']==cmd and alt['v1']==root['v1']
            out=HERE/f'seat{seat}';out.mkdir(exist_ok=True)
            row=dict(q,family='early_region',root={**q['root'],'alternative':cmd},old_protocol_sha256=sr.sha(p),
                     selection_sha256=sr.sha(HERE/'protocol.md'),selector_sha256=sr.sha(Path(__file__)))
            sr.retain(out/'live_protocol.json',row);sr.retain(out/'live_shared_init.json',sr.load(p.parent/'live_shared_init.json'))
            selected.append(dict(seat=seat,seed=q['seed'],step=q['root_t'],scores=scores,new_route=cmd,prior_route=root['new']))
            seen.add(q['seed']);break
    sr.retain(HERE/'selection.json',selected)
    if len(selected)!=2:raise ValueError('insufficient eligible early roots')

if __name__=='__main__':
    third.HERE=HERE
    if len(sys.argv)>1:
        if sys.argv[1]=='verify':
            original_audit.HERE=HERE;original_audit.verify()
        else:third.one(int(sys.argv[1]),sys.argv[2])
    else:
        with (third.TREE/'.continuation.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);choose()
            for seat in (0,1):
                jobs=[subprocess.Popen(['taskset','-c',str((seat+i)%2),sys.executable,__file__,str(seat),arm]) for i,arm in enumerate(('parent','third'))]
                codes=[job.wait() for job in jobs]
                if any(codes):raise RuntimeError(codes)
