"""Freeze two exposed-tree action forks using just parent public seats and parent root commands."""
from pathlib import Path
import gzip, hashlib, json, zipfile
HERE=Path(__file__).resolve().parent
FAMILY=HERE.parent
def sha(raw):return hashlib.sha256(raw).hexdigest()
def target(c,a):
    y,x=divmod(c,15);dx,dy={'UP':(0,-1),'RIGHT':(1,0),'DOWN':(0,1),'LEFT':(-1,0),'NONE':(0,0)}[a]
    return (y+dy)%15*15+(x+dx)%15
def dist(a,b):return sum(min((i-j)%15,(j-i)%15) for i,j in zip(divmod(a,15),divmod(b,15)))
def items():
    defs=[(14600909205,621758590,0,'takeover'),(14601625563,621758591,1,'plant')]
    for ar,seed,s,kind in defs:
        path=FAMILY/'cold_start/development/hosted'/f'{ar}.zip'
        with zipfile.ZipFile(path) as z:
            prefix='cold_repair/'
            reg=json.loads(z.read(prefix+'registration.json'))
            receipt=json.loads(z.read(prefix+'receipt.json'))
            fr=z.read(prefix+'observations.jsonl.gz');cmd=z.read(prefix+'turns.jsonl')
            assert receipt['turns_sha256']==sha(cmd) and (receipt['seed'],receipt['candidate_seat'])==(seed,s)
            frames=gzip.decompress(fr).splitlines(); commands=cmd.splitlines()
            for t in range(245,276):
                row=json.loads(commands[t]);pub=json.loads(frames[t]);ob=pub['observations'][s]
                own,foe=ob['players'][s],ob['players'][1-s]
                joint=row['calls'][s]['response']['action']
                bases={v for pl in ob['players'] for v in pl[1].values()}
                trees={v[0]:i for i,pl in enumerate(ob['players']) for v in pl[3].values()}
                found=False
                for uid,(cell,_,role) in sorted(own[2].items()):
                    if role!='PLANTER' or cell in bases or own[0]<300:continue
                    move=joint.get(uid,'NONE')
                    if kind=='takeover' and (trees.get(cell)!=1-s or move=='NONE'):continue
                    if kind=='plant' and (trees.get(cell) is not None or move!='NONE'):continue
                    destinations={target(w[0],joint.get(k,'NONE')) for k,w in own[2].items() if k!=uid}
                    if cell in destinations:continue
                    if move=='NONE':
                        options=[(min((dist(target(cell,a),w[0]) for w in foe[2].values()),default=15),-i,a)
                                 for i,a in enumerate(['UP','RIGHT','DOWN','LEFT']) if target(cell,a) not in destinations|bases]
                        if not options:continue
                        leave=max(options)[-1]
                    else:leave=move
                    stay=dict(joint);stay.pop(uid,None)
                    go={**joint,uid:leave}
                    found=True
                    yield dict(seed=seed,seat=s,step=t,uid=uid,cell=cell,kind=kind,historical=move,leave_direction=leave,
                               archive=str(ar),archive_sha256=sha(path.read_bytes()),public_stream_sha256=sha(fr),
                               public_root_hashes=row['public_observation_sha256'],terrain=reg['physical_map_sha256'],
                               stay=stay,leave=go,treasury=own[0],rng=[(seed+7919+31*seat)%2**32 for seat in (0,1)],
                               execution_order=['stay','leave'] if s==0 else ['leave','stay'])
                    break
                if found:break
            else: raise RuntimeError('Missing declared opportunity')
if __name__=='__main__':
    p=dict(candidate='7cf5c6c91fc59b46b2a38924c4c5ab56143fb1242d082f9426303af7336ac3ed',
           original=json.loads((FAMILY/'freight_response/protocol.json').read_bytes())['original_sha256'],
           purpose='Two training examples on previously exposed archived maps; no new game or formal rate. Enumerate first parent departure off an enemy tree or first parent stationary planting on empty non-base, in t245..275 with no ally targeting that cell and cash>=300. Other team-player orders from archived issued parent; none informed by root enemy order. For a vacant cell choose leave without allied occupancy or base, maximizing distance to nearest known enemy then fixed direction. New independent registered seat RNGs, true .58 wall-clocks calibrated on first three own public requests; both arms use same native calls/snapshots and opponent root command. Natural full continuation; retain failures/noops, no result-driven retest.',tickets=list(items()))
    path=HERE/'protocol.json';raw=(json.dumps(p,sort_keys=True,indent=2)+'\n').encode()
    assert not path.exists() or path.read_bytes()==raw
    path.write_bytes(raw)
    print(sha(raw),[(v['seed'],v['seat'],v['step'],v['cell'],v['historical'],v['leave_direction']) for v in p['tickets']])
