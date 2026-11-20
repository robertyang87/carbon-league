"""The cash replay may see both historical commands. The neural input keeps only the acting seat's."""
import gzip
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from register import ROOT, DER, sha, read
sys.path.insert(0,str(ROOT/'reports/sapling_1200k'))
from turning_point_ledger import FastCarbon, CashEvents

def hosted(c):
    path=ROOT/c['source']
    assert sha(path.read_bytes())==c['sha256']
    with zipfile.ZipFile(path) as z:
        name=c['repeat']+'/'
        reg=z.read(name+'registration.json');receipt=json.loads(z.read(name+'receipt.json'))
        frame_bytes=gzip.decompress(z.read(name+'observations.jsonl.gz')).splitlines()
        moves=z.read(name+'turns.jsonl')
    assert sha(reg)==receipt['protocol_sha256'] and sha(moves)==receipt['turns_sha256']
    reg=json.loads(reg);assert reg['seed']==c['seed'] and reg['physical_map_sha256']==c['terrain']
    assert reg['files']['candidate']['sha256']==c['own'] and reg['files']['opponent']['sha256'].startswith('65f268b2919c')
    assert receipt['status']=='complete' and receipt['counted'] and len(frame_bytes)==receipt['official_transitions']
    config=ROOT/'reports/sapling_1200k/master6_actor_dev/cargo_tree_action_learning_dev/direct_original/games/candidate/raw_621758346_1.json.gz'
    sim=FastCarbon(json.loads(gzip.decompress(config.read_bytes()))['game_configuration']).reset(seed=c['seed'])
    assert sha(bytes(map(int,sim.carbon)))==c['terrain']
    sim.cash=CashEvents(sim.cash)
    pending=[];returns=[]
    for t,(fr,move) in enumerate(zip(frame_bytes,moves.splitlines(),strict=True)):
        fr=json.loads(fr); obs=fr['observations']; move=json.loads(move)
        assert t==fr['step']==move['step'] and fr['public_observation_sha256']==move['public_observation_sha256']
        assert np.allclose([p[0] for p in obs[0]['players']],list(sim.cash),rtol=0,atol=.000001)
        orders=[v['response']['action'] for v in move['calls']]
        if t in range(6,289,6):
            for seat in c['seats']:
                public=obs[seat]
                assert sha((json.dumps(public,separators=(',',':'),allow_nan=False)+'\n').encode())==fr['public_observation_sha256'][seat]
                pending.append((t,seat,public['players'],orders[seat]))
        sim.cash.events.clear();sim.step(*orders)
        dep=[0.,0.]
        for kind,s,value,*_ in sim.cash.events:
            if kind=='own_delivery': dep[s]+=value
        returns.append(dep)
    assert len(returns)==receipt['official_transitions']
    return pending,returns

def ladder(c):
    sys.path.insert(0,str(ROOT/'reports/sapling_1200k/all_top5'))
    import economy_extract
    path=ROOT/c['source']
    assert sha(path.read_bytes())==c['sha256']
    replay=json.loads(gzip.decompress(path.read_bytes())); frames=replay['steps'];seat=c['seats'][0]
    assert sha(bytes(frames[0][0]['observation']['full_carbon']))==c['terrain']
    assert replay['statuses']==['DONE','DONE']
    _,_,banks,_=economy_extract.reconstruct(frames,replay['configuration'])
    pending=[];returns=[]
    for t,txn in enumerate(banks):
        if t in range(6,289,6):
            pending.append((t,seat,frames[t][0]['observation']['players'],frames[t+1][seat]['action'] or {}))
        dep=[0.,0.]
        for s,uid,v in txn:dep[s]+=v
        returns.append(dep)
    return pending,returns

def rows(cards):
    for c in cards:
        pending,returns=(hosted if c['domain']=='h2h' else ladder)(c)
        for t,s,players,order in pending:
            if t+10>len(returns):
                continue
            both=np.sum(returns[t:t+10],axis=0)
            yield dict(seed=c['seed'],terrain=c['terrain'],domain=c['domain'],seat=s,source_arm=c['repeat'],step=t,
                       players=players,order=order,target=[float(both[s]),float(both[1-s])],
                       match=c['sha256']+'/'+c['repeat'])
