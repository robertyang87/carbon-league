"""Replay the official perspective and score a frozen endgame head on two completed V4 matches."""
from pathlib import Path
import copy
import gzip
import hashlib
import importlib.util
import json
import sys

HERE=Path(__file__).resolve().parent
D=HERE.parents[1]
ROOT=D.parents[1]
SRC=D/'endgame_cash_projection'
ENGINE=ROOT/'code/env_src/carbon_challenge-master'
STEPS=(259,269,274,277,279,284,289,294)
IDS=(('6e994d80674dc2430a4925b71bb6a947',0),('9ba910d9ee46e3bdf280604c756d004d',1))
WEIGHTS_SHA='ceed9c721d2ff70bcd2a47bb47dacb5ca88b51073e8a7fb9beefe3598e679cd6'
RUNTIME_SHA='3da2214bdefa8a2715d625379628f157153409464b310f75bcc5507097587a70'

def sha(data):return hashlib.sha256(data).hexdigest()
def read(p):return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())
def source(p):return {'path':str(p.relative_to(ROOT)),'sha256':sha(p.read_bytes())}
def prediction(rt,obs,cfg,models):
    # This function receives the very object the referee sends to this physical seat.
    assert 'full_carbon' not in obs
    players,s,t=obs['players'],obs['player'],obs['step']
    n=cfg['size'];left=cfg['episodeSteps']-1-t
    row=rt.features(players,s,left,n,cfg['treeLifespan'])
    opposing=rt.features(players,1-s,left,n,cfg['treeLifespan'])
    assert all(abs(a+b)<1e-10 for a,b in zip(row,opposing))
    order=(s,1-s); margin=players[s][0]-players[1-s][0];distance=rt.lattice(n)[0]
    counts=[];witnesses=[]
    for side in order:
        bases=list(players[side][1].values())
        carriers=[dict(uid=uid,pos=pos,cargo=load,d=min(distance[pos][b] for b in bases)) for uid,(pos,load,kind) in players[side][2].items() if kind=='COLLECTOR' and load>0]
        counts.append(dict(cash=players[side][0],all_cargo=sum(w['cargo'] for w in carriers),reachable=sum(w['cargo'] for w in carriers if w['d']<=left),
                      tight_1=sum(w['cargo'] for w in carriers if 0<=left-w['d']<=1),at_most_2_away=sum(w['cargo'] for w in carriers if w['d']<=min(2,left))))
        witnesses.append(sorted(carriers,key=lambda w:-w['cargo']))
    head={}
    for key in ('aggregate','enhanced'):
        delta=rt.predict_residual(row,models[key]);head[key]=dict(predicted_future_cash_change=delta,predicted_margin=margin+delta)
    assert abs(row[8]*1000-(counts[0]['reachable']-counts[1]['reachable']))<1e-6 and abs(row[0]*1000-margin)<1e-6
    return dict(t=t,remaining=left,teams=counts,carriers=witnesses,cash_margin=margin,projected=head,public_input=sha(json.dumps([players,s,t],separators=(',',':')).encode()))

def run(mid,s,rt,weights):
    folder=D/'ladder/v4_observations/increment_0904'
    path=folder/'replays'/(mid+'-1.json.gz');ledger_path=folder/(mid+'-1.ledger.json.gz')
    game=read(path); ledger=read(ledger_path); cfg=game['configuration']; frames=game['steps']
    assert ledger['replay_sha256']==sha(path.read_bytes()) and ledger['seat']==s and ledger['official']['matched']==299
    sys.path.insert(0,str(ENGINE))
    from zerosum_env import make
    import numpy as np
    np.random.seed(cfg['randomSeed']);env=make('carbon',configuration=cfg)
    env.states[0].observation.full_carbon=copy.deepcopy(frames[0][0]['observation']['full_carbon'])
    rows=[]
    for t,f in enumerate(frames):
        if t:env.step([x['action'] or {} for x in f],[])
        public=env._Environment__get_shared_state(s).observation
        assert public['step']==t and public['player']==s and public['players']==f[0]['observation']['players']
        assert public['trees']==f[0]['observation']['trees'] and public['carbon']==f[s]['observation']['carbon'] and 'full_carbon' not in public
        if 249<=t<=298:rows.append(prediction(rt,public,cfg,weights))
    scores=[frames[-1][i]['reward'] for i in (s,1-s)]
    assert scores==ledger['official_scores'] and game['statuses']==['DONE','DONE']
    truth=scores[0]-scores[1]
    for r in rows:
        r['eventual_cash_change']=truth-r['cash_margin']
    return dict(match=mid,opponent=ledger['opponent'],seat=s,terrain=sha(bytes(frames[0][0]['observation']['full_carbon'])),official_scores=scores,official_margin=truth,
             verified_public_states=len(frames),replay=source(path),ledger=source(ledger_path),display_steps=STEPS,rows=rows)

def main():
    p=SRC/'runtime.py';w=SRC/'weights.json'
    assert sha(p.read_bytes())==RUNTIME_SHA and sha(w.read_bytes())==WEIGHTS_SHA
    spec=importlib.util.spec_from_file_location('frozen_projection',p);rt=importlib.util.module_from_spec(spec);spec.loader.exec_module(rt)
    weights=read(w);assert weights['runtime_sha256']==RUNTIME_SHA
    assert weights['enhanced']['features']==list(rt.FULL) and weights['aggregate']['features']==list(rt.BASE)
    schema=ENGINE/'zerosum_env/envs/carbon/carbon.json';fields=read(schema)['observation']
    assert fields['full_carbon']['hidden'] and fields['players']['shared'] and not fields['carbon'].get('shared')
    results=[run(mid,s,rt,weights) for mid,s in IDS]
    training_maps=SRC/'maps.json'
    assert sha(training_maps.read_bytes())==weights['maps_sha256']
    panels={r['terrain'] for r in read(training_maps)['tickets']}
    assert len(panels)==200 and not panels.intersection(r['terrain'] for r in results)
    out=dict(training_maps=source(training_maps),map_intersections=[],runtime=source(p),weights=source(w),official_schema=source(schema),
       official_delivery=source(ENGINE/'zerosum_env/core.py'),official_interpreter=source(ENGINE/'zerosum_env/envs/carbon/carbon.py'),
       order='teams and scores in V4, opponent order. Whole opponent inventory in players is shared and verified via actual referee request projection; cell carbon and full_carbon are excluded from the model. Nonzero cargo at shortest path distance <=remaining is counted as geometrically reachable with the same bearer, per original runtime. This does not predict certain real delivery and omits new cargo and transfers.',
       cases=results)
    (HERE/'forecast.json').write_text(json.dumps(out,separators=(',',':'),allow_nan=False,ensure_ascii=False)+'\n')
    for c in results:
        print(c['opponent'],c['official_margin'])
        for r in c['rows']:
            if r['t'] in STEPS:print(r['t'],*[round(x,2) for x in [r['cash_margin'],*[a['reachable'] for a in r['teams']],r['projected']['aggregate']['predicted_margin'],r['projected']['enhanced']['predicted_margin']]])
if __name__=='__main__':main()
