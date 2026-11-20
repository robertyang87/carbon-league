"""Replay a fixed ladder loss through the two frozen forecast-and-search packages."""
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
LADDER = HERE.parents[2]
PKG = LADDER/'prediction_feedback/package'
sys.path.insert(0, str(PKG))
import probe
import numpy as np


def action(orders, uid):
    return (orders.get(uid) or 'NONE') if uid is not None else None


def focal(side, number):
    return f'player-{side}-worker-{number}' if number is not None else None


def main():
    os.sched_setaffinity(0,{int(os.getenv('FABLE_CPU','2'))})
    probe.torch.set_num_threads(1);probe.torch.set_num_interop_threads(1)
    p=json.loads((HERE/'protocol.json').read_bytes());s=p['seat'];enemy=1-s
    prefs=[(LADDER.parent/'cold_start/candidate.py').read_bytes(),(PKG/'candidate.py').read_bytes()]
    assert [probe.sha(b) for b in prefs]==[p['prior'],p['child']]
    old,new=[probe.module(b) for b in prefs];stats=json.loads(gzip.decompress((PKG/'roots.json.gz').read_bytes()));costs=stats['costs']
    raw=(LADDER/'replays'/(p['game']+'-1.json.gz')).read_bytes();assert probe.sha(raw)==p['replay']
    data=json.loads(gzip.decompress(raw));frames=data['steps'];cfg={k:v for k,v in data['configuration'].items() if k!='randomSeed'}
    tr=old.BeliefTracker(old._Bel,s);prev=None;cases={row[0]:row[1:] for row in p['cases']};selected=[];allunits=[];full=[];history={}
    for t,frame in enumerate(frames[:-1]):
        v=frame[s]['observation'];both=frame[0]['observation'];players=both['players']
        obs=dict(step=t,player=s,players=players,trees=both['trees'],carbon=v['carbon'],remainingOverageTime=v['remainingOverageTime'])
        players=obs['players'];actual=[f['action'] or {} for f in frames[t+1]]
        # Snapshot before tracker.step, which mutates the prior. The assessor enters through the public API.
        prior=copy.deepcopy(tr) if t in cases or len(selected)<2 else None
        est,_=tr.step(players,t,obs['carbon']);x=old._quant(old.encode(players,t,enemy,est,[1.]*old.N,prev))[None]
        lw,lb,_=old._net()(x.astype(np.float32),old._SID);lw,lb=old._opponent_root_readout(x,lw,lb,players,enemy)
        offsets=new._feedback_adjust(lw.copy(),players,enemy,new._FEEDBACK_WEIGHTS,new._OPP_STEPS,new.FEAT_DIST)
        seed=int(1000003*(t+1)+17*s)%(2**31)+7
        choices=[old._sample(players,enemy,z[0],lb[0],np.random.RandomState(seed),t) for z in (lw,offsets)]
        collectors=[u for u,info in players[enemy][2].items() if info[2]=='COLLECTOR']
        for u in collectors:
            pos,cargo,_=players[enemy][2][u];ai=old.WIDX[action(actual[enemy],u)];zs=[lw[0,pos],offsets[0,pos]];pr=[np.exp(q.astype(float)-np.max(q)) for q in zs]
            allunits.append(dict(step=t,id=u,cargo=cargo,pos=pos,real=action(actual[enemy],u),old=action(choices[0],u),new=action(choices[1],u),raw_p_real=[float(q[ai]/q.sum()) for q in pr]))
        if len(selected)<2 and any(action(choices[0],u)!=action(choices[1],u) for u in collectors):selected.append(t)
        if t in cases or t in selected:
            rng=int.from_bytes(hashlib.sha256(f"{p['game']}/{t}".encode()).digest()[:4],'big')
            sequence=['original_1','candidate_1','candidate_2','original_2'] if len(full)%2==0 else ['candidate_1','original_1','original_2','candidate_2']
            arms={key:probe.root_call(new if key.startswith('candidate') else old,prior,prev,obs,cfg,rng,costs) for key in sequence}
            out=probe.assess(arms);a,b,x,y=[arms[tag] for tag in ('original_1','original_2','candidate_1','candidate_2')]
            isrepeat=a['depth_by_arm']==b['depth_by_arm'] and x['depth_by_arm']==y['depth_by_arm'];samples=all(g['opponent_samples']==[choices[int(k.startswith('candidate'))]] for k,g in arms.items())
            out.update(repeat_depth_per_id=isrepeat,samples_verified=samples,stable_change=bool(out['attributable_changed'] and isrepeat and samples))
            foe,worker,opt=cases.get(t,(None,None,None));fid,uid=focal(enemy,foe),focal(s,worker)
            for k,g in arms.items():
                g['test_worker_menu']=[action(cmd,uid) for cmd in g['menu']];g['test_worker_issued']=action(g['issued'],uid)
            full.append(dict(t=t,focus_enemy=fid,focus_worker=uid,reviewed_alternative=opt,actual=actual,enemy_predictions=choices,focus_enemy_predictions=[action(c,fid) for c in choices],arms=arms,assessment=out,
                    lookback=[dict(t=j,workers={u:dict(pos_cargo_role=frames[j][0]['observation']['players'][side][2].get(u),action=action(frames[j+1][side]['action'] or {},u)) for side,u in [(enemy,fid),(s,uid)] if u and u in frames[j][0]['observation']['players'][side][2]}) for j in range(max(0,t-10),t+1)]))
        tr.set_actions(players,actual[s]);prev=players
    def metrics(rows):
        return dict(n=len(rows),correct=[sum(r[q]==r['real'] for r in rows) for q in ('old','new')],
          changed=sum(r['old']!=r['new'] for r in rows),wrong_to_right=sum(r['old']!=r['real']==r['new'] for r in rows),right_to_wrong=sum(r['new']!=r['real']==r['old'] for r in rows),
          nll=[float(np.mean([-np.log(max(r['raw_p_real'][j],1e-100)) for r in rows])) for j in (0,1)])
    result=dict(protocol_sha=probe.sha((HERE/'protocol.json').read_bytes()),costs=costs,old=p['prior'],new=p['child'],scores=[a['reward'] for a in frames[-1]],all=metrics(allunits),positive=metrics([r for r in allunits if r['cargo']>0]),heavy=metrics([r for r in allunits if r['cargo']>=200]),sample_changed_steps=sorted({r['step'] for r in allunits if r['old']!=r['new']}),first_two=selected,roots=full,units=allunits)
    dest=HERE/'evidence.json.gz';dest.write_bytes(gzip.compress(json.dumps(result,ensure_ascii=False,allow_nan=False,sort_keys=True,separators=(',',':')).encode(),mtime=0))
    print({k:result[k] for k in ('scores','all','heavy','first_two')},flush=True)

if __name__=='__main__':main()
