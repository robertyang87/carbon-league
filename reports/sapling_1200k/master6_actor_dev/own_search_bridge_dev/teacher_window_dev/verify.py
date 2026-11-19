"""Independently replay every submitted command and report the registered two-map gate."""
from pathlib import Path
import run
from collections import Counter
import numpy as np
import sys
sys.path.insert(0,str(run.BASE))
from turning_point_ledger import FastCarbon,CashEvents
H=run.HERE;P=run.P
def profile(rows):
    out=dict(calls=len(rows),searched=sum(z['delta']['searched'] for z in rows),aborted=sum(z['delta']['aborted'] for z in rows),
             changed=sum(z['delta']['changed'] for z in rows),single=sum(z['delta']['single'] for z in rows),zero_completed_attempts=sum(z['zero_depth'] for z in rows))
    for k in ('wall','cpu'):
        z=[r[k] for r in rows]
        out[k]=dict(sum=sum(z),median=float(np.median(z)) if z else None,max=max(z,default=None))
    out['attempted_candidates_depth_rolls']=[dict(Counter(z['last'][j] for z in rows if z['last'] is not None)) for j in range(3)]
    return out
def main():
    result=[]
    for c in P['cases']:
        raw=run.read(run.ROOT/c['source']);s=c['seat'];arms={}
        for arm in ('parent','teacher'):
            out=H/f"{c['seed']}_{s}_{arm}.json.gz";r=run.read(out);assert r['program']==run.sha(H/'run.py') and r['protocol']==run.sha(H/'protocol.json')
            env=run.begin(raw)
            for a in raw['actions'][:150]+r['tape']:
                assert not env.done;env.step(a)
            assert env.done and r['scores']==[z.reward for z in env.states] and all(z.status=='DONE' for z in env.states)
            sim=FastCarbon(raw['game_configuration']).reset(seed=raw['seed'])
            for action in raw['actions'][:150]:sim.step(*action)
            pre=list(sim.cash);sim.cash=CashEvents(sim.cash);channels=[Counter(),Counter(),Counter()]
            for k,action in enumerate(r['tape']):
                sim.cash.events.clear();sim.step(*action)
                phase=0 if k<8 else 1 if k<18 else 2
                for kind,who,v,*attr in sim.cash.events:channels[phase][kind]+=v if who==s else -v
            physical_cash=sim.cash[s]-sim.cash[1-s];roundoff=sum(v for ch in channels for v in ch.values())-(physical_cash-pre[s]+pre[1-s])
            assert abs(roundoff)<1e-6
            arms[arm]=dict(focal_net_cash_flow_150_157_158_167_168_end=[dict(ch) for ch in channels],terminal_adjustment=r['margin']-physical_cash,cash_closure_error=roundoff,sha256=run.sha(out),scores=r['scores'],margin=r['margin'],examiner=profile(r['enemy']),teacher=profile(r['teacher']),
                changed_joint_in_window=sum(z['baseline']!=z['submitted'] for z in r['checks']),
                changed_unit_slots=sum(sum(x['baseline'].get(uid)!=x['submitted'].get(uid) for uid in x['baseline'].keys()|x['submitted'].keys()) for x in r['checks']),
                wall=r['wall'],snapshots=r['snap'],root_op_sha=r['root_op_sha'],enemy_root_cmd=r['tape'][0][1-s],end=env.states[0].observation.step)
        a,b=arms['parent'],arms['teacher'];assert a['root_op_sha']==b['root_op_sha'] and a['enemy_root_cmd']==b['enemy_root_cmd']
        result.append(dict(seed=c['seed'],seat=s,arms=arms,margin_delta=b['margin']-a['margin'],teacher_matched_root=True))
    run.live.save(H/'results.json',dict(protocol=run.sha(H/'protocol.json'),verification=run.sha(Path(__file__)),maps=result,continue_limited_demos=all(z['margin_delta']>0 and z['arms']['teacher']['changed_joint_in_window']>0 for z in result),
                  new_from_opening_games=0,student_wins_credited=0))
if __name__=='__main__':main()
