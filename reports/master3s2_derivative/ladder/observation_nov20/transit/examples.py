"""Tie two late examples to exact physical actions on previously audited games."""
import sys,gzip,json
from pathlib import Path
p=Path(__file__).resolve().parent
root=Path(sys.argv[1]); sys.path.insert(0,str(root/'reports/sapling_1200k/all_top5'));import economy_extract as E
data=json.loads(gzip.decompress((p/'evidence.json.gz').read_bytes()))
examples={}
for opp,uid,t0,t1 in [('安生N27','player-0-worker-63',285,288)]:
    path=root/'reports/master3s2_derivative/ladder/replays'/(data['games'][opp]['match']+'-1.json.gz')
    d=json.loads(gzip.decompress(path.read_bytes()));frames=d['steps'];sim=E.FastCarbon(d['configuration']).load_observation(frames[0][0]['observation']);sim.cash=E.CashEvents(sim.cash);record=[]
    for t,frame in enumerate(frames[1:],1):
        acts=[row['action'] or {} for row in frame];sim.cash.events.clear();sim.step(*acts)
        if t0<t<=t1:
            s=0;old=frames[t-1][0]['observation']['players'][s][2][uid];new=frame[0]['observation']['players'][s][2][uid]
            record.append(dict(t_pre=t-1,t_after=t,worker=uid,physical_order=acts[s].get(uid,'NONE'),before=old,after=new,all_collision_legs=sim.last_collisions,own_deposit_by_id=[ev for ev in sim.cash.events if ev[0]=='own_delivery' and ev[3]==uid]))
        if t>=t1:break
    examples[opp]=record
for opp,t_list in [('1C-Test',[278,283,284,289]),('claude-AlphaCarbon-Master2s',[289])]:
    examples[opp]=[{k:row[k] for k in ('opp','t','current_cash_gap','forecast_transit','eligible','future_bank','bank_gap','next_cash_gap')} for row in data['roots'] if row['opp']==opp and row['t'] in t_list]
examples['Master2s_end_interceptions']=data['games']['claude-AlphaCarbon-Master2s']['near_terminal_steals']
(p/'examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2)+'\n')
