"""Classify each realized failed setup and list any delayed payments separately."""
from pathlib import Path
import gzip,json,sys
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
import tree_audited_game as game
from turning_point_ledger import CashEvents
def follow():
    report=json.loads((HERE/'candidate/results.json').read_bytes());rows=[]
    for g in report['games']:
        seat=g['candidate_seat'];changes=json.loads((HERE/f'candidate_economics_seat{seat}.json').read_bytes())['changes']
        raw=json.loads(gzip.decompress((HERE/'candidate'/g['raw_file']).read_bytes()))
        sim=game.exam.FastCarbon(raw['game_configuration']).reset(seed=raw['seed']);sim.cash=CashEvents(sim.cash)
        events=[];landings={}
        for t,orders in enumerate(raw['actions']):
            for c in changes:
                if t==c['root']+1:
                    uid=c['worker'];w=sim.workers.get(uid);cell=c['proposed_cell'];tree=next((v for v in sim.trees.values() if v.pos==cell),None)
                    if w is not None and w.pos!=cell:raise ValueError('new landing not reached')
                    landings[c['root']]={'worker_alive_on_next_root':w is not None,'command_on_next_root':orders[seat].get(uid) if w else None,
                                  'actual_tree_then':None if tree is None else {'seat':tree.player,'age':tree.age},
                                  'cash_then':sim.cash[seat]}
            sim.cash.events=[];sim.step(*orders)
            for kind,s,delta,*detail in sim.cash.events:
                if s==seat and kind in ('plant','seize'):events.append((t,kind,detail[0],detail[1],-delta))
        for c in changes:
            t=c['root'];uid=c['worker'];cell=c['proposed_cell']
            recent=[(et,k,w,x,amt) for et,k,w,x,amt in events if t<et<=t+5 and (w==uid or x==cell)]
            rows.append({'seat':seat,'root':t,'worker':uid,'cell':cell,'predicted_payment_probability':c['prob_pay'],
                         'next_action_purchased':c['purchased_next_action'],**landings[t],'next_5_other_payments_same_worker_or_cell':recent})
    (HERE/'landing_followup.json').write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':follow()
