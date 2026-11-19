"""Verify full tapes, referee timing and actual candidate actions under the registered focus policy."""
import sys
from rescue import HERE,choose
sys.path.insert(0,str(HERE.parent/'bank_under_pressure_dev'))
import full_ledger,timing,periods
from audit import sr
def check():
    focus={}
    def decision(obs,orders):
        seat=obs['player']
        if obs['step']==0:focus.pop(seat,None)
        if obs['step']<280:return None
        move=choose(obs,orders,focus.get(seat))
        if move:focus[seat]=move['worker']
        return move
    full_ledger.HERE=timing.HERE=HERE
    full_ledger.choose=decision;full_ledger.summary();timing.check()
    rows=sr.load(HERE/'scores.json')
    sr.retain(HERE/'paired_periods.json',[periods.compare(HERE/'games'/r['arm']/f"raw_{r['seed']}_{r['seat']}.json.gz",r['seat']) for r in rows])
    for r in rows:print(r['seat'],r['arm'],r['score_by_seat'],r['margin'])
if __name__=='__main__':check()
