"""Tabulate native timing, public outcomes, and the candidates' realized earning attribution."""
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
def read(p):return json.loads((HERE/p).read_bytes())
def get(m,k):return m.get(k,0.)
if __name__=='__main__':
    summary={}
    for group in ('candidate','parent'):
        games=read(f'{group}/results.json')['games']
        summary[group]=[]
        for game in games:
            seat=game['candidate_seat'];s=game['search'];t=s['time']
            econ=read(f'{group}_economics_seat{seat}.json')
            cash=[]
            for k in (seat,1-seat):
                phase=econ['mid_cash'][k]
                cash.append({a:round(get(phase,a),3) for a in ('own_delivery','tree_absorption','tree_collision_cash','plant','seize')})
            credited=[change for change in econ['changes'] if change['purchased_next_action']]
            row=dict(seat=seat,steps=game['actual_transitions'],valid=game['valid'],score=game['official_score_by_seat'],margin=game['margin'],
                     search_invocations=game['search_invocations'],return_types=game['search_return_types'],searched=s['searched'],search_changed=s['changed'],
                     search_aborted=s['aborted'],depth=s['depth'],rollouts=s['rolls'],n_search_seconds=len(t),
                     search_seconds_p50_p95_max=[float(np.quantile(t,p)) for p in (.5,.95,1.)],
                     mean_depth=sum(int(k)*count for k,count in s['depth'].items())/sum(s['depth'].values()),
                     native_call_seconds_by_seat=game['call_seconds_by_seat'],wall_seconds=game['wall_seconds'],midgame_cash_our_then_opponent=cash,
                     counts=econ['counts'],route_funnel=econ['funnel'],paid_next_step_count=len(credited),
                     first_tenure_realized_net=sum(x['first_tenure_net'] for x in credited))
            summary[group].append(row)
    (HERE/'comparison.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
