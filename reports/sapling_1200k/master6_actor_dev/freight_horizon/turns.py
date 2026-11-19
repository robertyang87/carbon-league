"""Freeze forecasts from public pre-turn histories; then compare to matching IDs in official development traces."""
import json
import gzip
from pathlib import Path
from predictor import FreightPredictor
from data import HERE, BASE, FastCarbon,FreightCash,add_step,target,check_cfg,compact_obs,sha,gz


def main():
    native=BASE/'hybrid_candidate';ledger=gz(native/'income.json.gz');proof=json.loads((native/'official_replay.json').read_bytes())
    official={g['seat']:g for r in proof['runs'].values() for g in r['games']}
    model=FreightPredictor(HERE/'model.json');baseline=FreightPredictor(HERE/'baseline.json');cases=[]
    for seat,t in ((0,234),(1,157)):
        p=native/f'raw_621458920_{seat}.json.gz';raw=gz(p);g=next(r for r in ledger['games'] if r['seat']==seat);ref=official[seat]
        assert sha(p.read_bytes())==ref['transcript_sha256'] and ref['official_reward_by_seat']==raw['official_score_by_seat'] and ref['statuses']==['DONE','DONE']
        cfg=raw['game_configuration'];check_cfg(cfg);sim=FastCarbon(cfg).reset(seed=raw['seed']);sim.cash=FreightCash(sim.cash)
        history=[];events=[];pred=old=None;cash=[]
        for step,actions in enumerate(raw['actions']):
            obs=compact_obs(sim.obs(seat),seat)
            if step==t:
                pred=model.predict(obs,history);old=baseline.predict(obs,history);cash=[a[0] for a in obs['players']]
            history=(history+[obs])[-5:];evt=add_step(sim,actions);events.append(evt)
            if [round(v,2) for v in sim.cash]!=g['events'][step]['cash']:raise ValueError('per-turn cash audit differs')
        canonical={(int(j),int(s),u):float(a) for j,s,u,a,kind in g['receipts'] if kind=='own_delivery' and s==1-seat and t<j<=t+10}
        actual={(e['t'],s,u):float(a) for e in events[t:t+10] for u,(s,a) in e['payments'].items() if s==1-seat and u.startswith(f'player-{s}-')}
        if set(actual)!=set(canonical) or max((abs(canonical[k]-a) for k,a in actual.items()),default=0)>.0001:raise ValueError('named-worker receipts do not agree with prior ledger')
        rows=[]
        for u,pr in pred.items():
            frac,hit,gross,principal,hit_t=target(u,pr['cargo'],1-seat,t,events)
            rows.append(dict(uid=u,root_prediction=pr,baseline=old[u],actual_amount_all=gross,actual_principal=principal,actual_fraction=frac,hit=hit,first_t=hit_t))
        rows.sort(key=lambda v:-v['root_prediction']['expected_root_cash_10']);rich=sum(r['root_prediction']['expected_root_cash_10'] for r in rows);base=sum(r['baseline']['expected_root_cash_10'] for r in rows)
        cases.append(dict(seat=seat,t=t,root_public_cash=cash,raw_sha256=sha(p.read_bytes()),ranked=rows,scope='positive cargo at root; replenishment/newly loaded carriers separate',
                          all_enemy_gross=sum(actual.values()),existing_loaded_workers_gross=sum(r['actual_amount_all'] for r in rows),
                          current_cargo=sum(r['root_prediction']['cargo'] for r in rows),current_cargo_paid=sum(r['actual_principal'] for r in rows),
                          predicted_root_cash=rich,baseline_root_cash=base,receipts=[[j,s,u,a] for (j,s,u),a in actual.items()]))
    (HERE/'development_turns.json').write_text(json.dumps(dict(model=sha((HERE/'model.json').read_bytes()),purpose='retrospectively identified actual swings; forecast uses only decision times and their prehistory',cases=cases),indent=1,allow_nan=False)+'\n')
    for c in cases:print(c['seat'],c['t'],c['predicted_root_cash'],c['baseline_root_cash'],c['current_cargo_paid'],c['all_enemy_gross'])

if __name__=='__main__':main()
