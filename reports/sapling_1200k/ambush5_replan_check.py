"""Retest 70 isolated source roots with the deployed re-observation adapter."""
import argparse
import gzip
import json
from pathlib import Path

from ambush5_probe import H,HERE,arena,branch


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',required=True,type=Path)
    a=ap.parse_args()
    pack=HERE.parents[1]/'data/weights/sapling_1200k/soft_it50_v5_ambush5.py'
    packed=arena.prior.shared.module(pack,'candidate_replan_check')
    source=json.loads((HERE/'ambush5_exploratory_results.json').read_text())
    result=[]
    for game in source['sapling']['games']:
        seed,me=game['seed'],game['seat']
        payload=(HERE/game['run']/f'raw_{seed}_{me}.json.gz').read_bytes()
        raw=json.loads(gzip.decompress(payload))
        sim=arena.exam.FastCarbon(raw['game_configuration']).reset(seed=seed)
        states,obs=[sim.get_state()],[sim.obs(me)]
        for cmd in raw['actions']:
            sim.step(*cmd)
            states.append(sim.get_state())
            obs.append(sim.obs(me))
        if arena.prior.hashed(states[-1]) != game['final_state_sha256']:
            raise ValueError('archive digest')
        rows=[]
        for row in game['learned']:
            tick=row['tick']; segment=states[tick:tick+H+1]; trail=raw['actions'][tick:tick+H]
            packed._AMBUSH_SEATS.clear()
            packed._AMBUSH_SEATS[me]={'step':tick-1,'previous':obs[tick-1]['players'][1-me][2] if tick else {},'plan':None}
            adm=[]
            def control(observed,orders):
                act,decision=packed._ambush_update(observed,orders)
                adm.append(decision)
                return act
            parent=branch(sim,segment,trail,me,row,apply=False)
            if sim.get_state()!=segment[-1]:
                raise ValueError('parent action reconstruction')
            new=branch(sim,segment,trail,me,row,controller=control)
            if not adm or adm[0]['phase']!='start' or adm[0]['target']!=row['target'] or adm[0]['attacker']!=row['attacker']:
                raise ValueError('start mismatch')
            delta={k:round(new[k]-parent[k],4) for k in ('own_cash','enemy_cash','own_cargo','enemy_cargo','own_worker_deposits','enemy_worker_deposits','enemy_cargo_acquired')}
            rows.append({'tick':tick,'live_decisions':adm,'delta':delta,**{k:round(new[k],4) for k in
                        ('target_captured','target_cargo_banked','target_cargo_still_carried','missing_command_ids','missing_before_first_target_capture')}})
        result.append({'run':game['run'],'seed':seed,'seat':me,'eligible_roots':game['eligible_roots'],'attempts':rows})
    all_rows=[r for g in result for r in g['attempts']]
    def metrics(rows):
        return {'n':len(rows),'caught':sum(r['target_captured']>0 for r in rows),'target_cargo':round(sum(r['target_captured'] for r in rows),3),
                'banked':round(sum(r['target_cargo_banked'] for r in rows),3),'retained':round(sum(r['target_cargo_still_carried'] for r in rows),3),
                'deltas':{k:round(sum(r['delta'][k] for r in rows),3) for k in ('own_cash','enemy_cash','own_cargo','enemy_cargo','own_worker_deposits','enemy_worker_deposits')}}
    out={'pack_sha256':arena.prior.digest(pack),'source':arena.prior.digest(HERE/'ambush5_exploratory_results.json'),'games':result,
         'all':metrics(all_rows),'no_invalid_future_command':metrics([r for r in all_rows if not r['missing_command_ids']])}
    a.out.write_text(json.dumps(out,separators=(',',':'))+'\n')
    print(out['all'],out['no_invalid_future_command'])


if __name__=='__main__':
    main()
