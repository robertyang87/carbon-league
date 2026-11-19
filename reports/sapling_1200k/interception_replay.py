"""Recover the live candidate's own selected interventions and one-step realized transfers."""
import argparse
from collections import Counter
import gzip
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'code'))
from fastsim import FastCarbon
from interception_screen import outcome
from interception_fit import sha
from interception_cash import changed_transfers


def agent(path):
    spec=importlib.util.spec_from_file_location('sapling_interception_candidate',path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--actual-candidate',action='store_true')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    pack=ROOT/'data/weights/sapling_1200k/soft_it50_v5_interception.py'
    expected=json.loads((HERE/'interception_build.json').read_text())['candidate_sha256']
    if sha(pack.read_bytes())!=expected:raise ValueError('wrong candidate')
    report=json.loads((a.run/'results.json').read_text())
    rows=[]
    plan={g['id']:{e['step']:e for e in g['changed']} for g in json.loads((HERE/'interception_screen.json').read_text())['games']['v5_original_master3s2']}
    for g in report['games']:
        blob=(a.run/g['raw_file']).read_bytes()
        if sha(blob)!=g['raw_sha256']:raise ValueError('game evidence has changed')
        raw=json.loads(gzip.decompress(blob))
        if raw['pack_sha256']!=(expected if a.actual_candidate else 'dceed7891e9ff25cf9e211e1c9b61d979d27a253b4c0935f2439804372b31389'):
            raise ValueError('wrong recorded policy')
        side=raw['candidate_seat']; key=f"{raw['seed']}/{side}"
        sim=FastCarbon(raw['game_configuration']).reset(raw['seed'])
        choose=agent(pack);counts=Counter();events=[]
        for step,joint in enumerate(raw['actions']):
            picked=choose(sim.obs(side),sim.config)
            base=choose.last_baseline
            if (picked if a.actual_candidate else base)!=joint[side]:
                raise ValueError(f'inconsistent candidate action in {key} at {step}')
            change=picked!=base
            if not a.actual_candidate:
                if change!=(step in plan[key]) or (change and choose.last_interventions!=plan[key][step]['decisions']):
                    raise ValueError(f'packed layer differs from reference inference in {key} at {step}')
            if change:
                frozen=sim.get_state()
                b=outcome(frozen,sim.config,base,joint[1-side],side)
                c=outcome(frozen,sim.config,picked,joint[1-side],side)
                transferred=changed_transfers(frozen,sim.config,base,picked,joint[1-side],side)
                event=dict(step=step,interventions=choose.last_interventions,base_actions=base,selected_actions=picked,
                           **transferred,
                           own_cash_delta=c['our_cash']-b['our_cash'],
                           own_carried_delta=c['our_carried']-b['our_carried'],
                           opposing_cash_delta=c['their_cash']-b['their_cash'],
                           extra_empty_enemy_workers_lost=c['opponent_zero_cargo_workers_lost_in_any_collision']-b['opponent_zero_cargo_workers_lost_in_any_collision'])
                for name,v in event.items():
                    if name not in ('step','interventions','base_actions','selected_actions'):
                        counts[name]+=v
                        if name in ('additionally_captured','capture_forgone','own_transfer_prevented','additional_own_transfer') and v>0:
                            counts[name+'_roots']+=1
                events.append(event)
            sim.step(*joint)
        rows.append(dict(seed=raw['seed'],seat=side,raw_sha256=g['raw_sha256'],steps=len(raw['actions']),
                         changed=len(events),stats=choose.last_stats,totals=counts,events=events,
                         official_score_by_seat=g['official_score_by_seat'],result=g['result']))
    if report['status']!='complete' or not rows:
        raise ValueError('complete archived results required')
    a.out.write_text(json.dumps(dict(candidate_sha256=expected,actual_deployed=a.actual_candidate,
                     ledger_sha256=sha((a.run/'results.json').read_bytes()),games=rows),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
