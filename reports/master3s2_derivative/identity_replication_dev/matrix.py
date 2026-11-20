"""Replicate the unchanged original CRN challenger and parent on four previously unused development maps."""
import argparse
import importlib.util
import json
from pathlib import Path
from statistics import mean
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OLD=HERE.parent/'paired_improvement_dev'
FIRST_SHA='b8d7fd55947142230ff7201f0a3d361ad0759a9692aadce9fffc48d9856c9728'
SEEDS=tuple(range(621758602,621758606))
ARMS=('parent','challenger')
WORKFLOW='master3s2-identity-replication-dev.yml'
EXPOSE='Play the original CRN pair on a registered replication seat'
s=importlib.util.spec_from_file_location('unchanged_pair',OLD/'matrix.py')
initial=importlib.util.module_from_spec(s);s.loader.exec_module(initial)
paired=initial.paired


def contract():
    reference=OLD/'protocol.json'
    paired.require(paired.sha(reference)==FIRST_SHA,'Changed first comparison contract')
    prev=paired.read(reference)
    replaced={'matrix','workflow','clearance','clearance_code'}
    files={k:c for k,c in prev['files'].items() if k not in replaced}
    additions={'matrix':HERE/'matrix.py','workflow':ROOT/'.github/workflows'/WORKFLOW,'clearance':HERE/'clearance.json',
               'clearance_code':HERE/'clearance.py','previous_pair_adapter':OLD/'matrix.py','previous_pair_protocol':reference}
    files.update({k:paired.runner.record(p) for k,p in additions.items()})
    paired.require(all(paired.runner.record(ROOT/c['path'])==c for c in files.values()),'Frozen source byte mismatch')
    census=paired.read(additions['clearance']);boards=census['by_seed']
    paired.require(census['passed'] and census['previous_two_map_sha256']==FIRST_SHA and
                   set(map(int,boards))==set(SEEDS) and len(set(boards.values()))==4 and boards==census['boards'] and
                   not (set(boards.values()) & set(prev['physical_boards'].values())) and
                   not census['seed_intersections'] and not census['physical_intersections'] and
                   census['verifier_sha256']==files['clearance_code']['sha256'],'Incomplete or mismatched four-map clearance')
    schedule=[dict(seed=seed,seat=seat,order=list(ARMS if (i+seat)%2==0 else reversed(ARMS)))
              for i,seed in enumerate(SEEDS) for seat in (0,1)]
    return dict(schema='replication-frozen-original-crn/1',arms=list(ARMS),seeds=list(SEEDS),schedule=schedule,
                physical_boards=boards,files=files,platform=prev['platform'],examiner=prev['examiner'],registration_class='development',
                scope='16 complete games: each unchanged policy plays original Master3s2 in both seats on four new physical maps. One workflow attempt; sequential same-host pair in each map/seat block. Independent natural action sampling.',
                outcomes='Retain all 16 assigned games including execution faults and timeouts, report each policy win fraction on each two-seat map, and differences in paired official margins with equal map weights. Earlier 598/599 development data are reported separately. No optional stopping or map replacement.')


def summarize(original,outcomes,games,seeds):
    result=original(games,seeds)
    z=result['paired']['challenger_minus_parent'];maps=z['by_physical_map']
    def average(rows,key):
        nums=[r[key] for r in rows]
        return mean(nums) if nums and None not in nums else None
    z['leave_one_map_out']=[dict(omitted=r['seed'],retained_maps=[m['seed'] for m in maps if m['seed']!=r['seed']],
                                 mean_margin_change=average([m for m in maps if m['seed']!=r['seed']],'margin_change'),
                                 mean_win_fraction_change=average([m for m in maps if m['seed']!=r['seed']],'win_fraction_change'))
                              for r in maps]
    for arm in ARMS:
        result['arms'][arm]['by_physical_map']=[{'seed':n,**outcomes([g for g in games if g['arm']==arm and g['seed']==n])} for n in seeds]
    result['cluster_unit']='Four independent development maps. Each contains two rotated-seat comparisons; leave-one-map values retain the three remaining maps with equal weight.'
    return result


def main():
    paired.HERE,paired.PLAN,paired.SEEDS,paired.ARMS=HERE,HERE/'protocol.json',SEEDS,ARMS
    paired.EXPOSURE_STEP,paired.WORKFLOW,paired.contract=EXPOSE,WORKFLOW,contract
    if sys.argv[1:2]==['settle']:
        parser=argparse.ArgumentParser()
        parser.add_argument('--run',type=int,required=True);parser.add_argument('--commit',required=True)
        parser.add_argument('--matrix-sha',required=True);parser.add_argument('--out',type=Path,required=True)
        parser.add_argument('--archives',type=Path,nargs='+',required=True)
        a=parser.parse_args(sys.argv[2:]);sys.argv[1:]=['check','--matrix-sha',a.matrix_sha]
        paired.main();referee=paired.load('replication_settlement',paired.PREVIOUS/'settle.py')
        audit=referee.audit;audit.CONTRASTS=(('challenger','parent'),);agg=audit.aggregate
        audit.aggregate=lambda games,seeds:summarize(agg,audit.outcomes,games,seeds)
        report=audit.merge(a.run,a.commit,a.archives,a.out,a.matrix_sha)
        print(json.dumps(report['results'],indent=2))
        return int(not report['records_reconciled'])
    return paired.main()


if __name__=='__main__':raise SystemExit(main())
