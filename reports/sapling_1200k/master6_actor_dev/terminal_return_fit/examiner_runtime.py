"""Read-only aggregation of archived original-examiner telemetry and independent historical scores."""
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from tree_win_gate import EXAMINER, validate_examiner_configuration, validate_game


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def pct95(xs):
    return sorted(xs)[math.ceil(.95*len(xs))-1]  # nearest rank, without interpolation


def read_run(directory):
    data=(directory/'results.json').read_bytes()
    report=json.loads(data)
    header=report['header']
    if report['status']!='complete' or header['examiner_md5']!=EXAMINER:
        raise ValueError('incomplete or unexpected opponent')
    rawrows=[]
    for g in report['games']:
        validate_game(g)
        path=directory/g['raw_file']
        packed=path.read_bytes()
        tape=json.loads(gzip.decompress(packed))
        assert sha(packed)==g['raw_sha256']
        validate_examiner_configuration(tape['configuration'])
        assert tape['examiner_md5']==EXAMINER
        assert tape['errors']==[] and tape['audit_messages']==[]
        seat=g['candidate_seat']
        t=g['actual_transitions']
        calls=tape['search_calls']
        assert [e[0] for e in calls]==list(range(t))
        kinds=Counter(e[1] for e in calls)
        ss=g['search']
        assert kinds==g['search_return_types'] and g['search_invocations']==t
        assert kinds['searched']==ss['searched']==sum(ss['depth'].values())==len(ss['time'])
        assert kinds['single']==ss['single']
        opp=g['call_seconds_by_seat'][1-seat]
        assert opp['n']==t and len(g['cpu_affinity'])==1
        row={'map':g['seed'],'candidate_seat':seat,'margin':g['margin'],'steps':t,
             'class_counts':dict(kinds),'changed':ss['changed'],'aborted':ss['aborted'],
             'partial_depth_aborted':ss['aborted']-kinds['depth_zero_return_base'],
             'depth':ss['depth'],'candidates':ss['cands'],'rollouts':ss['rolls'],
             'opponent_public_call_seconds':opp,'opponent_recorded_agent_elapsed_s_searched':{
                 'mean':statistics.mean(ss['time']),'p95_nearest_rank':pct95(ss['time']),'max':max(ss['time'])},
             'cpu_affinity':g['cpu_affinity'],'game_wall_seconds':g['wall_seconds'],'raw_sha256':g['raw_sha256']}
        rawrows.append((row,ss['time']))
    got={(r['map'],r['candidate_seat']) for r,_ in rawrows}
    assert got=={(s,side) for s in header['seeds'] for side in (0,1)}
    return {'results_sha256':sha(data),'quota':header['cpu_quota'],'pack_sha256':header['pack_sha256'],
            'cpu_list':header['cpus'],'games':[r for r,_ in rawrows]},rawrows


def aggregate(rawrows):
    classes=Counter(); depths=Counter(); candidates=Counter(); rolls=Counter()
    for row,_ in rawrows:
        classes.update(row['class_counts']);depths.update(row['depth']);candidates.update(row['candidates']);rolls.update(row['rollouts'])
    n=sum(classes.values()); searched=classes['searched']; xx=[t for _,ts in rawrows for t in ts]
    return {'calls':n,'classes':dict(classes),'changed':sum(r['changed'] for r,_ in rawrows),
            'aborted':sum(r['aborted'] for r,_ in rawrows),'depth':dict(depths),
            'depth_mean_among_searched':sum(int(k)*v for k,v in depths.items())/searched,
            'rollouts':dict(rolls),'candidate_count':dict(candidates),
            'candidate_count_mean_among_searched':sum(int(k)*v for k,v in candidates.items())/searched,
            'public_call_mean_seconds':sum(r['opponent_public_call_seconds']['sum'] for r,_ in rawrows)/n,
            'public_call_max_seconds':max(r['opponent_public_call_seconds']['max'] for r,_ in rawrows),
            'searched_agent_elapsed_p95_nearest_rank_s':pct95(xx), 'game_wall_worker_hours':sum(r['game_wall_seconds'] for r,_ in rawrows)/3600}


def main():
    names=('train_plus','train_minus','dev_parent','dev_learned')
    arms={}; allrows=[]; by_arm={}
    for key in names:
        rec,rows=read_run(HERE/key)
        arms[key]=rec
        allrows.extend(rows)
        by_arm[key]=aggregate(rows)
    recent,rows=read_run(BASE/'tree_exam_it50')
    history=json.loads((BASE/'master3s2_references.json').read_bytes())
    ref=next(x for x in history['analysis'] if x['arm']=='it50')
    archived=subprocess.check_output(['git','show',history['archive_revision']+':'+ref['archive_path']],cwd=ROOT)
    assert sha(archived)==ref['archive_sha256'] and recent['pack_sha256']==ref['candidate_sha256']
    games=[json.loads(x) for x in archived.splitlines()]
    assert len(games)==len({g['seed'] for g in games})==300
    wrappers={g['bh'] for g in games}; harnesses={g['hh'] for g in games}
    assert wrappers=={history['historical_opponent_wrapper_md5']}
    assert harnesses=={hashlib.md5((ROOT/'code/bigeval.py').read_bytes()).hexdigest()}
    assert all(g['ah']==ref['candidate_md5'] and not g['err'] and g['serr']==g['berr']==0 and
               g['win']==(g['margin']>0) and abs(g['margin']-(g['sa']-g['sb']))<.021 and
               g['side']==g['seed']%2 for g in games)
    seats={s:[g for g in games if g['side']==s] for s in (0,1)}
    hist={'blob':ref['archive_path'],'revision':history['archive_revision'],'raw_sha256':sha(archived),
          'candidate_sha256':ref['candidate_sha256'],'wrapper_md5':next(iter(wrappers)),
          'harness_md5':next(iter(harnesses)),'unique_maps':len(games),'wins':sum(g['win'] for g in games),
          'draws':sum(g['margin']==0 for g in games),'margin':statistics.mean(g['margin'] for g in games),
          'by_seat':{k:{'games':len(x),'wins':sum(g['win'] for g in x)} for k,x in seats.items()},
          'observed_opponent_stats_keys':sorted({k for g in games for k in g.get('bst',{})}),
          'overlap_with_eight_maps':sorted({g['seed'] for g in games}&{r['map'] for r,_ in allrows}),
          'overlap_with_two_audited_it50_games':sorted({g['seed'] for g in games}&{r['map'] for r,_ in rows})}
    output={'eight_games':aggregate(allrows),'by_arm':by_arm,'runs':arms,'historical_it50':hist,
            'same_it50_pack_audited_reference':{'directory':'../../tree_exam_it50','run':recent,'summary':aggregate(rows)}}
    (HERE/'examiner_runtime.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'all':output['eight_games'],'subst':{k:(a['depth'].get('8'),a['public_call_mean_seconds']) for k,a in by_arm.items()},'ref':output['same_it50_pack_audited_reference']['summary']},ensure_ascii=False))


if __name__=='__main__':
    main()
