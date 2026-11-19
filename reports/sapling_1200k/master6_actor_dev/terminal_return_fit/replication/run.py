"""Execute or audit the eight registered development games. A partial run keeps its original seats."""
import argparse
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean, stdev
import subprocess
import sys
import tempfile

HERE=Path(__file__).resolve().parent
FIT=HERE.parent
BASE=FIT.parents[1]
ROOT=BASE.parents[1]
sys.path[:0]=[str(FIT),str(BASE)]
import trial
import tree_win_gate as gate

PLAN=trial.read(HERE/'protocol.json')
MAPS=PLAN['maps']
ARMS=PLAN['arms']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_sources():
    if (len(MAPS)!=2 or len(set(MAPS))!=2 or PLAN['physical_seats']!=[0,1] or
            len(set(MAPS+[PLAN['prior_training_seed'],PLAN['prior_development_trigger_seed']]))!=4):
        raise ValueError('invalid registered map or seat set')
    dev=trial.read(FIT/'development.json')
    if (digest(FIT/'development.json')!=PLAN['trigger_sha256'] or not dev['replication_signal'] or
            digest(FIT/'learned_checkpoint.json')!=PLAN['checkpoint_sha256'] or
            trial.read(FIT/'learned_checkpoint.json')!=trial.compute_fit()):
        raise ValueError('prior results or frozen training checkpoint changed')
    student=ROOT/'data/weights/sapling_1200k/master6_actor_v1.npz'
    if digest(student)!=PLAN['student_npz_sha256']:
        raise ValueError('wrong underlying Sapling NPZ')
    clearance=trial.read(HERE/'map_clearance.json')
    if (clearance['verdict']!='eligible_in_checked_scope' or set(map(int,clearance['boards']))!=set(MAPS) or
            [gate.map_identity.local_board(s) for s in MAPS]!=[clearance['boards'][str(s)] for s in MAPS]):
        raise ValueError('new development maps not cleared')
    return clearance['boards']


def previous_jobs_released():
    prior=FIT.parent/'freight_deployment'/'first_response'
    for name in ('transfer_parent','transfer_candidate'):
        doc=trial.read(prior/name/'official_check.json')
        if doc['native_results_sha256']!=digest(prior/name/'results.json'):
            raise ValueError('previous transfer certificate is stale')
        games=doc['games']
        if {g['seat'] for g in games}!={0,1} or len(games)!=2 or any(g['statuses']!=['DONE','DONE'] for g in games):
            raise ValueError(f'waiting for completed transfer games and official replays in {name}')


def run_arm(arm):
    verified_sources()
    previous_jobs_released()
    directory=HERE/arm
    directory.mkdir(exist_ok=True)
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    with (directory/'.orchestrator.lock').open('a') as lock, tempfile.TemporaryDirectory() as tmp:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pack,weights,pack_sha=trial.recipe(arm,Path(tmp))
        config=PLAN['pack_identity'][arm]
        if digest(pack)!=pack_sha or pack_sha!=config['pack_sha256'] or digest(weights)!=config['weight_sha256']:
            raise ValueError('frozen pack or weight manifest mismatch')
        cpu_list=PLAN['cpu_allocation'][arm]
        subprocess.run([sys.executable,'-u',str(BASE/'tree_pack_arena.py'),'--pack',str(pack),'--weights',str(weights),
                        '--pack-sha256',pack_sha,'--maps',*map(str,MAPS),'--cpus',*map(str,cpu_list),'--run',str(directory)],
                       env=env,check=True)
        official=Path(tmp)/'verified.json'
        subprocess.run([sys.executable,str(BASE/'official_game_replay.py'),str(directory),'--out',str(official)],env=env,check=True)
        trial.frozen_file(directory/'official_replay.json',trial.read(official))


def rows(arm, boards):
    directory=HERE/arm
    result=trial.read(directory/'results.json')
    header=result['header']
    official=trial.read(directory/'official_replay.json')
    off=official['runs'][arm]
    ident=PLAN['pack_identity'][arm]
    if (result['status']!='complete' or header['seeds']!=MAPS or header['examiner_md5']!=PLAN['examiner_md5']
            or off['pack_sha256']!=ident['pack_sha256'] or any(header[k]!=v for k,v in ident.items())
            or len(result['games'])!=4 or len(off['games'])!=4):
        raise ValueError(f'{arm} incomplete or identities do not match')
    engine=official['engine_source_sha256']
    prior_engine=trial.read(FIT/'dev_parent'/'official_replay.json')['engine_source_sha256']
    if engine!=prior_engine:
        raise ValueError('official interpreter changed since the original development run')
    keys={(s,p) for s in MAPS for p in PLAN['physical_seats']}
    official_rows={(r['seed'],r['seat']):r for r in off['games']}
    actual_rows={(r['seed'],r['candidate_seat']):r for r in result['games']}
    if set(official_rows)!=keys or set(actual_rows)!=keys:
        raise ValueError('missing or substituted game')
    checked={}
    for key in sorted(keys):
        r=actual_rows[key]
        s,p=key
        o=official_rows[key]
        raw=directory/r['raw_file']
        tape=json.loads(gzip.decompress(raw.read_bytes()))
        gate.validate_game(r)
        gate.validate_examiner_configuration(tape['configuration'])
        gate.verify_raw(directory,r,ident['pack_sha256'],ident['weight_sha256'],expected_terrain=boards[str(s)])
        m=o['official_reward_by_seat'][p]-o['official_reward_by_seat'][1-p]
        if (o['official_reward_by_seat']!=r['official_score_by_seat'] or o['statuses']!=['DONE','DONE']
                or o['map_sha256']!=boards[str(s)] or o['steps']!=r['actual_transitions']
                or o['transcript_sha256']!=digest(raw) or o['result']!=r['result'] or not math.isclose(m,r['margin'])
                or tape['examiner_md5']!=PLAN['examiner_md5'] or tape.get('audit_messages')!=[]):
            raise ValueError(f'{arm} {key} official or source audit failed')
        checked[key]={'seed':s,'seat':p,'official_scores_by_seat':o['official_reward_by_seat'],'margin':m,
                      'U':trial.terminal(m)[0],'result':r['result'],'steps':o['steps'],'termination':r['termination'],
                      'raw_sha256':r['raw_sha256'],'source_search_types':r['search_return_types'],
                      'wall_seconds':r['wall_seconds']}
    return checked


def compare():
    boards=verified_sources()
    arms={arm:rows(arm,boards) for arm in ARMS}
    players={}
    for arm,items in arms.items():
        def summarize(rr):
            return {'games':len(rr),'w_d_l':[sum(r['result']==k for r in rr) for k in ('win','draw','loss')],
                    'mean_margin':mean(r['margin'] for r in rr),'mean_U':mean(r['U'] for r in rr)}
        players[arm]={'all':summarize(list(items.values())),
                      'by_physical_seat':{p:summarize([r for (seed,seat),r in items.items() if seat==p]) for p in (0,1)}}
    pairs=[]
    for (s,p),pa in sorted(arms['parent'].items()):
        le=arms['learned'][s,p]
        pairs.append({'seed':s,'seat':p, 'parent':pa,'learned':le,
                      'margin_delta':le['margin']-pa['margin'],'U_delta':le['U']-pa['U']})
    maps=[]
    for s in MAPS:
        qr=[r for r in pairs if r['seed']==s]
        maps.append({'seed':s,'terrain_sha256':boards[str(s)],'mean_margin_delta':mean(r['margin_delta'] for r in qr),
                     'mean_U_delta':mean(r['U_delta'] for r in qr),'both_U_higher':all(r['U_delta']>0 for r in qr)})
    gains=[m['mean_margin_delta'] for m in maps]
    se=stdev(gains)/math.sqrt(len(gains))
    center=mean(gains)
    return {'schema':'terminal-return-fit-paired-replication-v1','status':'complete','new_games':8,
            'registration_sha256':digest(HERE/'protocol.json'),'runner_sha256':digest(Path(__file__)),
            'clearance_sha256':digest(HERE/'map_clearance.json'),'checkpoint_sha256':PLAN['checkpoint_sha256'],
            'prior_training_map':PLAN['prior_training_seed'],
            'prior_activation_map_reported_separately':trial.read(FIT/'development.json'),
            'replication_arms':players,'new_map_seats':pairs,'map_clusters':maps,
            'mean_new_map_margin_delta':center,'mean_new_map_U_delta':mean(m['mean_U_delta'] for m in maps),
            'illustrative_t95_map_interval':[center-12.706204736*se,center+12.706204736*se],
            'cluster_count':2,'cluster_se':se,'degrees_of_freedom':1,
            'both_map_directional_rule':all(m['mean_margin_delta']>0 and m['both_U_higher'] for m in maps),
            'formal':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['parent','learned','evaluate'])
    cmd=parser.parse_args().command
    if cmd=='evaluate':
        evidence=compare()
        trial.frozen_file(HERE/'comparison.json',evidence)
        print(json.dumps(evidence['replication_arms']))
    else:
        run_arm(cmd)
