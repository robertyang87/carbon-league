"""Collect the four registered original-search v1 trajectories, then replay and verify all four official scores."""
import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
ROOT=BASE.parents[1]
FIT=HERE.parent/'terminal_return_fit'
sys.path[:0]=[str(FIT),str(BASE)]
import trial
import tree_win_gate as gate

PLAN=trial.read(HERE/'protocol.json')
DIR=HERE/'source'
ENV={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def check_maps():
    receipt=trial.read(HERE/'clearance.json')
    source=PLAN['train_maps'];held=PLAN['development_maps']
    if len(set(source+held))!=4 or receipt['verdict']!='eligible_in_checked_scope':
        raise ValueError('map allocation conflict')
    for seed in source+held:
        if receipt['boards'].get(str(seed))!=gate.map_identity.local_board(seed):
            raise ValueError('terrain conflict')
    return receipt['boards']

def verify():
    boards=check_maps()
    result=trial.read(DIR/'results.json')
    official=trial.read(DIR/'official_replay.json')['runs'][DIR.name]
    expected={(s,p) for s in PLAN['train_maps'] for p in PLAN['physical_seats']}
    by_key={(r['seed'],r['candidate_seat']):r for r in result['games']}
    by_off={(r['seed'],r['seat']):r for r in official['games']}
    head=result['header']
    if (set(by_key)!=expected or set(by_off)!=expected or len(result['games'])!=len(expected) or len(official['games'])!=len(expected) or
            head['pack_sha256']!=PLAN['parent_pack_sha256'] or head['weight_sha256']!=PLAN['parent_weight_manifest_sha256'] or
            official['pack_sha256']!=PLAN['parent_pack_sha256'] or head['seeds']!=PLAN['train_maps'] or result['status']!='complete'):
        raise ValueError('source collection differs from registration')
    rows=[]
    for key in sorted(expected):
        r=by_key[key];o=by_off[key];s,p=key
        path=DIR/r['raw_file']
        tape=json.loads(gzip.decompress(path.read_bytes()))
        gate.validate_examiner_configuration(tape['configuration'])
        gate.validate_game(r)
        gate.verify_raw(DIR,r,head['pack_sha256'],head['weight_sha256'],expected_terrain=boards[str(s)])
        if (o['official_reward_by_seat']!=r['official_score_by_seat'] or o['statuses']!=['DONE','DONE']
                or o['transcript_sha256']!=sha(path) or o['steps']!=r['actual_transitions'] or o['map_sha256']!=boards[str(s)]
                or tape['audit_messages']!=[] or head['examiner_md5']!=PLAN['examiner_md5']):
            raise ValueError(f'original source or terminal differences at {key}')
        rows.append({'seed':s,'seat':p,'raw_path':str(path.relative_to(ROOT)),'raw_sha256':sha(path),
                     'map_sha256':o['map_sha256'],'steps':o['steps'],'scores':o['official_reward_by_seat'],
                     'official_margin':r['margin'],'termination':r['termination'],'outcome':r['result'],
                     'search_types':r['search_return_types'],'wall_seconds':r['wall_seconds']})
    trial.frozen_file(HERE/'source_receipt.json',{'protocol_sha256':sha(HERE/'protocol.json'),'clearance_sha256':sha(HERE/'clearance.json'),
                                                'official_sha256':sha(DIR/'official_replay.json'),'games':rows})
    return rows


def play():
    check_maps()
    if PLAN['cpu_affinity']!=[0,1]:
        raise ValueError('expected two allotted workers')
    DIR.mkdir(exist_ok=True)
    with (DIR/'.source_schedule.lock').open('a') as lock, tempfile.TemporaryDirectory() as tmp:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pack,weights,fingerprint=trial.recipe('learned',Path(tmp))
        if (fingerprint!=PLAN['parent_pack_sha256'] or sha(weights)!=PLAN['parent_weight_manifest_sha256'] or
                sha(ROOT/'data/weights/sapling_1200k/master6_actor_v1.npz')!=PLAN['student_npz_sha256']):
            raise ValueError('actual v1 recipe or genuine Sapling NPZ mismatch')
        subprocess.run([sys.executable,'-u',str(BASE/'tree_pack_arena.py'),'--pack',str(pack),'--weights',str(weights),
                        '--pack-sha256',fingerprint,'--maps',*map(str,PLAN['train_maps']),'--cpus',*map(str,PLAN['cpu_affinity']),
                        '--run',str(DIR)],env=ENV,check=True)
        out=Path(tmp)/'official.json'
        subprocess.run([sys.executable,str(BASE/'official_game_replay.py'),str(DIR),'--out',str(out)],env=ENV,check=True)
        trial.frozen_file(DIR/'official_replay.json',trial.read(out))
    print(json.dumps(verify()),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('play','verify'));ap.add_argument('--await-clearance',action='store_true')
    args=ap.parse_args()
    if args.await_clearance:
        while not (HERE/'clearance.json').is_file():
            time.sleep(3)
    if args.mode=='play':
        play()
    else:
        verify()
