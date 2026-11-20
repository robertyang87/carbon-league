"""Schedule four truthful matchups on the same development map-seat and verify each actual opponent."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
FAMILY=HERE.parent
ROOT=FAMILY.parents[1]
BASE=ROOT/'reports/sapling_1200k/master6_actor_dev/strategic_destination_learning/development'
STRATEGY=BASE.parent
PLAN=HERE/'protocol.json'
WORKFLOW='master3s2-nn-policy-dev.yml'
SEEDS=(621758608,621758609)
MATCHUPS={'new_vs_v3':('new','v3'),'new_vs_v2':('new','v2'),'new_vs_original':('new','original'),'v2_vs_original':('v2','original')}
ARMS=tuple(MATCHUPS)
EXPOSE='Play all four frozen head-to-head and reference games'
PREVIOUS=FAMILY/'v3_release/development/protocol.json'
PREVIOUS_SHA='5aea3e3c11ad2da124fd191cdae03f4742e75a1d3887007dc5a5396db9eae97a'
NEW_SHA='cd230d40a08f7dea07e261202b865f951b88b3b030febc9828627aa5f47c7d2a'
SOURCE=FAMILY/'advantage_policy_update'
PLANS=HERE/'matchups'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())


def require(ok,why):
    if not ok:raise ValueError(why)


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def persist(path,doc):
    raw=(json.dumps(doc,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    if path.exists():
        require(path.read_bytes()==raw,'Different freeze already exists at '+str(path))
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(raw)


def contract(freeze=False):
    require(sha(PREVIOUS)==PREVIOUS_SHA,'Earlier refereeing contract changed')
    previous=read(PREVIOUS)
    old=previous['files']
    keys=('official_helpers','rules','environment','ladder_runner','worker','container','base_registration',
          'original_game_configuration','current_guide','match_runner','replayer','native_timing_audit','previous_exposure_guard',
          'registered_runner','registered_settler','registered_sparse','observation_replay')
    files={k:old[k] for k in keys}
    paths={'matrix':HERE/'matrix.py','workflow':ROOT/'.github/workflows'/WORKFLOW,'settler':HERE/'settle.py',
           'clearance_code':HERE/'clearance.py','clearance':HERE/'clearance.json','previous_v3_protocol':PREVIOUS,
           'new':SOURCE/'v2_bc.py','new_identity':SOURCE/'release.json','new_builder':SOURCE/'build.py',
           'trained_weights':SOURCE/'weights.npz','training_protocol':SOURCE/'protocol.json','v2_transfer':SOURCE/'v2_transfer_protocol.json',
           'training_code':SOURCE/'train_awr.py','data_code':SOURCE/'data.py'}
    files.update((k,runner.record(path)) for k,path in paths.items())
    files.update(v2=old['parent'],v3=old['challenger'],original=old['opponent'])
    require(all(runner.record(ROOT/c['path'])==c for c in files.values()),'Frozen dependency changed')
    require(files['new']['sha256']==NEW_SHA and read(paths['new_identity'])['online_parent']==files['v2']['sha256']
            and read(paths['new_identity'])['v2_bc_sha']==NEW_SHA and read(paths['new_identity'])['weights']==files['trained_weights']['sha256'],'Incorrect deployed comparison package')
    cleared=read(paths['clearance']);boards=cleared['by_seed']
    require(cleared['passed'] and set(map(int,boards))==set(SEEDS) and not set(boards.values())&set(previous['physical_boards'].values()) and
            not cleared['physical_intersections'] and not cleared['seed_intersections'] and
            cleared['verifier_sha256']==files['clearance_code']['sha256'],'Uncleared development maps')
    tickets=read(paths['training_protocol'])['tickets']
    require(all(r['seed'] not in SEEDS and r['map'] not in boards.values() for r in tickets),'Exposed training/validation map')
    for name,(ours,theirs) in MATCHUPS.items():
        sub={'arm':name,'candidate_identity':ours,'opponent_identity':theirs,'seeds':list(SEEDS),'physical_boards':boards,
             'files':{name:files[ours],name+'_weights':files['trained_weights'] if ours=='new' else files[ours],'opponent':files[theirs]}}
        dest=PLANS/(name+'.json')
        if freeze:persist(dest,sub)
        require(read(dest)==sub,'Unexpected opponent or tested package in '+name)
        files['matchup_'+name]=runner.record(dest)
    # A four-by-four Latin square assigns every matchup to each running position once.
    schedule=[]
    for i,seed in enumerate(SEEDS):
        for s in (0,1):
            n=2*i+s;schedule.append(dict(seed=seed,seat=s,order=list(ARMS[n:]+ARMS[:n])))
    return dict(schema='trained-policy-direct-and-reference-screen/1',registration_class='development',arms=list(ARMS),
                matchups=MATCHUPS_as_json(),seeds=list(SEEDS),physical_boards=boards,schedule=schedule,files=files,platform=previous['platform'],
                model_change='Behavior-cloned root action-logit correction learned from archived play, on the online v2 package. Frozen trunk and native opponent model and time budget.',
                amendment='The two-response research package 6f6c93c2 was considered first. Its local unpublished draft 32067bfa was replaced after cd230d40 became available, before dispatch or exposure of either development map. No results on these maps were available to select the candidate.',
                scope='Four complete games per matchup on two fixed prior-unplayed maps, share host per map-seat and independent native random draws. Four independent denominators; no early stopping, replacement or ladder publication.',
                outcomes='Primary head-to-head new vs last delivered v3; also direct vs online v2. New vs original and online v2 vs original are secondary, paired by identical physical map and seat. Report all outcomes, both seats and per-map differences; forfeits retained, no imputation of scores or wins for unscored bilateral failure.')


def MATCHUPS_as_json():
    return {k:{'candidate':a,'opponent':b} for k,(a,b) in MATCHUPS.items()}


def check(digest):
    require(sha(PLAN)==digest and read(PLAN)==contract(),'Matrix identity changed')
    return read(PLAN)


def attach():
    global runner,play,timing,verify_block
    runner=load('direct_paired_runner',BASE/'matrix.py')
    play,timing=runner.play,runner.timing
    runner.HERE,runner.PLAN,runner.SEEDS,runner.ARMS,runner.check=HERE,PLAN,SEEDS,ARMS,check
    native_plan=play.game_plan
    native_verify=runner.verify_game
    def game_plan(master,base,arm,seed,seat):
        sub=PLANS/(arm+'.json')
        require(master['files']['matchup_'+arm]['sha256']==sha(sub),'Mismatched opponent contract')
        ticket=native_plan(read(sub),base,arm,seed,seat)
        ticket.update(comparison_protocol_sha256=sha(PLAN),matrix_sha256=sha(sub))
        return ticket
    def verify_game(dest,seed,seat,arm,master):
        sub=PLANS/(arm+'.json')
        require(master==PLAN and read(master)['files']['matchup_'+arm]['sha256']==sha(sub)
                and read(dest/'registration.json')['comparison_protocol_sha256']==sha(master),'Unbound opponent or matchup')
        game=native_verify(dest,seed,seat,arm,sub)
        who,foe=MATCHUPS[arm];files=read(sub)['files']
        game['physical_role_identities']={'candidate':who,'candidate_sha256':files[arm]['sha256'],
                                        'opponent':foe,'opponent_sha256':files['opponent']['sha256']}
        return game
    play.game_plan,runner.verify_game=game_plan,verify_game
    verify_block=runner.verify_block
    sys.modules['matrix']=sys.modules[__name__]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('prepare','guard','exposure','freeze','check','list','play','verify','observations','settle'))
    parser.add_argument('--matrix-sha');parser.add_argument('--seed',type=int,choices=SEEDS);parser.add_argument('--seat',type=int,choices=(0,1))
    parser.add_argument('--out',type=Path);parser.add_argument('--image');parser.add_argument('--run',type=int)
    parser.add_argument('--commit');parser.add_argument('--archives',type=Path,nargs='+')
    a=parser.parse_args()
    if a.mode in ('prepare','guard','exposure'):
        sparse=load('sparse_matchups',BASE/'workflow_cli.py')
        sparse.HERE,sparse.SUITE,sparse.EXPOSURE_STEP,sparse.ARMS=HERE,WORKFLOW,EXPOSE,ARMS
        if a.mode=='prepare':sparse.prepare(a.matrix_sha)
        elif a.mode=='guard':sparse.prior().guard(a.seed,a.seat)
        else:sparse.prior().exposure(a.out,a.seed,a.seat)
        return 0
    attach()
    if a.mode=='freeze':
        persist(PLAN,contract(freeze=True));print(sha(PLAN));return 0
    plan=check(a.matrix_sha)
    if a.mode=='check':return 0
    if a.mode=='list':
        print(json.dumps({'include':[{'seed':t['seed'],'seat':t['seat']} for t in plan['schedule']]}));return 0
    if a.mode=='settle':
        result=load('independent_four_matchups',HERE/'settle.py').settle(a.run,a.commit,a.archives,a.out,a.matrix_sha)
        return int(not result['records_reconciled'])
    require(a.seed in SEEDS and a.seat in (0,1) and a.out,'Unregistered ticket, seat or directory')
    if a.mode=='play':return runner.run(a.matrix_sha,a.seed,a.seat,a.out,a.image)
    if a.mode=='verify':
        verified=verify_block(a.out,a.seed,a.seat)
        print(json.dumps(verified,allow_nan=False));return int(bool(verified['verification_errors']))
    replay=load('actual_public_views',STRATEGY/'native_action_onpolicy_intake/observations.py')
    for arm in ARMS:
        dest=a.out/arm
        if (dest/'turns.jsonl').exists():(dest/'observations.jsonl.gz').write_bytes(replay.recover(dest))
    return 0


if __name__=='__main__':raise SystemExit(main())
