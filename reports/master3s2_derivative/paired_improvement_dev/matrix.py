"""Two-map, cross-seat, same-host paired comparison against the frozen original Master3s2."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
FAMILY=HERE.parent
ROOT=FAMILY.parents[1]
SOURCE=FAMILY/'ladder/cargo_defense/development'
PLAN=HERE/'protocol.json'
RELEASE=HERE/'release.json'
REGISTERED='244d7cc3f531ab0806db3c859a651298cddff962aa63c81e63fc51cebe1fac6e'
ORIGINAL='65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5'
WORKFLOW='master3s2-paired-improvement-dev.yml'
SEEDS=(621758598,621758599)
ARMS=('parent','challenger')
EXPOSE='Play parent and challenger against the original'
spec=importlib.util.spec_from_file_location('paired_development',SOURCE/'matrix.py')
paired=importlib.util.module_from_spec(spec);sys.modules[spec.name]=paired;spec.loader.exec_module(paired)


def persist(path,doc):
    b=(json.dumps(doc,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    if path.exists():
        if b!=path.read_bytes():raise ValueError('A different freeze already exists at '+str(path))
    else:path.write_bytes(b)
    return hashlib.sha256(b).hexdigest()


def card(filename,expected=None):
    p=(ROOT/filename).resolve(strict=True)
    rel=p.relative_to(ROOT).as_posix();raw=p.read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    if expected is not None and expected!=digest:raise ValueError('Verified file hash mismatch: '+rel)
    return dict(path=rel,sha256=digest,bytes=len(raw))


def bind(args):
    parser=argparse.ArgumentParser()
    for role in ('challenger','parent'):
        parser.add_argument('--'+role,required=True)
        parser.add_argument('--'+role+'-sha',required=True)
        parser.add_argument('--'+role+'-weights')
    parser.add_argument('--source',action='append',default=[])
    a=parser.parse_args(args)
    files={role:card(getattr(a,role),getattr(a,role+'_sha')) for role in ARMS}
    for role in ARMS:
        weights=getattr(a,role+'_weights')
        files[role+'_weights']=card(weights) if weights else files[role]
        if files[role]['bytes']>10_000_000 or not files[role]['path'].endswith('.py'):raise ValueError('Ladder single-file form exceeded')
    if files['challenger']['sha256'] in (ORIGINAL,files['parent']['sha256']):raise ValueError('Challenger duplicates an existing policy')
    for i,v in enumerate(a.source):files['source_'+str(i)]=card(v)
    print(persist(RELEASE,dict(files=files,scope='Registered local development comparison; no publication to ladder.')))


def contract():
    require=paired.require
    reference=SOURCE/'protocol.json'
    require(paired.sha(reference)==REGISTERED,'Original runner registration changed')
    original=paired.read(reference)
    keys=('opponent','official_helpers','rules','environment','ladder_runner','worker','container','base_registration',
          'original_game_configuration','current_guide','match_runner','replayer','native_timing_audit','previous_exposure_guard',
          'registered_runner','registered_settler','registered_sparse','observation_replay','paired_settler')
    files={k:original['files'][k] for k in keys}
    release=paired.read(RELEASE)['files']
    require(all(k in release for k in ('parent','parent_weights','challenger','challenger_weights')),'Incomplete release pair')
    files.update(release)
    require(all(paired.runner.record(ROOT/c['path'])==c for c in files.values()),'Referee, challenger or parent byte identity changed')
    require(files['opponent']['sha256']==ORIGINAL and files['challenger']['sha256'] not in (files['parent']['sha256'], ORIGINAL),
            'Wrong examiner or duplicated competing package')
    prereqs={'release_pair':RELEASE,'prior_reference':reference,'previous_unplayed_plan':FAMILY/'return_rank/development/protocol.json',
             'previous_census':FAMILY/'return_rank/development/clearance.json','clearance':HERE/'clearance.json',
             'clearance_code':HERE/'clearance.py','paired_adapter':SOURCE/'matrix.py','matrix':HERE/'matrix.py',
             'workflow':ROOT/'.github/workflows'/WORKFLOW}
    files.update({k:paired.runner.record(v) for k,v in prereqs.items()})
    rec=paired.read(prereqs['clearance']);boards=rec['physical_boards']
    require(rec['passed'] and rec['former_workflow_run_count']==0 and rec['prior_protocol_sha256']==files['previous_unplayed_plan']['sha256']
            and rec['prior_whole_inventory_sha256']==files['previous_census']['sha256'] and not rec['numerical_intersections']
            and not rec['physical_intersections'] and rec['verifier_sha256']==files['clearance_code']['sha256']
            and set(map(int,boards))==set(SEEDS) and len(set(boards.values()))==2,'Development map clearance mismatch')
    schedule=[dict(seed=seed,seat=s,order=list(ARMS if (i+s)%2==0 else reversed(ARMS))) for i,seed in enumerate(SEEDS) for s in (0,1)]
    return dict(schema='master3s2-paired-improvement-development/1',registration_class='development',arms=list(ARMS),seeds=list(SEEDS),
                physical_boards=boards,schedule=schedule,files=files,outcomes=original['outcomes'],scope=original['scope'],
                platform=original['platform'],examiner=original['examiner'])


def main():
    if sys.argv[1:2]==['bind']:
        bind(sys.argv[2:]);return 0
    paired.HERE,paired.PLAN,paired.SEEDS,paired.ARMS=HERE,PLAN,SEEDS,ARMS
    paired.EXPOSURE_STEP,paired.WORKFLOW,paired.contract=EXPOSE,WORKFLOW,contract
    if sys.argv[1:2]==['settle']:
        parser=argparse.ArgumentParser()
        parser.add_argument('--run',type=int,required=True);parser.add_argument('--commit',required=True)
        parser.add_argument('--matrix-sha',required=True);parser.add_argument('--out',type=Path,required=True)
        parser.add_argument('--archives',type=Path,nargs='+',required=True)
        a=parser.parse_args(sys.argv[2:])
        sys.argv[1:]=['check','--matrix-sha',a.matrix_sha]
        paired.main()
        referee=paired.load('improvement_settlement',paired.PREVIOUS/'settle.py')
        referee.audit.CONTRASTS=(('challenger','parent'),)
        scores=referee.audit.merge(a.run,a.commit,a.archives,a.out,a.matrix_sha)
        print(json.dumps(scores['results'],indent=2))
        return int(not scores['records_reconciled'])
    return paired.main()


if __name__=='__main__':raise SystemExit(main())
