"""Run one natural training game per authorized map in the original examiner's ladder cgroups."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "development"
SAMPLER = HERE.parent / "native_action_onpolicy"
REFERENCE = HERE.parent / "elite_direct_policy/development/protocol.json"
SPEC = importlib.util.spec_from_file_location('shared_ladder_runner', BASE/'matrix.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
ROOT, read, record, require, sha = runner.ROOT, runner.read, runner.record, runner.require, runner.sha
timing, play = runner.timing, runner.play
PLAN = HERE / 'protocol.json'
ARMS = ('collector',)
SEEDS = tuple(range(621758420, 621758428))
WORKFLOW = 'sapling-native-action-onpolicy-intake.yml'
BASE_REF = 'a97d156a28b039ff54e6e0cf55a6cf1c114fdafbfeee7b2c72456fab872949eb'
DESIGN = 'e5e85fac970a59f5c507b6f482d9ff0bf1ec99a7cd5b6dfafb9f7da88fc3d1de'
POLICY = '87655af5309b6c5468f76441da46ce9f4bd7ff923315b2196b7d636994dd0a9b'
WEIGHTS = '6851b91122d2e0b699ae05fe4513b7eaa6b6c2098e1ace14dc0e699470368bf2'


def contract():
    require(sha(REFERENCE)==BASE_REF and sha(HERE/'design.json')==DESIGN, 'Different reference or maps')
    design=read(HERE/'design.json')
    reference=read(REFERENCE)
    keys=('opponent','official_helpers','rules','environment','ladder_runner','worker','container','base_registration',
          'original_game_configuration','current_guide','match_runner','replayer','native_timing_audit','previous_exposure_guard')
    files={key:reference['files'][key] for key in keys}
    require(all(record(ROOT/r['path'])==r for r in files.values()),'Shared original/referee bytes differ')
    inputs = {'collector':SAMPLER/'sampler.py', 'collector_weights':SAMPLER/'behavior.npz',
              'eligibility':SAMPLER/'gate.json','preflight':SAMPLER/'preflight.py','collector_builder':SAMPLER/'build.py',
              'behavior_kernel':SAMPLER/'kernel.py','behavior_telemetry':SAMPLER/'telemetry.py',
              'maps':HERE/'design.json','map_register':HERE/'register_maps.py','matrix':HERE/'matrix.py','cli':HERE/'workflow_cli.py',
              'observation_replay':HERE/'observations.py','settler':HERE/'settle.py', 'referee_reference':REFERENCE,
              'registered_runner':BASE/'matrix.py','registered_settler':BASE/'settle.py','registered_sparse':BASE/'workflow_cli.py',
              'workflow':ROOT/'.github/workflows'/WORKFLOW}
    files.update({k:record(path) for k,path in inputs.items()})
    gate=read(inputs['eligibility'])
    require(gate['eligible_for_training_collection'] is True and gate['strength_evaluation_eligible'] is False
            and gate['sampler']==files['collector']['sha256']==POLICY and gate['head']==files['collector_weights']['sha256']==WEIGHTS
            and 0<files['collector']['bytes']<=10_000_000,'Sampler is not eligible for natural training collection')
    require(files['opponent']['sha256']=='65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5'
            and files['current_guide']['sha256']=='97adbc81e39ef2b1834af96e261cd10e76424352abde5e04868b3467c2a5cec8','Original hash or platform guide drift')
    require(design['schedule']==[dict(seed=s,seat=i%2,order=['collector']) for i,s in enumerate(SEEDS)]
            and len(set(design['physical_boards'].values()))==8,'Wrong natural-training sample')
    return {**design,'platform':reference['platform'],'arms':['collector'],'files':files,'method':'Fresh unseeded action draws under the frozen physical sampler. Immutable original T=1/no RNG assignment/0.58s, full natural opponent response. No winner selection or paired parent. Single-game terminal targets remain training data.'}


def check(digest):
    require(sha(PLAN)==digest and read(PLAN)==contract(),'Frozen protocol or model changed')
    return read(PLAN)


runner.HERE, runner.PLAN, runner.SEEDS, runner.ARMS, runner.check = HERE, PLAN, SEEDS, ARMS, check
verify_block=runner.verify_block


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('freeze','check','list','play','verify'))
    parser.add_argument('--matrix-sha');parser.add_argument('--seed',type=int,choices=SEEDS)
    parser.add_argument('--seat',type=int,choices=(0,1));parser.add_argument('--out',type=Path);parser.add_argument('--image')
    opts=parser.parse_args()
    if opts.mode=='freeze':
        reg=contract()
        if PLAN.exists(): require(read(PLAN)==reg,'Different existing freeze')
        else: play.live.save(PLAN,reg)
        print(sha(PLAN))
    else:
        plan=check(opts.matrix_sha)
        if opts.mode=='list': print(json.dumps({'include':[{'seed':r['seed'],'seat':r['seat']} for r in plan['schedule']]}))
        if opts.mode in ('play','verify'):
            require(any((r['seed'],r['seat'])==(opts.seed,opts.seat) for r in plan['schedule']),'Undeclared map/seat')
            if opts.mode=='play': sys.exit(runner.run(opts.matrix_sha,opts.seed,opts.seat,opts.out,opts.image))
            result=verify_block(opts.out,opts.seed,opts.seat)
            print(json.dumps(result,allow_nan=False));sys.exit(int(bool(result['verification_errors'])))
