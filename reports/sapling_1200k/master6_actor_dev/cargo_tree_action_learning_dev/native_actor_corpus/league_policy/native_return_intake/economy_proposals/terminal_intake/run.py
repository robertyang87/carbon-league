"""Execute sequentially the two frozen conditional training experiments under the original examiner."""
import argparse
import gzip
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import traceback

import preflight
import select_root
from external_menu import issue_external
from fixture_shared import RequestLog
from shared import SharedEnemyRoot, record_transition, fixture, play, runner, require

HERE = Path(__file__).resolve().parent
SHARED = preflight.SHARED


def release_sources(release):
    require(re.fullmatch('[0-9a-f]{40}',release) is not None,'Pass the full published release commit')
    for cmd in (['rev-parse','--verify',release+'^{commit}'], ['merge-base','--is-ancestor',release,'HEAD']):
        subprocess.run(['git','-C',str(play.ROOT),*cmd],check=True,stdout=subprocess.DEVNULL)
    reg=play.read(HERE/'protocol.json')
    gates=play.read(HERE/'preflight.json')
    bindings=preflight.bindings()
    require(gates['sources']==bindings and gates['observed_historical_roots']==48 and
            gates['scheduled_history_turns']==16,'Different signed preflight inputs')
    for cat,permission in gates['permitted'].items():
        evidence=sum(r['verified_positive_focal_contrast'] for r in gates['rows'] if r['category']==cat)
        require(gates['verified_focal_contrasts_by_category'][cat]==evidence and permission==(evidence>=1),
                'Prelaunch decision does not match historical events')
    registry=play.read(SHARED.parent/'protocol.json')['actors']
    paths={play.ROOT/name for name in bindings}
    paths.update([Path(__file__).resolve(),HERE/'preflight.json',SHARED/'shared.py',SHARED/'fixture_shared.py',
                  SHARED.parent/'protocol.json',play.INTAKE/'protocol.json',Path(play.__file__),Path(runner.__file__),
                  Path(fixture.__file__),Path(runner.original_state.__file__),Path(runner.student_state.__file__),
                  Path(fixture.official_state.__file__)])
    for role in ('student','original'):
        source=play.ROOT/registry[role]['file']
        require(fixture.hfile(source)==reg[role+'_sha256'],'Unregistered deployed '+role)
        paths.add(source)
    signed={}
    for f in sorted(paths):
        name=str(f.relative_to(play.ROOT))
        blob=subprocess.check_output(['git','-C',str(play.ROOT),'show',release+':'+name])
        signed[name]=fixture.hfile(f)
        require(play.sha(blob)==signed[name],'Dirty or unpublished runtime input '+name)
    return reg,gates,signed


def save_requests(folder,requests):
    dest=folder/'native_requests.json.gz'
    tmp=dest.with_suffix('.gz.tmp')
    tmp.write_bytes(gzip.compress(play.encode(requests.entries),mtime=0))
    os.replace(tmp,dest)


def natural(env,cfg,live,student,request,seat,bank,terrain,student_measured=None,event=None):
    if student_measured is None:
        student_measured,event=runner.action(student,live.observe(env,seat,bank[seat]),cfg)
    enemy,rid=request(live.observe(env,1-seat,bank[1-seat]),cfg)
    return record_transition(env,live,bank,terrain,seat,student_measured,enemy,event,rid,False)


def play_fixture(case,reg,folder,release,signed):
    seed,seat,terrain=case['seed'],case['seat'],case['board_sha256']
    category=case['category']
    start=time.perf_counter()
    env,cfg,live=play.official(seed,terrain,seat)
    require(play.sha(bytes(env.states[0].observation.full_carbon))==terrain and
            play.read(play.INTAKE/'protocol.json')['physical_boards'][str(seed)]==terrain,'Wrong training board')
    roster=play.read(SHARED.parent/'protocol.json')['actors']
    cold={}
    actors={}
    for role in ('original','student'):
        t=time.perf_counter()
        actors[role]=play.audit_menu.collect.Player(play.ROOT/roster[role]['file'])
        cold[role]=time.perf_counter()-t
    runner.set_affinity()
    original,student=actors['original'],actors['student']
    mod=original.mod
    require(mod._SEARCH_OK and mod._SEED is None and mod._TEMP==1 and mod._S_ABORT==.58
            and mod._torch.get_num_threads()==2,'Examiner search, temperature or compute changed')
    src=dict(seed=seed,seat=seat,terrain=terrain,sources=signed,release=release)
    requests=RequestLog(original,live)
    resources=runner.resources()
    resources['torch_threads']=mod._torch.get_num_threads()
    bank=[5.,5.]
    probes=[]
    registered=None
    pre=runner.Ledger(folder/'prefix.jsonl.gz')
    try:
        try:
            while not env.done:
                step=env.states[0].observation.step
                measured=event=None
                if step in reg['probe_turns']:
                    pos=fixture.LivePosition(env,original,student,bank)
                    measured,selected=select_root.probe(student,live.observe(env,seat,bank[seat]),cfg,category)
                    probes.append(dict(step=step,supported=selected is not None,actual_A=measured['order']))
                    if selected is not None:
                        registered=(pos,selected)
                        bank=pos.restore()
                        break
                bank,row=natural(env,cfg,live,student,requests,seat,bank,terrain,measured,event)
                pre.append(row,seat)
        finally:
            prefix_receipt=pre.close()
        info=dict(**src,**prefix_receipt,probes=probes,environment=resources,cold=cold,
                  pre_end_step=env.states[0].observation.step,prefix_wall_seconds=time.perf_counter()-start)
        if registered is None:
            info.update(status='no_registered_supported_root',root_step=None,terminal=env.done,
                        official_scores_by_seat=[v.reward for v in env.states],
                        terminal_status=[v.status for v in env.states],terminal_economy=play.econ(env))
            play.save(folder/'prefix.json',info)
            play.save(folder/'finish.json',dict(status=info['status'],seed=seed,seat=seat,physical_calls=len(requests.entries)))
            return
        pos,menu=registered
        t=env.states[0].observation.step
        common=SharedEnemyRoot(pos,cfg,live,seat,terrain,menu,requests)
        root=dict(root_step=t,root_fingerprints=pos.fingerprint,root_public_by_seat=common.receipt['pre_public_by_seat'],
                  root_board_sha256=common.receipt['pre_board_sha256'],root_economy=play.econ(env))
        play.save(folder/'prefix.json',dict(**info,**root,root_menu=menu,root_orders=menu['options'],status='root'))
        play.save(folder/'shared.json',common.receipt)
        save_requests(folder,requests)
        arms={}
        for label in case['arm_order']:
            idx=['A','B'].index(label)
            tick=time.perf_counter()
            child=runner.Ledger(folder/(label+'.jsonl.gz'))
            try:
                bank,first,proof=issue_external(common,idx,
                    lambda p,obs,c,i:select_root.issue_student(p,obs,c,i,category,menu))
                child.append(first,seat)
                while not env.done:
                    bank,row=natural(env,cfg,live,student,requests,seat,bank,terrain)
                    child.append(row,seat)
            finally:
                totals=child.close()
            scores=[v.reward for v in env.states]
            require(all(math.isfinite(x) for x in scores),'Invalid terminal scores')
            margin=scores[seat]-scores[1-seat]
            wdl=1. if margin>0 else 0. if margin<0 else .5
            result=dict(**src,**root,**totals,arm=label,index=idx,root_proof=proof,
                        root_intervention=dict(event=first['student_event'],orders=first['orders']),
                        terminal_step=env.states[0].observation.step,official_scores_by_seat=scores,
                        terminal_status=[v.status for v in env.states],terminal_economy=play.econ(env),
                        margin=margin,wdl=wdl,u=.5*wdl+.25*(1+math.tanh(margin/2000.)),
                        elapsed_seconds=time.perf_counter()-tick,pid=os.getpid())
            play.save(folder/(label+'.json'),result)
            save_requests(folder,requests)
            arms[label]=result
        bank=pos.restore()
        play.save(folder/'finish.json',dict(status='paired_raw',seed=seed,seat=seat,root_step=t,
                  arm_order=case['arm_order'],shared_request_id=common.receipt['request_id'],
                  root_examiner_commands_equal=arms['A']['root_intervention']['orders'][1-seat]==arms['B']['root_intervention']['orders'][1-seat],
                  final_root_restored=pos.signature(bank)==pos.fingerprint,physical_requests=len(requests.entries),
                  second_minus_first_margin=arms['B']['margin']-arms['A']['margin'],
                  second_minus_first_u=arms['B']['u']-arms['A']['u'],elapsed_seconds=time.perf_counter()-start))
    finally:
        save_requests(folder,requests)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release',required=True,help='full frozen implementation commit')
    ap.add_argument('--check',action='store_true',help='verify hashes and preflight without constructing players or starting games')
    ap.add_argument('--seed',type=int,help='execute just this registered training receipt; never reroll an existing claim')
    args=ap.parse_args()
    plan,gates,sources=release_sources(args.release)
    require(args.seed is None or args.seed in {v['seed'] for v in plan['fixtures']},'Invalid training ticket')
    if args.check:
        print(json.dumps(dict(permitted=gates['permitted'],bound_files=len(sources),release=args.release)))
        return
    require(all(os.environ.get(k)=='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')),
            'Set all native thread limit environment variables to 1')
    runner.set_affinity()
    for case in plan['fixtures']:
        if args.seed is not None and args.seed!=case['seed']:
            continue
        seed,seat=case['seed'],case['seat']
        folder=HERE/'outputs'/f'{seed}_{seat}'
        folder.mkdir(parents=True,exist_ok=True)
        with (folder/'claim.json').open('x') as f:
            json.dump(dict(release=args.release,seed=seed,seat=seat,terrain=case['board_sha256'],
                      protocol_sha256=preflight.PLAN_SHA,arm_order=case['arm_order'],source_sha256=sources,
                      category=case['category'],pid=os.getpid(),time=runner.utc()),f)
        if not gates['permitted'][case['category']]:
            play.save(folder/'finish.json',dict(status='preflight_hold',seed=seed,seat=seat,category=case['category']))
            continue
        try:
            play_fixture(case,plan,folder,args.release,sources)
        except BaseException as exc:
            play.save(folder/'failure.json',dict(reason=str(exc),time=runner.utc(),trace=traceback.format_exc()))
            raise


if __name__=='__main__':
    main()
