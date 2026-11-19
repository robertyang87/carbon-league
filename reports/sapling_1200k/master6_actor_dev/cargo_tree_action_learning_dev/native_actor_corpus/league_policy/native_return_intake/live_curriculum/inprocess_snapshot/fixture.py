"""Exercise two distinct physical Sapling orders under a common live, restored native game history."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import play
import official_state
import original_state
import student_state

PROTOCOL_SHA='b73ef0608c890264822a814584043ccf643cec269b9755681de2073c28f7160e'


def require(ok,why):
    if not ok:raise ValueError(why)


def hfile(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class LivePosition:
    """A reusable bundle of three snapshots, including the two live agents' separate histories."""
    def __init__(self,env,orig,student,clockbank):
        self.env=env;self.orig=orig;self.student=student
        self.saved_official=official_state.take(env,clockbank)
        self.saved_original=original_state.take(orig.mod)
        self.saved_student=student_state.snapshot(student)
        self.fingerprint=self.signature(clockbank)
        self.sampler=play.student_history.behavior_rng_snapshot(student)
        self.examiner_rng=play.state_hash(orig.mod._ST['rng'].get_state())

    def signature(self,clockbank):
        return dict(official=official_state.fingerprint(self.env,clockbank),
                    original=original_state.fingerprint(self.orig.mod),student=student_state.fingerprint(self.student))

    def restore(self):
        original_state.restore(self.orig.mod,self.saved_original)
        student_state.restore(self.student,self.saved_student)
        bank=official_state.restore(self.env,self.saved_official)
        require(self.signature(bank)==self.fingerprint,'Whole live restoration fingerprint mismatch')
        require(self.sampler==play.student_history.behavior_rng_snapshot(self.student) and
                self.examiner_rng==play.state_hash(self.orig.mod._ST['rng'].get_state()),'Private generator changed')
        return list(bank)


def student_call(actor,obs,cfg,choice):
    for tid in Path('/proc/self/task').iterdir():
        try:os.sched_setaffinity(int(tid.name),{0})
        except ProcessLookupError:pass
    clock=time.perf_counter();cpu=time.process_time()
    if obs['step']==0:
        require(choice is None,'Cannot intervene during cold start')
        order,event=actor.act(obs,cfg),None
    else:
        order,event=student_state.act_with_choice(actor,obs,cfg,choice=choice)
    return dict(order=order,wall=time.perf_counter()-clock,cpu=time.process_time()-cpu),event


def original_call(original,obs,cfg):
    for tid in Path('/proc/self/task').iterdir():
        try:os.sched_setaffinity(int(tid.name),{1})
        except ProcessLookupError:pass
    m=original.mod
    counters=('single','searched','aborted','changed')
    before={k:m._S_STATS[k] for k in counters}
    wall=time.perf_counter();cpu=time.process_time()
    order=original.act(obs,cfg)
    elapsed=time.perf_counter()-wall;compute=time.process_time()-cpu
    after={k:m._S_STATS[k]-before[k] for k in counters}
    if after['single']:
        require(not after['searched'] and not after['aborted'],'Invalid native single-candidate counters')
        n,depth,rolls,native=1,0,0,None
    else:
        n,depth,rolls,native,*_=m._S_STATS.get('last',())
        require(n>=2 and ((after['searched']==1 and depth>0) or
                          (after['searched']==0 and after['aborted']==1 and depth==0)),
                f'Unrepresented native search transition at {obs["step"]}: {after}')
    return dict(order=order,wall=elapsed,cpu=compute,core=1,search=dict(counts=after,candidates=n,
                depth=depth,rolls=rolls,elapsed=native,measured_cost=list(m._S_COST)))


def public_hashes(live,env,bank):
    return [play.sha(live.encoded(live.observe(env,s,bank[s]))) for s in (0,1)]


def transition(env,live,bank,terrain,student_order,orig_order):
    before=public_hashes(live,env,bank);t=env.states[0].observation.step
    next_bank=[bank[0]-max(student_order['wall']-1.,0),bank[1]-max(orig_order['wall']-1.,0)]
    orders=[student_order['order'],orig_order['order']]
    env.step(orders)
    entry=dict(step=t,public_before_by_seat=before,orders=orders,bank_after=next_bank,
               public_after_by_seat=public_hashes(live,env,next_bank),board_after_sha256=play.board_hash(env,terrain),
               student=student_order,original=orig_order)
    return next_bank,entry


def run():
    plan=play.read(HERE/'protocol.json')
    require(hfile(HERE/'protocol.json')==PROTOCOL_SHA,'Mechanism fixture changed')
    binding=hfile(play.INTAKE/'hosted/settlement.json')
    require(binding==plan['sources']['hosted_fixture_map_settlement_sha256'],'Fixture training terrain registration changed')
    seed=plan['fixture']['existing_training_seed'];terrain=play.read(play.INTAKE/'protocol.json')['physical_boards'][str(seed)]
    env,cfg,live=play.official(seed,terrain,0)
    actors=play.read(play.HERE/'protocol.json')['actors']
    student=play.audit_menu.collect.Player(play.ROOT/actors['student']['file'])
    original=play.audit_menu.collect.Player(play.ROOT/actors['original']['file'])
    require(hfile(original.mod.__file__)==plan['sources']['original_sha256'] and hfile(student.mod.__file__)==plan['sources']['student_sha256'],
            'Source bytes changed')
    require(original.mod._TEMP==1 and original.mod._SEED is None and original.mod._S_ABORT==.58 and original.mod._SEARCH_OK,'Wrong original')
    probes=plan['fixture']['root_steps'];bank=[5.,5.];pre=[];checks=[]
    root=None
    while not env.done and env.states[0].observation.step<=max(probes):
        t=env.states[0].observation.step;public=live.observe(env,0,bank[0])
        snapshot=LivePosition(env,original,student,bank) if t in probes else None
        own,event=student_call(student,public,cfg,None)
        if t in probes:
            checks.append(dict(step=t,supported=event is not None))
            if event is not None:
                root=snapshot;natural_menu=event;break
        opp=original_call(original,live.observe(env,1,bank[1]),cfg)
        bank,frame=transition(env,live,bank,terrain,own,opp)
        pre.append(frame)
    result=dict(schema='sapling-live-serial-snapshot-mechanism/1',seed=seed,seat=0,terrain=terrain,pid=os.getpid(),
                fixture_protocol_sha256=PROTOCOL_SHA,source_training_settlement_sha256=binding,
                scripts={p.name:hfile(p) for p in (Path(__file__),Path(official_state.__file__),Path(original_state.__file__),Path(student_state.__file__))},
                probes=checks,prefix=pre,new_full_games=0,terminal_labels=0)
    if root is None:
        result.update(status='no_registered_supported_root')
        return result
    bank=root.restore()
    t=env.states[0].observation.step
    root_board=play.board_hash(env,terrain); root_public=public_hashes(live,env,bank)
    arms={}
    for idx,label in enumerate(('A','B')):
        bank=root.restore()
        require(public_hashes(live,env,bank)==root_public and play.board_hash(env,terrain)==root_board,'Different pre-action root')
        steps=[]
        for offset in range(plan['fixture']['horizon_joint_transitions']):
            require(not env.done,'Early terminal in mechanism fixture')
            own,event=student_call(student,live.observe(env,0,bank[0]),cfg,idx if offset==0 else None)
            if offset==0:
                require(event is not None and all(event[k]==natural_menu[k] for k in ('options','p','unforced_choice','slot','actions','prefix','legal','native_logits','route','uid'))
                        and own['order']==event['options'][idx],'Wrong restored support or executed intervention')
                menu=event
            opponent=original_call(original,live.observe(env,1,bank[1]),cfg)
            bank,frame=transition(env,live,bank,terrain,own,opponent)
            steps.append(frame)
        post=root.signature(bank)
        require(post['official']!=root.fingerprint['official'] and post['student']!=root.fingerprint['student'] and
                post['original']!=root.fingerprint['original'],'Successor failed to exercise the snapshotted state')
        arms[label]=dict(root_menu=menu,post_state_sha256=post,steps=steps)
    require(arms['A']['steps'][0]['orders'][0]!=arms['B']['steps'][0]['orders'][0],'No physical intervention')
    for arm in arms.values():
        bank=root.restore()
        for row in arm['steps']:
            require(public_hashes(live,env,bank)==row['public_before_by_seat'],'Rule-only replay ignored prior observation')
            env.step(row['orders']);bank=list(row['bank_after'])
            require(play.board_hash(env,terrain)==row['board_after_sha256'] and public_hashes(live,env,bank)==row['public_after_by_seat'],
                    'Rule-only playback changed state, IDs or public observations')
        arm['replayed_transitions']=len(arm['steps'])
    bank=root.restore()
    require(play.board_hash(env,terrain)==root_board and public_hashes(live,env,bank)==root_public,'Final return failed')
    result.update(status='verified',root_step=t,root_public_by_seat=root_public,root_board_sha256=root_board,
                  root_fingerprints=root.fingerprint,root_original_rng_sha256=root.examiner_rng,
                  root_student_rng_sha256=root.sampler,physical_root_commands=natural_menu['options'],
                  branches=arms,final_root_restored=True)
    return result


if __name__=='__main__':
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        require(os.environ.get(key)=='1','Use identical single-threaded BLAS configuration')
    output=HERE/'receipt.json'
    require(not output.exists(),'Existing fixture receipt: use this registered sample only once')
    result=run()
    play.save(output,result)
    print(result['status'],hfile(output))
