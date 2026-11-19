"""Short local certification; no terminal trajectory, new training label or fresh development map."""
import os
from pathlib import Path
import time
from shared import HERE,fixture,play,runner,SharedEnemyRoot,record_transition,require

PLAN='7b917c1a53ace904fab729a1c83186dac53bebf4bea990bfda6697ed27351631'


class RequestLog:
    """Count physically executed original Player.act calls, outside snapshots and replay."""
    def __init__(self,player,live):
        self.player,self.entries=player,[]
        actual=player.act
        def observe(obs,cfg):
            t,c=time.perf_counter(),time.process_time()
            cmd=actual(obs,cfg)
            wall,cpu=time.perf_counter()-t,time.process_time()-c
            visible={k:v for k,v in obs.items() if k!='remainingOverageTime'}
            self.entries.append(dict(id=len(self.entries),step=obs['step'],wall=wall,cpu=cpu,
                  game_observation_sha256=play.sha(live.encoded(visible)),
                  observation_sha256=play.sha(live.encoded(obs)),order=dict(cmd)))
            return cmd
        player.act=observe

    def __call__(self,obs,cfg):
        n=len(self.entries)
        measurement,_=runner.action(self.player,obs,cfg,original=True)
        require(len(self.entries)==n+1 and self.entries[-1]['order']==measurement['order'],'Unexpected public request count')
        return measurement,n


def run():
    require(fixture.hfile(HERE/'protocol.json')==PLAN,'Different fixture')
    plan=play.read(HERE/'protocol.json')
    require(fixture.hfile(fixture.HERE/'receipt.json')==plan['source_fixture'],'Wrong base mechanism')
    parameters=plan['fixture'];seed=parameters['training_seed'];seat=parameters['seat']
    terrain=play.read(play.INTAKE/'protocol.json')['physical_boards'][str(seed)]
    runner.set_affinity()
    env,cfg,live=play.official(seed,terrain,seat)
    roster=play.read(play.HERE/'protocol.json')['actors']
    orig=play.audit_menu.collect.Player(play.ROOT/roster['original']['file'])
    own=play.audit_menu.collect.Player(play.ROOT/roster['student']['file'])
    require(fixture.hfile(orig.mod.__file__)==plan['source_examiner'] and fixture.hfile(own.mod.__file__)==plan['source_student'],'Source changed')
    require(orig.mod._SEED is None and orig.mod._TEMP==1 and orig.mod._S_ABORT==.58 and orig.mod._SEARCH_OK,'Different original')
    runner.set_affinity();machine=runner.resources()
    request=RequestLog(orig,live)
    bank=[5.,5.];prefix=[];probes=[];saved=None;h=parameters['joint_steps_per_arm']
    while not env.done and env.states[0].observation.step<=max(parameters['registered_steps']):
        t=env.states[0].observation.step
        snap=fixture.LivePosition(env,orig,own,bank) if t in parameters['registered_steps'] else None
        student,event=runner.action(own,live.observe(env,seat,bank[seat]),cfg)
        if snap is not None:
            probes.append(dict(step=t,supported=event is not None))
            if event is not None:saved=snap;menu=event;break
        enemy,num=request(live.observe(env,1-seat,bank[1-seat]),cfg)
        bank,row=record_transition(env,live,bank,terrain,seat,student,enemy,event,num,False);prefix.append(row)
    doc=dict(schema='sapling-native-shared-simultaneous-enemy-certification/1',
             seed=seed,seat=seat,terrain=terrain,pid=os.getpid(),protocol_sha256=PLAN,new_full_games=0,terminal_labels=0,
             prefix=prefix,probes=probes,resources=machine,sources=runner.source_shas()|{
                str(p.relative_to(play.ROOT)):fixture.hfile(p) for p in (Path(__file__).resolve(),HERE/'shared.py')})
    if saved is None:
        doc.update(status='no_registered_supported_root',native_requests=request.entries);return doc
    root=SharedEnemyRoot(saved,cfg,live,seat,terrain,menu,request)
    root_step=len(prefix);branches={}
    for index in parameters['arm_order']:
        bank,first,proof=root.issue(index);steps=[first]
        for _ in range(h-1):
            require(not env.done,'Unexpected terminal in short mechanism fixture')
            student,event=runner.action(own,live.observe(env,seat,bank[seat]),cfg)
            enemy,num=request(live.observe(env,1-seat,bank[1-seat]),cfg)
            bank,row=record_transition(env,live,bank,terrain,seat,student,enemy,event,num,False);steps.append(row)
        branches['AB'[index]]=dict(steps=steps,**proof)
    require(branches['A']['steps'][0]['orders'][seat]!=branches['B']['steps'][0]['orders'][seat] and
            branches['A']['steps'][0]['original']==branches['B']['steps'][0]['original']==root.enemy,
            'Wrong shared order or cost')
    for arm in branches.values():
        bank=saved.restore()
        for r in arm['steps']:
            require(fixture.public_hashes(live,env,bank)==r['public_before_by_seat'],'Changed pre-observations')
            env.step(r['orders']);bank=r['bank_after']
            require(fixture.public_hashes(live,env,bank)==r['public_after_by_seat'] and
                    play.board_hash(env,terrain)==r['board_after_sha256'],'Changed board or visible successor')
        arm['replayed_transitions']=len(arm['steps'])
    saved.restore()
    require([x['step'] for x in request.entries]==list(range(root_step))+[root_step]+list(range(root_step+1,root_step+h))*2,
            'Native original request count is not one shared root and live branch-specific next steps')
    doc.update(status='verified',root_step=root_step,shared=root.receipt,branches=branches,native_requests=request.entries,
               physical_original_calls=len(request.entries),conditional_branch_root_call_debits=2,
               physical_root_calls=1,restored_final=True)
    return doc

if __name__=='__main__':
    require(all(os.environ.get(k)=='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')),'BLAS settings')
    out=HERE/'receipt.json'
    require(not out.exists(),'Use published fixture')
    v=run();play.save(out,v);print(v['status'],fixture.hfile(out))
