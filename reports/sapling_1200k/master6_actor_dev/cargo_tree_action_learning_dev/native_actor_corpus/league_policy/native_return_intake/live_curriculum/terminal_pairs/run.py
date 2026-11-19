"""Execute the two preregistered, same-process training comparisons, never a formal match."""
import argparse
from collections import Counter
import datetime
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
POLICY=HERE.parent
MECHANISM=POLICY/'inprocess_snapshot'
sys.path.insert(0,str(MECHANISM))
import fixture
import original_state
import student_state
import official_state
from fixture import play

COMMIT='4952efa0e6be03d76c38115db60d56db045f87f5'
PLAN_SHA='013c21283c1e48477ebb56c3668bbf79e4c019b86fd38dfb96f839018e5227bd'
CORES={0,1}
require=fixture.require


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def set_affinity():
    for thread in Path('/proc/self/task').iterdir():
        try:os.sched_setaffinity(int(thread.name),CORES)
        except ProcessLookupError:pass
    require(os.sched_getaffinity(0)==CORES,'Cannot bind original adaptive search to the registered two cores')


def resources():
    c=Path('/sys/fs/cgroup')
    return dict(affinity=sorted(os.sched_getaffinity(0)),all_thread_affinities={
        str(x.name):sorted(os.sched_getaffinity(int(x.name))) for x in Path('/proc/self/task').iterdir()},
        cpu_max=(c/'cpu.max').read_text().strip(),memory_max=(c/'memory.max').read_text().strip(),
        omp=os.environ.get('OMP_NUM_THREADS'),blas=os.environ.get('OPENBLAS_NUM_THREADS'),
        local_diagnostic=True)


def action(player,obs,cfg,original=False,choice=None):
    before=None
    if original:
        before={k:player.mod._S_STATS[k] for k in ('searched','single','aborted','changed')}
    t,c=time.perf_counter(),time.process_time()
    if original or obs['step']==0:
        require(choice is None,'Unsupported cold/root intervention')
        order=player.act(obs,cfg);event=None
    else:
        order,event=student_state.act_with_choice(player,obs,cfg,choice)
    data=dict(order=order,wall=time.perf_counter()-t,cpu=time.process_time()-c)
    if original:
        stats=player.mod._S_STATS
        change={k:stats[k]-n for k,n in before.items()}
        if change['single']==1:
            require(change['searched']==change['aborted']==0,'Conflicting original telemetry')
            n,d,rolls,elapsed=1,0,0,None
        else:
            n,d,rolls,elapsed,*_=stats['last']
            require(n>=2 and ((change['searched']==1 and d>0) or (change['aborted']==1 and change['searched']==0 and d==0)),
                    'Unidentified original search execution')
        data['search']=dict(counts=change,candidates=n,depth=d,rolls=rolls,elapsed=elapsed,
                            measured_cost=list(player.mod._S_COST))
    return data,event


class Ledger:
    def __init__(self,path):
        self.path=path;self.count=0;self.sha=hashlib.sha256()
        self.raw=path.open('xb')
        self.zip=gzip.GzipFile(filename='',mode='wb',fileobj=self.raw,mtime=0)
        self.clock=[Counter(),Counter()]
        self.search=Counter()

    def append(self,row,seat):
        raw=play.encode(row);self.sha.update(raw);self.zip.write(raw);self.count+=1
        for who,p in (('student',seat),('original',1-seat)):
            dt=row[who]['wall'];c=self.clock[p]
            c['slow']+=int(dt>1);c['excess']+=max(dt-1.,0);c['hard']+=int(dt>3);c['max']=max(dt,c['max'])
        q=row['original']['search']
        self.search['single' if q['candidates']==1 else 'positive' if q['depth']>0 else 'zero']+=1
        if self.count%20==0:
            self.zip.flush();self.raw.flush()

    def close(self):
        self.zip.close();self.raw.close()
        return dict(turns_sha256=fixture.hfile(self.path),raw_turns_sha256=self.sha.hexdigest(),
                    stats=dict(steps=self.count,clock_by_seat=[dict(c) for c in self.clock],search=dict(self.search)))


def act_step(env,cfg,live,student,original,seat,bank,terrain,own,event):
    h=fixture.public_hashes(live,env,bank);t=env.states[0].observation.step
    opponent,_=action(original,live.observe(env,1-seat,bank[1-seat]),cfg,original=True)
    orders=[None,None];orders[seat]=own['order'];orders[1-seat]=opponent['order']
    after=bank.copy()
    for side,act in ((seat,own),(1-seat,opponent)):after[side]-=max(act['wall']-1.,0)
    env.step(orders)
    return after,dict(step=t,public_before_by_seat=h,orders=orders,bank_after=after.copy(),
                public_after_by_seat=fixture.public_hashes(live,env,after),board_after_sha256=play.board_hash(env,terrain),
                student=own,original=opponent,student_event=event)


def source_shas():
    return {str(p.relative_to(play.ROOT)):fixture.hfile(p) for p in
            [Path(__file__).resolve(),HERE/'protocol.json',MECHANISM/'fixture.py',
             Path(original_state.__file__),Path(official_state.__file__),Path(student_state.__file__),POLICY/'play.py']}


def run(spec,folder,plan):
    seed,seat,terrain=spec['seed'],spec['seat'],spec['board_sha256'];set_affinity()
    require(fixture.hfile(MECHANISM/'receipt.json')==plan['mechanism_sha256'],'Wrong qualification')
    require(play.read(play.INTAKE/'protocol.json')['physical_boards'][str(seed)]==terrain,'Wrong fixed training terrain')
    sources=source_shas()
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'claim.json').open('x') as flight:
        json.dump(dict(release=COMMIT,protocol_sha256=PLAN_SHA,seed=seed,seat=seat,terrain=terrain,
                       pid=os.getpid(),time=utc(),source_sha256=sources,arm_order=spec['arm_order']),flight)
    try:
        return play_pair(spec,folder,plan,sources)
    except BaseException as exc:
        play.save(folder/'failure.json',dict(time=utc(),reason=str(exc),trace=traceback.format_exc()))
        raise


def play_pair(spec,folder,plan,sources):
    seed,seat,terrain=spec['seed'],spec['seat'],spec['board_sha256'];t0=time.perf_counter()
    env,cfg,live=play.official(seed,terrain,seat)
    cards=play.read(POLICY/'protocol.json')['actors']
    actors={};cold={}
    for role in ('original','student'):
        path=play.ROOT/cards[role]['file']
        require(fixture.hfile(path)==plan[role+'_sha256'],'Wrong policy source')
        t=time.perf_counter();actors[role]=play.audit_menu.collect.Player(path);cold[role]=time.perf_counter()-t
    set_affinity()
    original,student=actors['original'],actors['student']
    m=original.mod
    require(m._SEARCH_OK and m._SEED is None and m._TEMP==1 and m._S_ABORT==.58 and m._torch.get_num_threads()==2,
            'Original search or native sampler altered')
    machine=resources();machine['torch_threads']=m._torch.get_num_threads()
    bank=[5.,5.];root=None;probes=[];prefix=Ledger(folder/'prefix.jsonl.gz')
    while not env.done:
        step=env.states[0].observation.step
        take=fixture.LivePosition(env,original,student,bank) if step in plan['probes'] else None
        own,event=action(student,live.observe(env,seat,bank[seat]),cfg)
        if take is not None:
            probes.append(dict(step=step,supported=event is not None))
            if event is not None:
                root=take;pre_menu=event;bank=root.restore();break
        bank,record=act_step(env,cfg,live,student,original,seat,bank,terrain,own,event)
        prefix.append(record,seat)
    info=dict(seed=seed,seat=seat,terrain=terrain,probes=probes,sources=sources,environment=machine,cold=cold,
              prefix_wall_seconds=time.perf_counter()-t0,pre_end_step=env.states[0].observation.step,**prefix.close())
    if root is None:
        info.update(status='no_registered_supported_root',root_step=None,terminal=env.done,
                    official_scores_by_seat=[q.reward for q in env.states] if env.done else None)
        play.save(folder/'prefix.json',info);play.save(folder/'finish.json',dict(status=info['status'],seed=seed,seat=seat))
        return
    step=env.states[0].observation.step
    identity=dict(root_step=step,root_fingerprints=root.fingerprint,root_public_by_seat=fixture.public_hashes(live,env,bank),
                  root_board_sha256=play.board_hash(env,terrain),root_economy=play.econ(env))
    info.update(status='root',root_orders=pre_menu['options'],root_menu=pre_menu,**identity)
    play.save(folder/'prefix.json',info)
    arms={}
    for idx in spec['arm_order']:
        label=('A','B')[idx];child=Ledger(folder/(label+'.jsonl.gz'))
        bank=root.restore();s0=time.perf_counter()
        require(fixture.public_hashes(live,env,bank)==identity['root_public_by_seat'],'Different root public input')
        first=None;first_commands=None
        while not env.done:
            offset=env.states[0].observation.step-step
            own,event=action(student,live.observe(env,seat,bank[seat]),cfg,choice=idx if offset==0 else None)
            if offset==0:
                require(event is not None and all(event[k]==pre_menu[k] for k in
                        ('options','p','unforced_choice','slot','actions','prefix','legal','native_logits','route','uid'))
                        and own['order']==event['options'][idx],'Wrong restored joint action')
                first=event
            bank,row=act_step(env,cfg,live,student,original,seat,bank,terrain,own,event)
            if not offset:first_commands=row['orders']
            child.append(row,seat)
        elapsed=time.perf_counter()-s0
        totals=child.close();scores=[q.reward for q in env.states]
        require(all(math.isfinite(x) for x in scores),'Non-numeric terminal reward')
        margin=scores[seat]-scores[1-seat];wdl=1. if margin>0 else 0. if margin<0 else .5
        result=dict(seed=seed,seat=seat,arm=label,index=idx,**identity,root_intervention=dict(event=first,orders=first_commands),
                    terminal_step=env.states[0].observation.step,sources=sources,pid=os.getpid(),
                    terminal_status=[q.status for q in env.states],terminal_economy=play.econ(env),official_scores_by_seat=scores,
                    margin=margin,wdl=wdl,u=.5*wdl+.25*(1+math.tanh(margin/2000.)),elapsed_seconds=elapsed,**totals)
        # Independent engine-only action tape oracle; compare both public histories, exact ID counters and official reward.
        bank=root.restore();replayed=0
        with gzip.open(folder/(label+'.jsonl.gz'),'rb') as f:
            for line in f:
                row=json.loads(line)
                require(fixture.public_hashes(live,env,bank)==row['public_before_by_seat'],'Suffix public input changed')
                env.step(row['orders']);bank=row['bank_after']
                require(play.board_hash(env,terrain)==row['board_after_sha256'] and
                        fixture.public_hashes(live,env,bank)==row['public_after_by_seat'],'Suffix physical outcome changed')
                replayed+=1
        require(env.done and [q.reward for q in env.states]==scores and replayed==totals['stats']['steps'],'Unreproducible terminal')
        result['replayed_transitions']=replayed;play.save(folder/(label+'.json'),result);arms[label]=result
    bank=root.restore()
    play.save(folder/'finish.json',dict(status='paired_raw',seed=seed,seat=seat,root_step=step,arm_order=spec['arm_order'],
        root_examiner_commands_equal=arms['A']['root_intervention']['orders'][1-seat]==arms['B']['root_intervention']['orders'][1-seat],
        second_minus_first_margin=arms['B']['margin']-arms['A']['margin'],second_minus_first_u=arms['B']['u']-arms['A']['u'],
        final_root_restored=root.signature(bank)==root.fingerprint,elapsed_seconds=time.perf_counter()-t0))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',required=True,type=int);args=parser.parse_args()
    require(all(os.environ.get(k)=='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')),'Set BLAS environment')
    require(fixture.hfile(HERE/'protocol.json')==PLAN_SHA,'Different registered protocol')
    plan=play.read(HERE/'protocol.json');cases=[v for v in plan['fixtures'] if v['seed']==args.seed]
    require(len(cases)==1,'Unregistered training case')
    spec=cases[0];folder=HERE/'outputs'/f"{spec['seed']}_{spec['seat']}"
    run(spec,folder,plan)
