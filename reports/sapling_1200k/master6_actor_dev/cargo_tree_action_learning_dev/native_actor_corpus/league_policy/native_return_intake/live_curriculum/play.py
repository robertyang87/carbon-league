"""Single-issue live training prefix and copy-on-write continuations against the untouched original."""
import argparse
from collections import Counter
from contextlib import contextmanager
import gc
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import traceback

HERE=Path(__file__).resolve().parent
INTAKE=HERE.parent
ROOT=next(p for p in HERE.parents if (p/'.git').exists())
sys.path.insert(0,str(INTAKE.parent/'natural_original'))
import audit_menu
import student_history
import numpy as np

PROTO='d4edce01c0686294f361d7fa6dfdcb2cb27af84a5b431df1cc82d9931b629141'
RULECOMMIT='9d24888a2bdf85587b5d9ceda398b8bb3319bd41'
read=lambda p: json.loads(p.read_bytes())
sha=lambda data: hashlib.sha256(data).hexdigest()
def need(ok,why):
    if not ok: raise ValueError(why)

def encode(obj):
    return json.dumps(obj,separators=(',',':'),sort_keys=True,allow_nan=False).encode()+b'\n'

def save(path,doc):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_bytes(encode(doc))
    os.replace(tmp,path)

def norm(obj):
    if isinstance(obj,np.ndarray): return ['array',str(obj.dtype),list(obj.shape),sha(obj.tobytes())]
    if isinstance(obj,np.generic):return obj.item()
    if isinstance(obj,dict):return ['dict',sorted([(str(k),norm(v)) for k,v in obj.items()])]
    if isinstance(obj,set):return ['set',sorted([norm(v) for v in obj],key=repr)]
    if isinstance(obj,(tuple,list)):return [norm(v) for v in obj]
    if isinstance(obj,type):return ['class',obj.__name__]
    if isinstance(obj,(bool,float,int,str)) or obj is None:return obj
    return ['object',type(obj).__name__,norm(vars(obj))]

def state_hash(obj):return sha(encode(norm(obj)))

def official(seed,terrain=None,seat=0):
    if terrain is None:terrain=read(HERE/'clearance.json')['by_seed'][str(seed)]
    rec=dict(seed=seed,physical_map_sha256=terrain,candidate_seat=seat,opponent_seat=1-seat,
             platform=read(INTAKE/'protocol.json')['platform'])
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)/'registration.json';save(path,rec)
        live=audit_menu.open_live(path)
    env,cfg=live.official()
    return env,cfg,live

def board_hash(env,terrain):
    from zerosum_env.envs.carbon import idgen
    o=env.states[0].observation
    return state_hash([terrain,o.step,o.players,o.trees,o.full_carbon,dict(idgen._id_dict),[[s.status,s.reward] for s in env.states]])

def econ(env):
    players=env.states[0].observation.players
    return dict(cash=[p[0] for p in players],cargo=[sum(v[1] for v in p[2].values()) for p in players],
                workers=[len(p[2]) for p in players],trees=[len(p[3]) for p in players])

class DrawTap:
    def __init__(self,generator):self.generator=generator;self.event=None
    def random(self):
        loc=sys._getframe(1).f_locals;u=self.generator.random()
        p=[float(v) for v in loc['probs']]
        opts=[dict(loc['base_cmd']),dict(loc['alt_cmd'])]
        need(self.event is None and len(p)==2 and 0<p[0]<1 and abs(sum(p)-1)<1.e-12 and opts[0]!=opts[1],'Invalid natural menu')
        pick=int(u>=p[0]);slot=int(loc['slot'])
        self.event=dict(p=p,options=opts,pick=pick,logp=math.log(p[pick]),slot=slot,
                        actions=[int(loc['best']),int(loc['second'])],prefix=loc['baseline'][:slot].tolist(),
                        legal=loc['trace'][slot][2].tolist(),native_logits=loc['logits'].tolist(),route=bool(loc['route']),
                        uid=str(loc['_FEAT'].own_uid[slot]) if '_FEAT' in loc else str(sys._getframe(1).f_globals['_FEAT'].own_uid[slot]))
        return u

def boundary(env,terrain,student,orig,live,bank):
    m=orig.mod;st=m._ST
    return dict(board_sha256=board_hash(env,terrain),step=env.states[0].observation.step,overage_by_seat=bank.copy(),
                public_sha256_by_seat=[sha(live.encoded(live.observe(env,p,bank[p]))) for p in (0,1)],
                tracker_sha256=state_hash(st['tr']),prev_players_sha256=state_hash(st['prev']),
                examiner_rng_sha256=state_hash(st['rng'].get_state()),examiner_cost=list(m._S_COST),
                examiner_search_stats_sha256=state_hash(m._S_STATS),student_history_sha256=student_history.snapshot(student),
                student_rng_sha256=student_history.behavior_rng_snapshot(student),economy=econ(env))

def call(player,obs,cfg,core,original=False):
    # The original explicitly creates two Torch threads. Move its existing
    # worker TIDs together; aggregate compute stays within the requested core.
    for thread in Path('/proc/self/task').iterdir():
        try: os.sched_setaffinity(int(thread.name),{core})
        except ProcessLookupError: pass
    before=Counter({k:int(player.mod._S_STATS.get(k,0)) for k in ('searched','single','aborted','changed')}) if original else None
    start=time.perf_counter();cpu=time.process_time()
    order=player.act(obs,cfg)
    out=dict(order=order,wall=time.perf_counter()-start,cpu=time.process_time()-cpu,core=core)
    if original:
        stats=player.mod._S_STATS
        changes={k:stats.get(k,0)-v for k,v in before.items()}
        need(changes['searched']+changes['single']==1,'Original public entry did not execute its registered search')
        candidate,depth,rolls,elapsed=1,0,0,None
        if changes['searched']:
            candidate,depth,rolls,elapsed,*_=stats['last']
        out['search']=dict(counts=changes,candidates=candidate,depth=depth,rolls=rolls,elapsed=elapsed,
                           measured_cost=list(player.mod._S_COST))
    return out

@contextmanager
def compressed(path):
    with path.open('xb') as f,gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as z:
        yield z

def turns(env,cfg,live,student,original,teacher,seat,bank,end,role,path):
    stats=Counter();tape=hashlib.sha256();clocks=[]
    with compressed(path) as stream:
        while not env.done and env.states[0].observation.step<end:
            t=env.states[0].observation.step
            obs=[live.observe(env,p,bank[p]) for p in (0,1)]
            h=[sha(live.encoded(x)) for x in obs]
            tap=None
            if role=='student' and t in audit_menu.STEPS:
                tap=DrawTap(student.mod._behavior_rng)
                student.mod._behavior_rng=tap
            active=call(student if role=='student' else teacher,obs[seat],cfg,0)
            if tap is not None:
                student.mod._behavior_rng=tap.generator
                if tap.event is not None:
                    need(active['order']==tap.event['options'][tap.event['pick']],'Natural action not sent')
                    stats['supported']+=1
                active['event']=tap.event
                stats['scheduled']+=1
            opponent=call(original,obs[1-seat],cfg,1,original=True)
            calls=dict(active=active,original=opponent)
            if role=='apex':
                calls['shadow']=call(student,obs[seat],cfg,0)
                stats['shadow_disagreements']+=int(active['order']!=calls['shadow']['order'])
            os.sched_setaffinity(0,{0})
            orders=[None,None];orders[seat]=active['order'];orders[1-seat]=opponent['order']
            for side,tm in ((seat,active['wall']),(1-seat,opponent['wall'])):
                bank[side]-=max(0,tm-1)
                stats['slow_'+str(side)]+=int(tm>1)
                stats['hard_'+str(side)]+=int(tm>3)
            env.step(orders);clocks.append([active['wall'],opponent['wall']])
            raw=encode(dict(step=t,public_sha256_by_seat=h,orders=orders,active_role=role,calls=calls,
                            overage_after=bank.copy()))
            stream.write(raw);tape.update(raw);stats['steps']+=1
    for i,name in enumerate(('active','original')):
        stats['max_wall_'+name]=max((pair[i] for pair in clocks),default=0.)
    return dict(stats=dict(stats),turns_sha256=sha(path.read_bytes()),raw_turns_sha256=tape.hexdigest())

def execute(seed,seat,folder,preflight=False,allow_fork=True):
    proto=read(HERE/'protocol.json')
    need(sha((HERE/'protocol.json').read_bytes())==PROTO,'Wrong preregistration')
    if preflight:
        need(seed==621758460,'Wrong historical mechanism fixture')
        terrain=read(INTAKE/'protocol.json')['physical_boards'][str(seed)]
    else:
        clearance=read(HERE/'clearance.json')
        need(sha((HERE/'clearance.json').read_bytes())=='7df29cd8b4323dcc49f5018645e679a9d19295e6ab2f571adffc65e7da64b840' and seed==proto['training_map']['seed'] and not clearance['seed_intersections'] and not clearance['physical_intersections'],'Uncleared source')
        terrain=clearance['by_seed'][str(seed)]
    folder.mkdir(parents=True,exist_ok=True)
    claim=folder/'claim.json'
    source_sha={role:x['sha256'] for role,x in proto['actors'].items()}
    with claim.open('x') as f:
        json.dump(dict(seed=seed,seat=seat,terrain=terrain,preflight=preflight,pid=os.getpid(),protocol_sha256=PROTO,
                       source_sha256=source_sha,runner_sha256=sha(Path(__file__).read_bytes()),utc=time.time()),f)
    os.sched_setaffinity(0,{0})
    env,cfg,live=official(seed,terrain,seat)
    paths={k:ROOT/v['file'] for k,v in proto['actors'].items()}
    need(all(sha(p.read_bytes())==source_sha[k] for k,p in paths.items()),'Policy bytes mismatch')
    need(sha(subprocess.check_output(['git','-C',str(ROOT),'show',RULECOMMIT+':sources/master3s2/master3s2.py']))==source_sha['original'],'Original source mismatch')
    actors={k:audit_menu.collect.Player(p) for k,p in paths.items()}
    original,student,teacher=[actors[k] for k in ('original','student','teacher')]
    need(original.mod._TEMP==1 and original.mod._SEED is None and original.mod._S_ABORT==.58 and original.mod._SEARCH_OK,
         'Wrong native examiner')
    bank=[5.,5.]
    cut=3 if preflight else 240
    end=cut+2 if preflight else 299
    prefix=turns(env,cfg,live,student,original,teacher,seat,bank,cut,'apex',folder/'prefix.jsonl.gz')
    prefix.update(seed=seed,seat=seat,terrain=terrain,source_sha256=source_sha,source_pid=os.getpid(),reached=env.states[0].observation.step)
    if env.done:
        prefix.update(status='early_terminal',official_scores_by_seat=[s.reward for s in env.states])
        save(folder/'prefix.json',prefix)
        return
    base=boundary(env,terrain,student,original,live,bank)
    prefix.update(status='root',boundary=base)
    save(folder/'prefix.json',prefix)
    def branch(label):
        before=boundary(env,terrain,student,original,live,bank)
        need(before==base,'Live root diverged')
        if label in ('draw_a','draw_b'):
            student.mod._behavior_rng=None
        start=time.perf_counter()
        row=turns(env,cfg,live,student,original,teacher,seat,bank,end,'student',folder/(label+'.jsonl.gz'))
        # This is the independent private student generator after the completed branch; no secret raw seed is recorded.
        token=student_history.behavior_rng_snapshot(student)
        row.update(seed=seed,seat=seat,arm=label,source_sha256=source_sha,boundary_before=before,
                   source_pid=prefix['source_pid'],pid=os.getpid(),parent_pid=os.getppid(),draw_token=token,
                   first_step=cut,terminal_step=env.states[0].observation.step,terminal=env.done,
                   official_scores_by_seat=[s.reward for s in env.states] if env.done else None,
                   terminal_economy=econ(env),terminal_status=[s.status for s in env.states],
                   overage_by_seat=bank.copy(),suffix_wall=time.perf_counter()-start)
        student.diagnostics()
        save(folder/(label+'.json'),row)
    def child(label):
        gc.collect()
        pid=os.fork()
        if pid==0:
            try:
                branch(label)
                os._exit(0)
            except BaseException:
                (folder/(label+'.error.txt')).write_text(traceback.format_exc())
                os._exit(1)
        deadline=time.monotonic()+(40 if preflight else 300)
        while True:
            done,status=os.waitpid(pid,os.WNOHANG)
            if done: return os.waitstatus_to_exitcode(status)
            if time.monotonic()>=deadline:
                os.kill(pid,signal.SIGKILL);os.waitpid(pid,0)
                return -9
            time.sleep(.1)
    outcome={}
    if allow_fork:
        for name in ('draw_a','draw_b'):
            outcome[name]=child(name)
            save(folder/'branches.json',outcome)
            if outcome[name]:
                if name=='draw_a' and not preflight:
                    branch('unforked_after_failure')
                return
        rows=[read(folder/(n+'.json')) for n in ('draw_a','draw_b')]
        need(rows[0]['draw_token']!=rows[1]['draw_token'] and all(r['boundary_before']==base for r in rows),'Duplicate entropy or different branch start')
    else:
        branch('unforked');outcome['unforked']=0;save(folder/'branches.json',outcome)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--seat',required=True,type=int,choices=(0,1))
    args=parser.parse_args()
    for var in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        need(os.environ.get(var)=='1','One native numerical thread required')
    if args.preflight:
        execute(621758460,args.seat,HERE/'mechanism',True)
    else:
        folder=HERE/'mechanism'
        okay=(folder/'branches.json').exists() and read(folder/'branches.json')=={'draw_a':0,'draw_b':0}
        execute(621758480,args.seat,HERE/'outputs'/f'621758480_{args.seat}',allow_fork=okay)
