"""Apex whole-team response on two already exposed, archived 8da4 games. Dry by default."""
import argparse
import base64
from collections import Counter
import copy
import fcntl
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[key]="1"
os.environ["CARBON_DEBUG"]="1"
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/"code/packs").is_dir() and (p/"reports/sapling_1200k").is_dir())
EXP=ROOT/"reports/sapling_1200k/master6_actor_dev/cargo_tree_action_learning_dev"
RUN=EXP/"native_actor_corpus/original_response/run.py"
PARENT=EXP/"candidate.py"
APEX=ROOT/"code/apexcarbon_v4tta_search.py"
AH="f2882881e71b036421c9236b4fa0f1deef1919ed8045822191ed1632323bd54a"
PH="8da4b465f7fa6e601e51bd7c2e81f25825cb59b9b5ab285f7f56d0f6bbfe1cac"
MH="65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"
START,END=200,208
TRAIN=EXP/"native_actor_corpus/league_policy/clearance.json"
CASES=[
 (621758345,1,"e9301a96ee173bebbcc33bfe30f6257d60e7da5a86a379528bafbad26ba91bb4","215fe59f5b0f090a6e18ccb149bafb490f37ff5ac1be0a9b9675acc825b00b8a"),
 (621758346,0,"4e2446209a74ae3cb2107157ce29a5c4061b2e130197599ccd2f7dcc2c2997a0","1ca974401229fc1d02dfe45d8430333b32cbc8796564a0d2bef9231bdb94b2e4"),
]
def insist(check,msg):
    if not check: raise ValueError(msg)
def digest(b):return hashlib.sha256(b).hexdigest()
def sha(p):return digest(p.read_bytes())
def read(p):return json.loads(gzip.decompress(p.read_bytes()) if p.suffix==".gz" else p.read_bytes())
def encode(v):return json.dumps(v,sort_keys=True,allow_nan=False,separators=(",",":")).encode()
def save(p,v):
    data=encode(v)
    if p.suffix==".gz":data=gzip.compress(data,mtime=0)
    tmp=p.with_suffix(".incomplete")
    with tmp.open("wb") as f:f.write(data);f.flush();os.fsync(f.fileno())
    tmp.replace(p)
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path); obj=importlib.util.module_from_spec(spec)
    sys.modules[name]=obj; spec.loader.exec_module(obj);return obj
def binding():
    insist(sha(APEX)==AH and sha(PARENT)==PH and sha(ROOT/"data/master3s2_source.py")==MH,"package identity")
    insist(digest(subprocess.check_output(["git","-C",str(ROOT),"show","HEAD:sources/master3s2/master3s2.py"]))==MH,"canonical examiner identity")
    used=read(TRAIN)["by_seed"]
    banks=[]
    for seed,seat,h,rh in CASES:
        base=EXP/"direct_original/games/candidate"; p=base/f"raw_{seed}_{seat}.json.gz"; q=base/f"receipt_{seed}_{seat}.json"
        insist(sha(p)==h and sha(q)==rh,"original game receipts")
        g=read(p); r=read(q)
        insist(g["pack_sha256"]==r["binding"]["candidate"]==PH and r["binding"]["original"]==MH and not g["errors"] and r["game"]["valid"],"archive identity")
        insist(r["game"]["raw_sha256"]==h and g["official_score_by_seat"]==r["official"]["official_reward_by_seat"] and g["steps"]>=END,"settlement")
        terrain=g["initial_terrain_u8_sha256"];insist(str(seed) not in used and terrain not in used.values(),"new league terrain overlap")
        banks.append(dict(seed=seed,seat=seat,path=str(p.relative_to(ROOT)),sha256=h,receipt=str(q.relative_to(ROOT)),receipt_sha256=rh,terrain=terrain))
    return dict(schema="whole-team-apex-original-t200/1",source_main="e74745c9d6398cd9c8de95aab5f40fade844c807",inputs=banks,parent=PH,apex=AH,examiner=MH,
        runner_sha256=sha(Path(__file__)),shell_concurrency="case 0 uses CPU 0 and case 1 uses CPU 1; for each map the two natural-random streams and both arms run serially, native single-thread affinity verified",
        evidence="Preserve full physical joint tape, seat-scoped orders & pre-state observation digests, both original and teacher timings, original root snapshots, actual per-side deposits/tree proceeds and cash, cargo and live assets per transition. Same-turn assessments receive separately copied simultaneous observations.",
        archival_release="Students can use whole-window demonstrations only after this diagnostic passes and a separately held-out training design is registered. These retired historical maps stay out of all 18+6+2+100 reserved sets.",
        native_runner_sha256=sha(RUN),plant_window_reference_sha256=sha(EXP/"planter_teacher_window/run.py"),official_kernel_sha256=sha(EXP/"learned_carbon_observation/native_response/run.py"),cross_check=sha(TRAIN),
        scope="Historical conditional diagnostic only. Both maps are retired from new development, training and formal assessment. No student win or dependence on the private examiner labels.",
        selection="Both direct_original maps that first evaluated 8da4, ascending seed, historical physical seat seed mod 2, chosen chronologically; no game-outcome filtering. Two independently sampled full-native enemy streams per map, same pre-fork enemy state, observations, root order and measured costs across arms within each pair. Same deterministic Apex proposal seed across streams of a map; wall time remains adaptive.",
        intervention="Entire legal public Apex command at steps 200..207 inclusive, all workers and bases, unchanged 8da4 otherwise. Parent and teacher share the whole prefix through the first different joint action. If 8 orders are identical, freeze a zero-treatment split on step 207 and collect both arms. Retain wall-clock divergence, all outcomes and early official termination.",
        supports="All eight complete official tails are required; preserve each failure. At least one actual non-planter difference per map; mean treatment-minus-parent official margin strictly positive on each terrain and positive in at least three of four paired streams. Descriptive conditional support only. Missing, errored or infeasible arms cannot pass. No selection by modified units, depth, cash ledger, outcome or map. Log both root-to-terminal normal deposits and tree income before attributing any effect.")
def public_pair(k,env,seat):
    views=[copy.deepcopy(k.public(env,side)) for side in (seat,1-seat)]
    keys={"player","step","players","trees","carbon","remainingOverageTime"}
    insist(all(set(obs)==keys and obs["player"]==side for side,obs in zip((seat,1-seat),views))
           and views[0]["step"]==views[1]["step"],"private observation or non-simultaneous seats")
    return views
def teacher_history(m,obs,executed):
    tr=m._ST["tr"];tr.step(obs["players"],obs["step"],obs["carbon"])
    tr.set_actions(obs["players"],executed);m._ST["prev"]=copy.deepcopy(obs["players"])
def teacher_action(m,obs,cfg,k):
    started=time.perf_counter(); cpu=time.process_time()
    a=k.R.live.clean(m.agent(copy.deepcopy(obs),dict(cfg)))
    bases,workers=obs["players"][obs["player"]][1:3]
    insist(set(a)<=set(bases)|set(workers),"illegal unit identity")
    insist(all((cmd in ("RECCOLLECTOR","RECPLANTER") if uid in bases else cmd in ("UP","RIGHT","DOWN","LEFT")) for uid,cmd in a.items()),"illegal command")
    return a,dict(wall=time.perf_counter()-started,cpu=time.process_time()-cpu,adaptive_cost=list(m._S_COST))
def shared(k,master,teacher,doc,raw,rec,seat,stream,progress):
    cfg={k:v for k,v in raw["game_configuration"].items() if k!="randomSeed"}; env=k.checked_env(raw); own=k.R.live.module(PARENT,"apex_window_origin")
    orig=rec.wrap(master.agent); trace=[]; hist=[]; initial_rng=None
    teacher._ST.clear();teacher._ST.update(tr=teacher.BeliefTracker(seat),prev=None,rng=np.random.RandomState(int(digest(f"apex-full-teacher-v1|{raw['seed']}|{seat}".encode())[:8],16)))
    for t in range(END):
        my,en=public_pair(k,env,seat)
        insist(my["step"]==t and en["step"]==t and not env.done,"invalid pre-fork state")
        world=k.R.live.clean(orig(copy.deepcopy(en),dict(cfg))); base,_=k.own_call(own,copy.deepcopy(my),dict(cfg))
        if t==0:initial_rng=digest(pickle.dumps(master._ST["rng"].get_state(),protocol=5))
        if t<START:
            committed=raw["actions"][t]
            insist(base==k.R.live.clean(committed[seat]),"8da4 behavior does not replay; cohort is not valid")
            master._ST["tr"].set_actions(en["players"],committed[1-seat])
            teacher_history(teacher,my,committed[seat]); hist.append(committed);env.step(committed)
            trace.append(dict(t=t,enemy_sampled=world,enemy_executed=committed[1-seat],own_public=digest(encode(my)),enemy_public=digest(encode(en))))
            if t%10==0:save(progress,dict(identity=doc,phase="prefix",through=t,trace=trace,clock=rec.calls))
            continue
        alt,wall=teacher_action(teacher,my,cfg,k)
        order=dict(t=t,parent=base,teacher=alt,enemy=world,teacher_clock=wall,own_public=digest(encode(my)),enemy_public=digest(encode(en)),
            roles={u:('base' if u in my['players'][seat][1] else my['players'][seat][2][u][2]) for u in set(alt)|set(base) if base.get(u)!=alt.get(u)})
        trace.append(order)
        if alt!=base or t==END-1:break
        both=k.physical(base,world,seat);hist.append(both);env.step(both)
        teacher._ST['tr'].set_actions(my['players'],base)
    snap=k.capture(master); teacher_snap=pickle.dumps((teacher._ST,teacher._S_COST),protocol=5)
    return dict(stream=stream,native_first_search_rng_state_sha=initial_rng,cpu=doc['cpu'],identity=doc,root_t=t,prefix=hist,trace=trace,clock=k.clock_result(rec,master),
                root=order,enemy_state=k.sig(master),enemy_snapshot=base64.b64encode(snap).decode(),enemy_snapshot_sha256=digest(snap),
                teacher_snapshot=base64.b64encode(teacher_snap).decode(),teacher_snapshot_sha256=digest(teacher_snap))
def tail(k,master,teacher,shared,raw,seat,arm,progress):
    cfg={k:v for k,v in raw["game_configuration"].items() if k!="randomSeed"};env=k.checked_env(raw);parent=k.R.live.module(PARENT,f"apex_suffix_parent_{arm}")
    t0=shared['root_t'];stamp=shared['root']
    for t,joint in enumerate(shared['prefix']):
        pub=k.public(env,seat);got,_=k.own_call(parent,pub,cfg)
        insist(digest(encode(pub))==shared['trace'][t]['own_public'] and got==joint[seat],"parent on common prefix")
        env.step(joint)
    snap=base64.b64decode(shared['enemy_snapshot'],validate=True);pool=base64.b64decode(shared['teacher_snapshot'],validate=True)
    insist(digest(snap)==shared['enemy_snapshot_sha256'] and digest(pool)==shared['teacher_snapshot_sha256'],"snapshot bytes")
    k.restore(master,snap);insist(k.sig(master)==shared['enemy_state'],"common native state")
    teacher._ST,teacher._S_COST=pickle.loads(pool)
    recorder=k.R.native_clock.Recorder(master);orig=recorder.wrap(master.agent)
    tape=[];steps=[];window=[]
    while not env.done:
        my,en=public_pair(k,env,seat);t=my['step'];insist(en['step']==t,"same turn")
        base,base_sec=k.own_call(parent,my,cfg)
        alt=None;teacher_time=None
        if t==t0:
            insist(base==stamp['parent'] and digest(encode(my))==stamp['own_public'] and digest(encode(en))==stamp['enemy_public'],"root identity")
            alt,world,teacher_time=stamp["teacher"],stamp["enemy"],stamp["teacher_clock"]
        else:
            world=k.R.live.clean(orig(copy.deepcopy(en),dict(cfg)))
            if t<END and arm=='teacher':alt,teacher_time=teacher_action(teacher,my,cfg,k)
        choice=alt if arm=='teacher' and t<END else base
        insist(choice is not None,"missing full order")
        if arm=='teacher' and t<END:teacher._ST["tr"].set_actions(my['players'],choice)
        if t<END:
            roles={u:('base' if u in my['players'][seat][1] else my['players'][seat][2][u][2]) for u in set(choice)|set(base) if base.get(u)!=choice.get(u)}
            window.append(dict(t=t,parent=base,teacher=alt,actual=choice,roles=roles,teacher_clock=teacher_time))
        joint=k.physical(choice,world,seat);tape.append(joint);steps.append(dict(t=t,own=digest(encode(my)),enemy=digest(encode(en)),parent_seconds=base_sec))
        env.step(joint)
        if len(tape)%10==0 or env.done:
            save(progress,dict(identity=shared["identity"],arm=arm,tape=tape,window=window,steps=steps,clock=recorder.calls))
    sc,cash,status=k.score(env);clocks=k.clock_result(recorder,master)
    insist(len(tape)==len(clocks['calls'])+1 and [x['step'] for x in clocks['calls']]==list(range(t0+1,t0+len(tape))),"partial native timing")
    # Independent official replay of the actual root-to-terminal physical orders, incl. extinction.
    independent=k.checked_env(raw)
    for pair in shared['prefix']+tape:independent.step(pair)
    insist(k.score(independent)==(sc,cash,status),"official settlement mismatch")
    sim=k.R.FastCarbon(cfg).reset(seed=raw["seed"])
    insist(list(sim.carbon)==raw["initial_full_carbon"],"economic map differs")
    for joint in raw["actions"][:START]:sim.step(*joint)
    sim.cash=k.R.CashEvents(sim.cash)
    credits={name:[Counter(),Counter()] for name in ("window","root_to_terminal")}
    def assets():
        out=[]
        public=sim.obs(0)["players"]
        for side in public:
            cash,bases,workers,trees=side
            counts=Counter(v[2] for v in workers.values())
            out.append(dict(cash=cash,collectors=counts["COLLECTOR"],planters=counts["PLANTER"],base_count=len(bases),
                            trees=len(trees),carried=sum(v[1] for v in workers.values())))
        return out
    initial=list(sim.cash)
    economic_path=[dict(before_step=START,assets=assets())]
    for i,joint in enumerate(shared["prefix"][START:]+tape):
        sim.cash.events.clear();sim.step(*joint)
        economic_path.append(dict(after_step=START+i,assets=assets(),cash_events=list(sim.cash.events)))
        for key,side,cost,*meta in sim.cash.events:
            values={"ground_rent":cost-meta[2],"cargo_tax":meta[2]} if key=="tree_absorption" else {key:cost}
            credits["root_to_terminal"][side].update(values)
            if START+i<END:credits["window"][side].update(values)
    insist(np.allclose(cash,sim.cash,atol=.011,rtol=0) and np.allclose(np.array(cash)-initial,
           [sum(a.values()) for a in credits["root_to_terminal"]],atol=.011,rtol=0),"cash ledger does not close")
    return dict(arm=arm,root_t=t0,cash_components=credits,assets_and_realized_cash=economic_path,root_state=shared['enemy_state'],shared_sha=digest(encode(shared)),
                tape=tape,tape_sha=digest(encode(tape)),window=window,steps=steps,clock=clocks,scores=sc,cash=cash,status=status,margin=sc[seat]-sc[1-seat])
def run(k,plan,out,case,stream,cpu):
    s=case["seat"]; seed=case["seed"];raw=read(ROOT/case['path'])
    ident=dict(plan=digest(encode(plan)),cpu=cpu,source=case['sha256'],stream=stream)
    folder=out/f"{seed}_{s}_stream{stream}";folder.mkdir(parents=True,exist_ok=True)
    with (folder/'process.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        os.sched_setaffinity(0,{cpu});m,host=k.load_master(cpu); t=module(APEX,'sapling_whole_apex');
        insist(sha(k.ORIGINAL)==MH and getattr(t,"_S_ABORT")==.62,"wrong native or Apex config")
        bank=folder/'shared.json.gz'
        def do_phase(path,fn):
            mark=path.with_suffix(".inflight")
            if path.exists(): return read(path)
            if mark.exists():raise RuntimeError(f"Unresolved previous draw {mark}; retain failure, do not resample")
            mark.write_text(encode(ident).decode());result=fn();save(path,result);mark.unlink();return result
        shared_state=do_phase(bank,lambda:shared(k,m,t,ident,raw,k.R.native_clock.Recorder(m),s,stream,folder/"prefix_progress.json.gz"))
        insist(shared_state['identity']==ident,"wrong archived prefix")
        order=('parent','teacher') if (stream+seed)%2==0 else ('teacher','parent')
        results={}
        for arm in order:
            result=do_phase(folder/(arm+'.json.gz'),lambda:tail(k,m,t,shared_state,raw,s,arm,folder/(arm+"_progress.json.gz")))
            insist(result['shared_sha']==digest(encode(shared_state)) and result['arm']==arm,"branch mismatch")
            results[arm]=result
            (folder/(arm+"_progress.json.gz")).unlink(missing_ok=True)
        (folder/"prefix_progress.json.gz").unlink(missing_ok=True)
        before,after=results['parent'],results['teacher']
        end=k.R.native_clock.host_snapshot(m);k.R.native_clock.validate_host(end,[cpu])
        save(folder/'pair.json',dict(identity=ident,initial_host=host,final_host=end,delta_margin=round(after['margin']-before['margin'],2),
            actual_nonplanter_changed=sum(v!='PLANTER' for row in after['window'] for v in row['roles'].values()),
            transaction="Same source prefix, full native pre-fork RNG+clock state restored, common original order at first actual divergence, common subsequent normal 8da4 continuation after t207.",
            parent_file_sha=sha(folder/'parent.json.gz'),teacher_file_sha=sha(folder/'teacher.json.gz')))
def summarize(plan,out):
    results=[]
    for case in plan["inputs"]:
        margins=[];other_roles=0;streams=set()
        for flow in (0,1):
            seat=case["seat"]
            folder=out/f"{case['seed']}_{seat}_stream{flow}"
            pair=read(folder/"pair.json");bank=read(folder/"shared.json.gz")
            insist(pair["identity"]==bank["identity"] and pair["identity"]["plan"]==digest(encode(plan)),"mixed protocol")
            streams.add(bank["native_first_search_rng_state_sha"])
            arms=[read(folder/(a+".json.gz")) for a in ("parent","teacher")]
            for arm,r in zip(("parent","teacher"),arms):
                insist(pair[arm+"_file_sha"]==sha(folder/(arm+".json.gz")) and r["shared_sha"]==digest(encode(bank)),"mixed draw or result")
            cash={}
            for window in ("window","root_to_terminal"):
                cash[window]={}
                for field in ("own_delivery","ground_rent","cargo_tax","plant","seize"):
                    x=[q["cash_components"][window] for q in arms]
                    cash[window][field]=round((x[1][seat].get(field,0)-x[1][1-seat].get(field,0))-(x[0][seat].get(field,0)-x[0][1-seat].get(field,0)),4)
            margins.append(pair["delta_margin"]);other_roles+=pair["actual_nonplanter_changed"]
            results.append(dict(seed=case["seed"],seat=seat,stream=flow,parent_score_physical=arms[0]["scores"],
                                teacher_score_physical=arms[1]["scores"],paired_margin=pair["delta_margin"],
                                actual_nonplanter_changed=pair["actual_nonplanter_changed"],paired_cash_margins=cash))
        insist(len(streams)==2,"independent original stream requirement")
    passed=sum(z["paired_margin"]>0 for z in results)>=3 and all(sum(z["paired_margin"] for z in results if z["seed"]==c["seed"])>0 and
          any(z["actual_nonplanter_changed"] for z in results if z["seed"]==c["seed"]) for c in plan["inputs"])
    save(out/"summary.json",dict(protocol=digest(encode(plan)),pairs=results,tails=2*len(results),supports_conditional_demonstration_only=passed))
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summarize',action='store_true');parser.add_argument('--run',action='store_true');parser.add_argument('--cpu',type=int);parser.add_argument('--case',type=int,choices=(0,1));parser.add_argument('--stream',type=int,choices=(0,1));parser.add_argument('--out',type=Path)
    args=parser.parse_args(); plan=binding()
    if args.run or args.summarize:
        insist(plan==read(HERE/"protocol.json"),"frozen preregistration not equal to executable and source bindings")
    if args.run:
        insist(args.case in (0,1) and args.cpu==args.case,"wrong assigned native CPU")
        os.sched_setaffinity(0,{args.cpu})
    if args.summarize:
        insist(args.out is not None,"--out required");summarize(plan,args.out);return
    if not args.run: print(json.dumps(plan,indent=2));return
    insist(None not in (args.cpu,args.case,args.stream,args.out),"run requires signed cpu, case, stream and out")
    k=module(RUN,"registered_window").kernel(); run(k,plan,args.out,plan['inputs'][args.case],args.stream,args.cpu)
if __name__=='__main__':main()
