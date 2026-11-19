"""Two frozen 16-turn policy interventions against naturally responding original Master3s2."""
import argparse
import fcntl
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

HERE = Path(__file__).resolve().parent
BALANCED=HERE.parent
CORPUS=BALANCED.parents[2]
CARGO=CORPUS.parent
ROOT=next(p for p in HERE.parents if (p/".git").exists())
OLD=CORPUS/"original_response"
KERNEL=CARGO/"learned_carbon_observation/native_response/run.py"
CFG=CARGO/"direct_original/games/candidate/raw_621758346_1.json.gz"
PARENT=CARGO/"candidate.py"
CHILD=BALANCED/"candidate.py"
ORIGINAL=ROOT/"data/master3s2_source.py"
TARGET=((621758400,0),(621758401,1))
ARMS=("parent","student")
BEGIN,END=200,216


def imp(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj,ensure_ascii=False,allow_nan=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def source(seed,seat):
    path=CORPUS/"fresh_original/hosted"/f"{seed}-{seat}.zip"
    with zipfile.ZipFile(path) as z:
        receipt=json.loads(z.read("parent/receipt.json"))
        reg=json.loads(z.read("parent/registration.json"))
        tape=z.read("parent/turns.jsonl")
        assert reg["files"]["candidate"]["sha256"]==sha(PARENT) and reg["files"]["opponent"]["sha256"]==sha(ORIGINAL)
        assert receipt["status"]=="complete" and receipt["seed"]==seed and receipt["candidate_seat"]==seat
        assert receipt["turns_sha256"]==hashlib.sha256(tape).hexdigest()
        rows=[json.loads(line) for line in tape.splitlines()]
        assert len(rows)==receipt["official_transitions"] and len(rows)>END
        actions=[[r["response"]["action"] for r in line["calls"]] for line in rows]
        assert all(row["step"]==i and len(row["calls"])==2 and all(r["status"]=="ok" for r in row["calls"]) for i,row in enumerate(rows))
    assert sha(CFG)=="7a5967383c78324d248f4b228541d7eb30f3ac3d39bf18e74f81a011a78c4631"
    conf=json.loads(gzip.decompress(CFG.read_bytes()))["game_configuration"]
    conf={k:v for k,v in dict(conf,agentTimeout=20,actTimeout=1,runTimeout=1500).items() if k!='randomSeed'}
    raw=dict(seed=seed,candidate_seat=seat,initial_terrain_u8_sha256=reg["physical_map_sha256"],actions=actions,game_configuration=conf)
    return raw, dict(seed=seed,seat=seat,terrain=reg["physical_map_sha256"],archive=str(path.relative_to(ROOT)),
                     archive_sha256=sha(path),turns_sha256=receipt["turns_sha256"],historical_prefix_sha256=digest(actions[:BEGIN]))


def register():
    cov=imp("balanced_map_sets",BALANCED/"historical_coverage.py")
    native,league=cov.frozen_training()
    cases=[]
    for seed,seat in TARGET:
        raw,card=source(seed,seat)
        for cohort in (native,league):
            assert seed not in cohort and card["terrain"] not in cohort.values()
        cases.append(dict(card,cpu=seat,arm_order=list(ARMS if seat==0 else reversed(ARMS))))
    assert len({r["terrain"] for r in cases})==2
    authoritative=subprocess.check_output(["git","-C",str(ROOT),"show","HEAD:sources/master3s2/master3s2.py"])
    assert hashlib.sha256(authoritative).hexdigest()==sha(ORIGINAL)
    legacy=json.loads((OLD/"protocol.json").read_bytes())
    files={ROOT/p for p in legacy["files"]};files.update([Path(__file__),HERE/"settle.py",OLD/"run.py",CFG,ORIGINAL,PARENT,CHILD,
        BALANCED/"fit.json",BALANCED/"protocol.json",BALANCED/"historical_coverage.py",CARGO/"candidate.npz",KERNEL,
        CARGO/"native_actor_corpus/league_policy/natural_original/economy.py"])
    fit=json.loads((BALANCED/"fit.json").read_bytes())
    assert sha(CHILD)==fit["candidate_sha256"]==cov.CHILD and sha(PARENT)==cov.PARENT and sha(ORIGINAL)==legacy["original_sha256"]
    return dict(schema=1,parent=sha(PARENT),student=sha(CHILD),original=sha(ORIGINAL),t0=BEGIN,t1_exclusive=END,
                cases=cases,files={str(f.relative_to(ROOT)):sha(f) for f in sorted(files)},
                training_physical_maps=dict(original=native,league=league),
                operations="Reconstruct actual historic physical action prefix and retarget the original belief tracker on each archived turn. From t200 generate one free natural original response while own interventions agree, including the first differing own action's opponent command. Fork full original tracker, adaptive costs, Python/NumPy/Torch RNG at that point. Student intervenes only until t215 inclusive, then 8da4 resumes to official terminal in both arms. If no treatment is issued, both policies are identical thereafter and use the same actual continuing trajectory. Persist all branches, errors, no-ops and negative effects. Opponent package, natural sampling and wall-clock adaptation unchanged.",
                hardware="Same host, logical CPU0 for 400/0 and CPU1 for 401/1 with affinities checked. Arms serial within each case, reverse order between cases. Capture every native public call including original candidate width/depth/abort, process CPU and wall. Locally pinned host, not a certified ladder executor.",
                criterion="Positive local evidence only if the policy actually issues different physical commands on EACH map inside [200,216), and each complete branch's official learner margin strictly improves over its paired 8da4 branch. No replacement, stopping on wins, promotion or full-game win-rate claim.")


def verify_release(commit):
    assert __debug__
    p=HERE/"protocol.json"
    b=subprocess.check_output(["git","-C",str(ROOT),"show",commit+":"+str(p.relative_to(ROOT))])
    if b!=p.read_bytes() or json.loads(b)!=json.loads(json.dumps(register())):
        raise ValueError("frozen registration or inputs differ")
    return json.loads(b)


def env_for(k,raw):
    # Board generation seeds NumPy inside the historic helper. Isolate it from
    # the unseeded examiner's NumPy RNG on both initial start and branch restore.
    state=k.np.random.get_state()
    try:return k.checked_env(raw)
    finally:k.np.random.set_state(state)


def run(which,commit):
    plan=verify_release(commit)
    case=plan["cases"][which];seat=case["seat"];cpu=case["cpu"]
    os.sched_setaffinity(0,{cpu})
    legacy=imp("frozen_window_tools",OLD/"run.py");k=legacy.kernel()
    assert k.R.sha(k.ORIGINAL)==plan["original"]
    folder=HERE/f"{case['seed']}_{seat}";folder.mkdir(exist_ok=True)
    lock=(folder/".execution.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    raw,_=source(case["seed"],seat);packs=dict(parent=PARENT,student=CHILD)
    cfg=raw["game_configuration"]
    who,host=k.load_master(cpu)
    identity=dict(protocol=sha(HERE/"protocol.json"),commit=commit,cpu=cpu,seed=case["seed"],seat=seat,
                  parent=plan["parent"],student=plan["student"],original=plan["original"])
    commonfile=folder/"shared.json.gz"
    def save(name,d):legacy.save(k,folder/name,d)
    if not commonfile.exists():
        env=env_for(k,raw); models=legacy.models(k,packs,"common")
        rec=k.R.native_clock.Recorder(who); enemy=rec.wrap(who.agent)
        prefix=[]; trace=[]; fork=None
        started=time.perf_counter()
        while not env.done:
            t=env.states[0].observation.step
            o,e=k.public(env,seat),k.public(env,1-seat)
            opposing=k.R.live.clean(enemy(e,cfg))
            if t<END:
                own,wall=legacy.commands(k,models,o,cfg)
            else:
                action,elapsed=k.own_call(models["parent"],o,cfg);own=dict(parent=action,student=action);wall={"parent":elapsed}
            trace.append(dict(t=t,public=[digest(o),digest(e)],proposed_enemy=opposing,orders=own,own_public_seconds=wall))
            if t<BEGIN:
                j=raw["actions"][t]
                assert own["parent"]==j[seat],("historical parent drift",t)
                who._ST["tr"].set_actions(e["players"],j[1-seat])
            elif t<END and own["parent"]!=own["student"]:
                fork=dict(t=t,own=own,enemy=opposing,public=trace[-1]["public"])
                break
            else:
                j=k.physical(own["parent"],opposing,seat)
            env.step(j);prefix.append(j)
        common=dict(identity=identity,host=host,clock=k.clock_result(rec,who),prefix=prefix,fork=fork,trace=trace,
                    prefix_sha=digest(prefix),wall_s=time.perf_counter()-started)
        if fork is None:common.update(terminal=k.score(env))
        else:
            snap=k.capture(who);common.update(snapshot_sha=k.digest(snap),state_sha=k.sig(who))
            k.blob_write(folder/"original_root.pickle.gz",gzip.compress(snap,mtime=0))
        k.R.native_clock.validate_host(k.R.native_clock.host_snapshot(who),[cpu])
        save("shared.json.gz",common)
    common=k.R.load(commonfile);assert common["identity"]==identity and digest(common["prefix"])==common["prefix_sha"]
    fork=common["fork"];prefix=common["prefix"]
    for arm in case["arm_order"]:
        branchfile=folder/(arm+".json.gz")
        if branchfile.exists():
            receipt=k.R.load(branchfile);assert receipt["identity"]==identity and receipt["shared_sha"]==sha(commonfile)
            continue
        if fork is None:
            score,cash,status=common["terminal"]
            tail=[];checks=[];clock={"calls":[],"summary":None};host_end=host;elapsed=0.
        else:
            env=env_for(k,raw);models=legacy.models(k,packs,arm)
            for t,actual in enumerate(prefix):
                own,_=legacy.commands(k,models,k.public(env,seat),cfg)
                assert own==common["trace"][t]["orders"]
                env.step(actual)
            snap=gzip.decompress((folder/"original_root.pickle.gz").read_bytes())
            assert k.digest(snap)==common["snapshot_sha"]
            k.restore(who,snap);assert k.sig(who)==common["state_sha"]
            rec=k.R.native_clock.Recorder(who);enemy=rec.wrap(who.agent)
            tail=[];checks=[];started=time.perf_counter()
            while not env.done:
                t=env.states[0].observation.step
                o,e=k.public(env,seat),k.public(env,1-seat)
                opp=fork["enemy"] if t==fork["t"] else k.R.live.clean(enemy(e,cfg))
                if t<END:
                    own,times=legacy.commands(k,models,o,cfg)
                    if t==fork["t"]: assert own==fork["own"] and [digest(o),digest(e)]==fork["public"]
                    chosen=own[arm]
                else:
                    chosen,w=k.own_call(models["parent"],o,cfg);own={"parent":chosen};times={"parent":w}
                joint=k.physical(chosen,opp,seat)
                checks.append(dict(t=t,public=[digest(o),digest(e)],orders=own,own_public_seconds=times))
                tail.append(joint);env.step(joint)
            score,cash,status=k.score(env);clock=k.clock_result(rec,who)
            elapsed=time.perf_counter()-started;host_end=k.R.native_clock.host_snapshot(who)
        k.R.native_clock.validate_host(host_end,[cpu])
        save(arm+".json.gz",dict(identity=identity,arm=arm,shared_sha=sha(commonfile),tail=tail,tail_sha=digest(tail),
             clock=clock,host_end=host_end,checks=checks,official_scores=score,physical_cash=cash,official_status=status,
             margin=score[seat]-score[1-seat],wall_s=elapsed,transitions=len(prefix)+len(tail)))
        print(case["seed"],arm,score,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest="mode",required=True)
    sub.add_parser("register")
    q=sub.add_parser("run");q.add_argument("--release",required=True);q.add_argument("--case",type=int,choices=(0,1),required=True)
    args=p.parse_args()
    if args.mode=="register":
        content=json.dumps(register(),sort_keys=True,indent=2,allow_nan=False)+"\n";dest=HERE/"protocol.json"
        if dest.exists() and dest.read_text()!=content:raise ValueError("previous registration differs")
        dest.write_text(content);print(sha(dest))
    else:run(args.case,args.release)
