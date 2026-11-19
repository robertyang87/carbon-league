"""One natural examiner draw per registered root, paired through the official terminal."""
import argparse
import fcntl
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
CARBON=HERE.parents[1]
EXP=CARBON.parent
spec=importlib.util.spec_from_file_location("decoder_native_kernel",CARBON/"native_response/run.py")
K=importlib.util.module_from_spec(spec)
spec.loader.exec_module(K)
R=K.R
PLAN_SHA="28596d4f162dddeed5a5d5d3a5d47384dcbce74e5f3380a1faad018e841c960c"


def check(raw,case,s,t,own_hash,enemy_hash):
    assert raw["seed"]==case["seed"] and s==case["seat"] and t==case["t"]
    assert own_hash and enemy_hash


def setup(idx,cpu):
    os.sched_setaffinity(0,{cpu})
    plan=R.load(HERE/"native_protocol.json"); assert R.sha(HERE/"native_protocol.json")==PLAN_SHA
    committed=subprocess.check_output(["git","-C",str(R.ROOT),"show","a494ec6ea:"+str((HERE/"native_protocol.json").relative_to(R.ROOT))])
    assert K.digest(committed)==PLAN_SHA
    assert R.sha(HERE/"roots.json")==plan["roots_sha256"]
    case=R.load(HERE/"roots.json")["cases"][idx]
    rc=plan["cases"][idx]
    assert rc["id"]==case["id"] and rc["root_identity"]==case["identity_sha256"]
    packs=dict(parent=CARBON/"candidate.py",student=CARBON/"decoder/candidate.py",continuation=EXP/"candidate.py")
    assert R.sha(packs["parent"])==plan["root_policies"]["parent_encoder_only"]
    assert R.sha(packs["student"])==plan["root_policies"]["alternative_encoder_decoder"]
    assert R.sha(packs["continuation"])==plan["continuation_sha256"]
    assert R.sha(K.ORIGINAL)==plan["examiner"]["sha256"] and hashlib.md5(K.ORIGINAL.read_bytes()).hexdigest()==plan["examiner"]["md5"]
    assert R.sha(R.ROOT/case["source"])==case["source_sha256"]
    assert R.sha(HERE/case["prefix_file"])==case["prefix_file_sha256"]
    raw=R.load(R.ROOT/case["source"]); witness=R.load(HERE/case["prefix_file"])
    pref=witness["physical_fixed_history"]
    assert pref==raw["actions"][:case["t"]] and K.digest(K.compact(pref))==case["prefix_sha256"]
    directory=HERE/"native"/case["id"];directory.mkdir(parents=True,exist_ok=True)
    identity=dict(plan=PLAN_SHA,roots=plan["roots_sha256"],case=case["identity_sha256"],runner=R.sha(Path(__file__)),
        kernel=R.sha(Path(K.__file__)),packs={k:R.sha(p) for k,p in packs.items()},original=R.sha(K.ORIGINAL),source=case["source_sha256"],cpu=cpu)
    return plan,case,rc,packs,raw,witness,directory,identity


def shared(case,packs,raw,witness,folder,identity,m,host):
    out=folder/"shared.json.gz";snap=folder/"original_root.pickle.gz"
    if out.exists():
        d=R.load(out)
        assert d["identity"]==identity and K.digest(gzip.decompress(snap.read_bytes()))==d["snapshot_sha"]
        return d
    # If the host stops during warmup, recover the actual prior examiner draw from the latest durable step.
    seq=witness["both_normal_commands_at_each_public_state"];prefix=witness["physical_fixed_history"]
    warm=folder/"warm_checkpoint.pickle.gz";meta=folder/"warm_checkpoint.json"
    tic=time.perf_counter();env=K.checked_env(raw);s=case["seat"];op=1-s;cfg=raw["game_configuration"]
    own={k:R.live.module(p,"decoder_shared_"+k) for k,p in packs.items()}
    rec=R.native_clock.Recorder(m);opponent=rec.wrap(m.agent);trace=[];prior_clock=[]
    if warm.exists():
        import pickle
        payload=pickle.loads(gzip.decompress(warm.read_bytes()))
        assert payload["identity"]==identity
        n=len(payload["trace"])
        trace=payload["trace"];prior_clock=payload["clock"]
        for i in range(n):
            o=K.public(env,s)
            assert K.digest(K.compact(o))==trace[i]["own_sha"]
            for k,model in own.items(): assert K.own_call(model,o,cfg)[0]==trace[i][k]
            env.step(prefix[i])
        K.restore(m,payload["snapshot"]);assert K.sig(m)==payload["state_sha"]
    for t in range(len(trace),case["t"]+1):
        my=K.public(env,s);enemy_obs=K.public(env,op)
        assert K.digest(K.compact({k:v for k,v in my.items() if k!="remainingOverageTime"}))==seq[t]["public_obs_sha256"]
        en=R.live.clean(opponent(enemy_obs,cfg))
        act={k:K.own_call(model,my,cfg)[0] for k,model in own.items()}
        assert (act["parent"],act["student"])==(seq[t]["parent"],seq[t]["student"])
        e=dict(t=t,own_sha=K.digest(K.compact(my)),enemy_sha=K.digest(K.compact(enemy_obs)),
               original_proposed=en,original_executed=en if t==case["t"] else R.live.clean(prefix[t][op]),**act)
        trace.append(e)
        if t==case["t"]:
            assert act["parent"]==case["parent"] and act["student"]==case["student"]
            root=dict(**{k:act[k] for k in ("parent","student")},t=t,enemy=en,archived_enemy=case["root_enemy"],
                      own_sha=e["own_sha"],enemy_sha=e["enemy_sha"])
            assert not rec.errors
            calls=prior_clock+rec.calls
            assert [z["step"] for z in calls]==list(range(t+1))
            frozen=K.capture(m);state=K.sig(m);K.restore(m,frozen);assert K.sig(m)==state
            K.blob_write(snap,gzip.compress(frozen,mtime=0))
            bank=dict(identity=identity,case=case,prefix=prefix,trace=trace,root=root,host=host,cpu=identity["cpu"],
                clock=dict(calls=calls,summary=R.native_clock.summarize(calls,m._S_ABORT,m._S_D2)),wall_s=time.perf_counter()-tic,
                snapshot_sha=K.digest(frozen),state_sha=state,prefix_sha=case["prefix_sha256"])
            R.write(out,bank)
            if warm.exists():warm.unlink()
            print("saved shared root",case["id"],"original root enemy",K.digest(K.compact(en)),flush=True)
            return bank
        m._ST["tr"].set_actions(enemy_obs["players"],prefix[t][op])
        env.step(prefix[t])
        if (t+1)%10==0:
            import pickle
            tmp=warm.with_suffix(".incomplete")
            tmp.write_bytes(gzip.compress(pickle.dumps(dict(identity=identity,trace=trace,clock=prior_clock+rec.calls,
                state_sha=K.sig(m),snapshot=K.capture(m))),mtime=0))
            tmp.replace(warm)


def branch(case,raw,folder,binding,m,common,arm,packs):
    out=folder/(arm+".json.gz")
    if out.exists():
        data=R.load(out);assert data["identity"]==binding and data["state_sha"]==common["state_sha"];return
    s=case["seat"];op=1-s;cfg=raw["game_configuration"];t0=time.perf_counter()
    env=K.checked_env(raw);keeper=R.live.module(packs["continuation"],"decoder_continue_"+arm)
    for i,both in enumerate(common["prefix"]):
        o=K.public(env,s);assert K.own_call(keeper,o,cfg)[0]==common["trace"][i]["continuation"]
        env.step(both)
    o=K.public(env,s);enemy=K.public(env,op);t=case["t"]
    assert K.digest(K.compact(o))==common["root"]["own_sha"] and K.digest(K.compact(enemy))==common["root"]["enemy_sha"]
    assert K.own_call(keeper,o,cfg)[0]==common["trace"][t]["continuation"]
    snap=gzip.decompress((folder/"original_root.pickle.gz").read_bytes())
    assert K.digest(snap)==common["snapshot_sha"]
    K.restore(m,snap);assert K.sig(m)==common["state_sha"]
    rec=R.native_clock.Recorder(m);native=rec.wrap(m.agent)
    root_joint=K.physical(common["root"][arm],common["root"]["enemy"],s)
    tape=[root_joint];env.step(root_joint);observed=[];own_wall=[]
    while not env.done:
        a=K.public(env,s);b=K.public(env,op)
        observed.append([K.digest(K.compact(a)),K.digest(K.compact(b))])
        en=R.live.clean(native(b,cfg));cmd,wall=K.own_call(keeper,a,cfg);own_wall.append(wall)
        joint=K.physical(cmd,en,s);tape.append(joint);env.step(joint)
    scores,cash,status=K.score(env);clock=K.clock_result(rec,m);end=R.native_clock.host_snapshot(m)
    R.native_clock.validate_host(end,[binding["cpu"]])
    assert [v["step"] for v in rec.calls]==list(range(t+1,t+len(tape)))
    d=dict(identity=binding,arm=arm,root_t=t,root_joint=root_joint,seat=s,seed=case["seed"],case_terrain=case["terrain"],tape=tape,tape_sha=K.digest(K.compact(tape)),clock=clock,own_public_seconds=own_wall,suffix_observations=observed,status=status,physical_cash=cash,official_scores=scores,official_margin=scores[s]-scores[op],host_at_start=rec.start,host_at_end=end,state_sha=common["state_sha"],snapshot_sha=common["snapshot_sha"],shared_sha=R.sha(folder/"shared.json.gz"),wall_s=time.perf_counter()-t0)
    R.write(out,d);print("terminal",case["id"],arm,scores,"margin",d["official_margin"],flush=True)


def run(idx,cpu):
    plan,case,rc,packs,raw,witness,folder,identity=setup(idx,cpu)
    with (folder/"execution.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        m,host=K.load_master(cpu)
        common=shared(case,packs,raw,witness,folder,identity,m,host)
        for arm in rc["arm_order"]:branch(case,raw,folder,identity,m,common,arm,packs)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--idx",required=True,type=int,choices=(0,1));parser.add_argument("--cpu",required=True,type=int)
    args=parser.parse_args();run(args.idx,args.cpu)
