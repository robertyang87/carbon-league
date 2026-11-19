"""Switch the entire Sapling partner policy at a shared historical decision, then let both sides respond to terminal."""
import argparse
from collections import Counter
import fcntl
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
DEV=HERE.parent
ACTOR=DEV.parent
ROOT=ACTOR.parents[2]
STRAT=ACTOR/"stratified_action_value_dev"
PARTNER=ACTOR/"actor_partner_dev"
COURT=ACTOR/"tree_response_training/.continuation.lock"
INDEX=STRAT/"source/inventory.json"
LEDGER=HERE/"active_time.json"
PACK={"it20":PARTNER/"candidate.py","it50":ROOT/"data/weights/sapling_1200k/terminal_return_fit_v1.py"}


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


engine=load("registered_per_turn_original_driver",STRAT/"native/run.py")
read,sha,dig,check,freeze,live=engine.read,engine.sha,engine.dig,engine.check,engine.freeze,engine.live
sys.path.insert(0,str(STRAT))
budget=load("registered_active_clock",STRAT/"budget.py")
MAX_SECONDS=2100


def manifest():
    plan=read(DEV/"protocol.json");rows=read(DEV/"roots.json")
    check(isinstance(rows,dict) and rows["protocol_sha256"]==sha(DEV/"protocol.json") and
          rows["census_sha256"]==sha(DEV/"census.json.gz"),"frozen study census changed")
    rows=rows["roots"]
    check(sha(INDEX)==plan["source_inventory_sha256"] and all(sha(path)==plan["partners"][arm] for arm,path in PACK.items()),"partner or raw register differs")
    observed=read(INDEX)["v1_inventory"];lookup={r["raw_sha256"]:r for r in observed}
    eligible={r["raw_sha256"]:r for r in plan["eligible"]}
    check(len({r["map"] for r in rows})==len(rows)==10 and len({r["id"] for r in rows})==10,"ten distinct source maps")
    check(Counter((r["split"],r["regime"]) for r in rows)=={("train","pressure"):3,("train","stable"):3,("held","pressure"):2,("held","stable"):2},"wrong allocation")
    check(plan["budgets"]["active_seconds"]==MAX_SECONDS and plan["budgets"]["max_native_arms"]==24 and plan["budgets"]["native_workers"]==1,"wrong resource limit")
    for r in rows:
        reg=eligible[r["raw_sha256"]];full=lookup[r["raw_sha256"]]
        check(all(r[k]==reg[k] for k in ("map","seat","seed","raw","split")) and
              all(r[k]==full[k] for k in ("map","seat","seed","raw")) and
              full["pack_sha256"]==sha(PACK["it50"]) and r["t"] in plan["budgets"]["root_times"] and re.fullmatch(r"[A-Za-z0-9_-]+",r["id"]),
              "unregistered source, seat, step or ID")
        r.update(official=full["official"],official_sha256=full["official_sha256"])
    return plan,rows


def artifacts():
    plan,rows=manifest()
    original=engine.prior.SOURCE
    canonical=subprocess.check_output(["git","show","HEAD:sources/master3s2/master3s2.py"],cwd=ROOT)
    check(sha(original)==live.MASTER_SHA==plan["partners"]["examiner"]==__import__("hashlib").sha256(canonical).hexdigest()
          and sha(DEV/"policy.py")==plan["frozen_policy_sha256"],"original or selector changed")
    return {str(p.relative_to(ROOT)):sha(p) for p in [DEV/"protocol.json",DEV/"roots.json",DEV/"policy.py",Path(__file__),
            STRAT/"native/run.py",STRAT/"budget.py",Path(engine.prior.__file__),Path(live.__file__),engine.prior.SHARED,
            ACTOR.parent/"root_recovery_branches/branches.py",ROOT/"code/fastsim.py",original,INDEX,*PACK.values()]}


def prepare():
    receipt={"inputs":artifacts(),"opponent_root":"actual logged simultaneous command common to both partners",
             "continuation":"Original default search and T=1. Per-pair prior history rebuilds the enemy belief and previous observation. Original historical MT19937 and latency estimates are unavailable: both continuations start with the identical new registered random stream and identical published warm search costs. Later clock-dependent search budgets may diverge.",
             "actor_memory":"Each branch initializes its own public pack and rebuilds all its own visible features to the root. The named partner controls every later complete joint order.",
             "budget_seconds":MAX_SECONDS,"max_simultaneous_native_workers":1}
    freeze(HERE/"frozen.json",receipt)
    return receipt


def cards_for(split):
    _,rows=manifest()
    out=[{**r,"stream":"primary"} for r in rows if r["split"]==split]
    if split=="train":
        for regime in ("pressure","stable"):
            first,=[r for r in out if r["regime"]==regime and r.get("repeat")]
            check(first==next(r for r in out if r["regime"]==regime),"unexpected repeat")
            out.append({**first,"stream":"repeat"})
    return out


def prefix(row,arm,master=None):
    tape=engine.source(row);seat=row["seat"];enemy=1-seat;turn=row["t"]
    actor=live.module(PACK[arm],"persistent_partner_"+arm)
    conf=actor._HYBRID
    check(conf["student_weight"]==.25 and conf["expert_weight"]==.75 and not conf["guard"] and not conf["search"],"nonregistered mixture")
    env=live.begin(tape)
    tracker=master.BeliefTracker(master._Bel,enemy) if master else None
    for t,joint in enumerate(tape["actions"][:turn+1]):
        o=live.public(env,seat);v=live.public(env,enemy)
        if tracker is not None:
            tracker.step(v["players"],t,v["carbon"]);tracker.set_actions(v["players"],joint[enemy])
        if t in (0,turn):
            command=actor.agent(o,tape["game_configuration"])
            if arm=="it50":check(joint[seat]==command,"frozen parent failed on its signed root/history")
        else:
            actor._FEAT.observe(o)
        if t<turn:env.step(joint)
    return (env,actor,tracker,o,v,tape),command,joint[enemy]


def extract(row):
    projections={}
    for arm in PACK:
        result,cmd,op=prefix(row,arm)
        _,_,_,own_public,enemy_public,_=result
        projections[arm]=(cmd,op,dig(own_public),dig(enemy_public))
    a,b=projections["it20"],projections["it50"]
    check(a[1:]==b[1:],"branches do not share the historical information and enemy order")
    for key,target in (("root_enemy",a[1]),("it20",a[0]),("it50",b[0])):
        if key in row:check(row[key]==target,"registered root command differs")
    return {**row,"it20":a[0],"it50":b[0],"root_enemy":a[1],"own_public":a[2],"enemy_public":a[3]}


def rng_seed(card):
    return int(dig(["sapling-partner-continuation-v1",card["raw_sha256"],card["t"],card["id"],card["stream"]])[:8],16)


SCALARS = ("searched", "single", "aborted", "changed")
HISTOGRAMS = ("depth", "rolls", "cands")


def stats_before(master):
    stats=master._S_STATS
    return ({k:int(stats[k]) for k in SCALARS},
            {k:dict(stats[k]) for k in HISTOGRAMS}, stats.get("last"))


def stats_after(master,before,cpu_ns,wall_ns):
    scalars,histograms,old_last=before
    now=master._S_STATS
    result={"process_time_ns":cpu_ns,"wall_time_ns":wall_ns,
            **{k:int(now[k])-scalars[k] for k in SCALARS}}
    added={}
    for k in HISTOGRAMS:
        d={int(x):int(n)-int(histograms[k].get(x,0)) for x,n in now[k].items() if int(n)!=int(histograms[k].get(x,0))}
        check(all(n==1 for n in d.values()) and len(d)==result["searched"],"search histogram increment mismatch")
        added[k]=d
        result[k]=next(iter(d),None)
    check(all(result[k] in (0,1) for k in SCALARS) and result["changed"]<=result["searched"] and cpu_ns>0 and wall_ns>0,
          "invalid original per-call diagnostics")
    result["histogram_deltas"]=added
    last=now.get("last")
    result["attempted_counts"]={"candidates":last[0],"depth":last[1],"rolls":last[2]} if last is not None and last is not old_last else None
    result["null_reason"]=None if result["searched"] else ("single candidate before search" if result["single"] else
                 "aborted without a completed depth" if result["aborted"] else "early return before search statistics")
    return result


def measured_arm():
    # Preserve the original game loop, issued ticket, RNG restoration and atomic turn commit.
    # The expensive part of the meter runs outside master.agent's internal wall-clock window.
    source=inspect.getsource(engine.run_arm)
    old='start=time.monotonic();other=master.agent(v,cfg);wall=time.monotonic()-start'
    new=("pre_stats=stats_before(master); wall0=time.perf_counter_ns(); cpu0=time.process_time_ns(); "
         "other=master.agent(v,cfg); cpu_delta=time.process_time_ns()-cpu0; wall_delta=time.perf_counter_ns()-wall0; "
         "wall=wall_delta/1e9; measurement=stats_after(master,pre_stats,cpu_delta,wall_delta)")
    tag='"wall_seconds":wall,"issue":ticket'
    check(source.count(old)==1 and source.count(tag)==1,"upstream continuation hook changed")
    source=source.replace(old,new).replace(tag,'"wall_seconds":wall,"original_search_meter":measurement,"issue":ticket')
    ns={**vars(engine),"stats_before":stats_before,"stats_after":stats_after}
    exec(compile(source,engine.__file__+"::per-turn-meter","exec"),ns)
    return ns["run_arm"]


def meter_fixture():
    from types import SimpleNamespace
    stats={**{k:0 for k in SCALARS},**{k:{} for k in HISTOGRAMS}}
    obj=SimpleNamespace(_S_STATS=stats)
    old=stats_before(obj); stats["searched"]=1; stats["changed"]=1; stats["aborted"]=1
    for k,n in zip(HISTOGRAMS,(2,3,4),strict=True):stats[k][n]=1
    stats["last"]=(4,2,3,.1,());yes=stats_after(obj,old,100,120)
    old=stats_before(obj);stats["single"]=1; single=stats_after(obj,old,200,260)
    old=stats_before(obj);stats["aborted"]=2;stats["last"]=(5,0,0,.2,());zero=stats_after(obj,old,80,95)
    assert yes["depth"]==2 and yes["changed"]==1 and yes["aborted"]==1 and single["depth"] is None and single["attempted_counts"] is None
    assert zero["attempted_counts"]["depth"]==0 and zero["depth"] is None and zero["searched"]==0
    measured_arm()
    return {"searched_and_cut":yes,"single_after_previous_search":single,"depth_zero":zero}



def sub_arm(row,arm):
    # Called solely by the lock-holding supervisor with the descriptor inherited into this child.
    fd=int(os.environ["PARTNER_ARENA_FD"])
    check(os.fstat(fd).st_ino==COURT.stat().st_ino and os.fstat(fd).st_dev==COURT.stat().st_dev,"missing shared arena fd")
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    check(os.sched_getaffinity(0)=={1} and os.environ["CARBON_DEBUG"]=="1" and
          all(os.environ[k]=="1" for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")),"native computational identity")
    card=extract(row)
    folder=HERE/"cases"/row["stream"]/row["id"];folder.mkdir(parents=True,exist_ok=True)
    freeze(folder/"card.json",card)
    stored=read(HERE/"frozen.json")
    check(stored["inputs"]==artifacts(),"source bytes or protocol changed since preflight")
    def root_state(c,master):
        state,command,op=prefix(c,arm,master)
        check(command==c[arm] and op==c["root_enemy"] and dig(state[3])==c["own_public"] and dig(state[4])==c["enemy_public"],"root replay mismatch")
        return state
    engine.root_state=root_state
    engine.random_seed=rng_seed
    engine.frozen_hash=lambda:sha(HERE/"frozen.json")
    completed=measured_arm()(card,arm,folder,float(os.environ["PARTNER_MONOTONIC_DEADLINE"]),298-card["t"])
    return 0 if completed else 75


def write_labels(split,requested):
    pairs=[]
    for row in requested:
        folder=HERE/"cases"/row["stream"]/row["id"];entry=read(folder/"card.json")
        record={k:row[k] for k in ("id","stream","regime","split","map","seed","seat","t")}
        data={arm:read(folder/(arm+".json.gz")) for arm in PACK}
        for arm,a in data.items():
            check(a["binding"]=={"id":row["id"],"card_sha256":dig(entry),"arm":arm,"frozen_sha256":sha(HERE/"frozen.json"),"paired_seed":rng_seed(entry)}
                  and a["initial_public"]==[entry["own_public"],entry["enemy_public"]]
                  and a["joint0"][row["seat"]]==entry[arm] and a["joint0"][1-row["seat"]]==entry["root_enemy"]
                  and a["completed_original_calls"]==len(a["history"])==a["terminal_step"]-row["t"]-1
                  and a["statuses"]==["DONE","DONE"],"incomplete official arm")
            history=a["history"]
            check([r["t"] for r in history]==list(range(row["t"]+1,a["terminal_step"])) and
                  all("original_search_meter" in r and abs(r["wall_seconds"]-r["original_search_meter"]["wall_time_ns"]/1e9)<1e-14 for r in history),
                  "missing or repeated search metric")
            met=[r["original_search_meter"] for r in history]
            summary={k:sum(r[k] for r in met) for k in (*SCALARS,"process_time_ns","wall_time_ns")}
            summary["depth_histogram"]=dict(Counter(m["depth"] for m in met if m["depth"] is not None))
            summary["no_valid_depth"]=sum(m["depth"] is None for m in met)
            record[arm]={"runtime":summary,"margin":a["official_margin"],"win":a["official_margin"]>0,"draw":a["official_margin"]==0,
                         "scores":a["official_scores"],"steps":a["terminal_step"],"original_followup_calls":len(a["history"]),
                         "raw":str((folder/(arm+".json.gz")).relative_to(HERE)),"sha256":sha(folder/(arm+".json.gz"))}
        record["delta"]=record["it20"]["margin"]-record["it50"]["margin"];pairs.append(record)
    freeze(HERE/(split+"_labels.json"),{"pairs":pairs,"roots_sha256":sha(DEV/"roots.json"),
           "protocol_sha256":sha(DEV/"protocol.json"),"frozen_sha256":sha(HERE/"frozen.json")})


def admission(split):
    # Read only completion presence/count for the other study, never expose its game outcomes as features or labels.
    rep=PARTNER/"replication"
    for marker in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command=marker.read_bytes().split(b"\0")
            if not command or b"python" not in os.path.basename(command[0]):continue
            cwd=(marker.parent/"cwd").resolve()
            if any((cwd/os.fsdecode(a)).resolve()==rep/"native.py" for a in command[1:] if a.endswith(b".py")):
                return False
        except (OSError,RuntimeError):
            continue
    if not ((rep/"comparison.json").is_file() and len(list((rep/"games").glob("*/receipt_*.json")))==8):
        return False
    mech=read(DEV/"mechanism.json.gz")
    check(mech["gate"]=="PASS" and mech["roots_sha256"]==sha(DEV/"roots.json")
          and mech["protocol_sha256"]==sha(DEV/"protocol.json"),"root / two-policy mechanism closed")
    fixture=read(HERE/"timing_fixture.json")
    check(fixture["passed"] is True and fixture["original_search_calls"]==0 and
          fixture["native_sha256"]==sha(Path(__file__)) and fixture["frozen_sha256"]==sha(HERE/"frozen.json"),
          "per-turn meter fixture gate")
    if split=="held":
        h=read(DEV/"train_gate.json")
        check(h["decision"]=="PASS" and h["roots_sha256"]==sha(DEV/"roots.json") and
              h["train_labels_sha256"]==sha(HERE/"train_labels.json"),"held responses remain closed")
    return True


def run(split):
    if not admission(split):
        print("WAIT_FOR_REGISTERED_EIGHT");return 75
    prepare()
    requested=cards_for(split)
    for row in requested:
        with COURT.open("a") as slot:
            try:fcntl.flock(slot,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                print("arena in use");return 75
            for arm in ("it50","it20"):
                out=HERE/"cases"/row["stream"]/row["id"]/(arm+".json.gz")
                if out.exists():continue
                with budget.reserve(MAX_SECONDS,LEDGER) as remaining:
                    if remaining<=1:
                        print("STOP_ACTIVE_BUDGET");return 3
                    deadline=time.monotonic()+remaining
                    env={**os.environ,"PARTNER_MONOTONIC_DEADLINE":str(deadline),"PARTNER_ARENA_FD":str(slot.fileno()),
                         "CARBON_DEBUG":"1","OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1"}
                    cmd=["taskset","-c","1",sys.executable,str(Path(__file__).resolve()),"child",split,row["id"],row["stream"],arm]
                    try:status=subprocess.run(cmd,env=env,pass_fds=(slot.fileno(),),timeout=max(.001,deadline-time.monotonic())).returncode
                    except subprocess.TimeoutExpired:status=75
                    if status==75:
                        print("STOP_ACTIVE_BUDGET with saved progress");return 3
                    check(status==0,"original response failed")
    write_labels(split,requested)
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","fixture","run-approved","child"));parser.add_argument("split",nargs="?",choices=("train","held"))
    parser.add_argument("job",nargs="*");args=parser.parse_args()
    if args.action=="prepare":prepare();return 0
    if args.action=="fixture":
        prepare();r=cards_for("train")[0];c=extract(r)
        freeze(HERE/"timing_fixture.json",{"passed":True,"original_search_calls":0,"native_sha256":sha(Path(__file__)),
               "frozen_sha256":sha(HERE/"frozen.json"),"fixture":meter_fixture()})
        freeze(HERE/"root_fixture.json",{"id":r["id"],"map":r["map"],"t":r["t"],"public":[c["own_public"],c["enemy_public"]],
               "it20":c["it20"],"it50":c["it50"],"recorded_enemy":c["root_enemy"],"original_search_calls":0,
               "meter_contract_fixture":meter_fixture(),"primary_rng":rng_seed(c),"repeat_rng":rng_seed({**c,"stream":"repeat"}),"frozen_sha256":sha(HERE/"frozen.json")})
        return 0
    check(args.split in ("train","held"),"explicit split required")
    if args.action=="run-approved":return run(args.split)
    key,stream,arm=args.job
    check(admission(args.split),"registered eight games still active");r,=[r for r in cards_for(args.split) if r["id"]==key and r["stream"]==stream]
    check(arm in PACK,"unregistered arm");return sub_arm(r,arm)


if __name__=="__main__":
    sys.exit(main())
