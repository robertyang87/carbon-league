"""Play and verify one preassigned map and seat at a time; resume official evidence without resampling."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean
import subprocess
import sys
import tempfile
import time

from model import HERE,FIT,BASE,ROOT,PLAN,PLAN_PATH,PLAN_SHA,sha,load,atomic,source_checks,theta,package,manifest,write_pack,trial
from official_game_replay import check as official_check
import tree_audited_game as arena
import tree_win_gate as gate

ARMS={"train":["plus","minus"],"development":["v1","candidate"]}
THREAD_ENV={**os.environ,"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1","PYTHONDONTWRITEBYTECODE":"1"}


def folder(stage,seed,arm):
    return HERE/stage/str(seed)/arm


def alpha_for(stage,arm):
    if stage=="train":
        return PLAN["epsilon"] if arm=="plus" else -PLAN["epsilon"]
    cp=load(HERE/"checkpoint.json")
    assert cp["training_gate_passed"] and cp==fit()
    return 0. if arm=="v1" else cp["alpha"]


def identities(alpha):
    raw=package(alpha)
    obj=manifest(alpha,raw)
    return hashlib.sha256(raw).hexdigest(),trial.manifest_sha(obj)


def verify(wd,seed,seat,alpha,replay=False):
    pkg,weight=identities(alpha)
    raw_path=wd/f"raw_{seed}_{seat}.json.gz"
    tape=load(raw_path)
    native=wd/f"native_seat{seat}.json"
    proof=wd/f"official_seat{seat}.json"
    map_sha=load(HERE/"clearance.json")["boards"][str(seed)]
    assert tape["seed"]==seed and tape["candidate_seat"]==seat
    assert tape["pack_sha256"]==pkg and tape["weight_identity"]==weight and tape["examiner_md5"]==PLAN["examiner_md5"]
    assert not tape["errors"] and not tape["audit_messages"] and tape["initial_terrain_u8_sha256"]==map_sha
    assert gate.map_identity.terrain_sha256(tape["initial_full_carbon"])==map_sha
    gate.validate_examiner_configuration(tape["configuration"])
    steps=tape["steps"];calls=tape["search_calls"];kinds=Counter(k for _,k in calls)
    assert [t for t,k in calls]==list(range(steps)) and kinds["searched"]>0
    assert set(kinds)<={"searched","single","depth_zero_return_base"}
    assert len(tape["actions"])==steps and 0<steps<=299 and (steps==299 or any(tape["extinct_by_seat"]))
    row={"seed":seed,"candidate_seat":seat,"raw_sha256":sha(raw_path),"raw_file":raw_path.name,"actual_transitions":steps,
         "margin":tape["margin"],"official_score_by_seat":tape["official_score_by_seat"],"result":tape["result"]}
    if native.exists():
        full=load(native);assert all(full[k]==v for k,v in row.items())
        gate.validate_game(full)
        gate.verify_raw(wd,full,pkg,weight,expected_terrain=map_sha)
    if proof.exists() and not replay:
        official=load(proof)["official"]
    else:
        official=official_check(wd,row)
    scores=official["official_reward_by_seat"];margin=scores[seat]-scores[1-seat]
    assert (official["seed"],official["seat"],official["transcript_sha256"],official["map_sha256"])==(seed,seat,row["raw_sha256"],map_sha)
    assert official["steps"]==steps and official["statuses"]==["DONE","DONE"] and scores==tape["official_score_by_seat"]
    assert abs(margin-tape["margin"])<1e-6 and official["result"]==("win" if margin>0 else "loss" if margin<0 else "draw")
    u,j=trial.terminal(margin)
    out=dict(protocol_sha256=PLAN_SHA,pack_sha256=pkg,weights_sha256=weight,official=official,
             raw_sha256=row["raw_sha256"],native_receipt_sha256=sha(native) if native.exists() else None,
             note="complete native counters" if native.exists() else "recovered original tape after interruption; search outcomes counted per recorded invocation",
             search_kinds=dict(kinds),margin=margin,U=u,J=j)
    trial.frozen_file(proof,out)
    return out


def game(stage,seed,arm,seat,cpu):
    assert stage in ARMS and arm in ARMS[stage] and seed in PLAN[stage+"_maps"] and seat in PLAN["seats"]
    assert inputs_ready() and os.sched_getaffinity(0)>={cpu}
    os.sched_setaffinity(0,{cpu})
    alpha=alpha_for(stage,arm)
    wd=folder(stage,seed,arm);wd.mkdir(exist_ok=True,parents=True)
    # Per-seat lock protects concurrent invocations and replay recovery.
    with (wd/f".seat{seat}.lock").open("a") as hold:
        fcntl.flock(hold,fcntl.LOCK_EX|fcntl.LOCK_NB)
        raw=wd/f"raw_{seed}_{seat}.json.gz"
        if not raw.exists():
            if (wd/f"native_seat{seat}.json").exists() or (wd/f"official_seat{seat}.json").exists():
                raise RuntimeError("published/source game missing; restore it before resuming")
            with tempfile.TemporaryDirectory() as tmp:
                agent,weights=write_pack(Path(tmp),alpha)
                row=arena.play((str(agent),sha(weights),seed,seat,cpu,str(wd),sha(agent)))
                trial.frozen_file(wd/f"native_seat{seat}.json",row)
        receipt=verify(wd,seed,seat,alpha,replay=True)
        print(stage,seed,arm,seat,round(receipt["margin"],2),receipt["official"]["result"],flush=True)


def consume(stage):
    results={}
    for seed in PLAN[stage+"_maps"]:
        results[seed]={}
        for arm in ARMS[stage]:
            alpha=alpha_for(stage,arm)
            results[seed][arm]=[verify(folder(stage,seed,arm),seed,s,alpha) for s in PLAN["seats"]]
    return results


def contrast(rows,a,b,key):
    return [v[key]-w[key] for v,w in zip(rows[a],rows[b])]


def decide(map_seat_deltas):
    if len(map_seat_deltas)!=2 or any(len(v)!=2 for v in map_seat_deltas):raise ValueError("need exactly two maps with both seats")
    ds=[mean(row) for row in map_seat_deltas]
    assert all(math.isfinite(v) for row in map_seat_deltas for v in row)
    avg=mean(ds);sign=1 if avg>0 else -1 if avg<0 else 0
    g=(sign!=0 and all(sign*v>=PLAN["minimum_each_absolute_map_delta_J"] for v in ds)
          and abs(avg)>=PLAN["minimum_absolute_map_mean_delta_J"]
          and sum(sign*v>0 for row in map_seat_deltas for v in row)>=PLAN["min_same_sign_seats"])
    slope=avg/(2*PLAN["epsilon"])
    cap=PLAN["scalar_update_cap"]
    return ds,avg,slope,g,max(-cap,min(cap,PLAN["learning_rate"]*slope)) if g else 0.


def fit():
    rows=consume("train")
    deltas=[contrast(rows[s],"plus","minus","J") for s in PLAN["train_maps"]]
    ds,avg,derivative,passed,alpha=decide(deltas)
    return dict(protocol_sha256=PLAN_SHA,parent_v1_sha256=PLAN["parent_sha256"],direction=PLAN["direction"],
                training_gate_passed=passed,mean_map_delta_J=avg,map_deltas=ds,derivative=derivative,alpha=alpha,
                coefficients=theta(alpha),pack_sha256=identities(alpha)[0],maps=[dict(seed=seed,seat_delta_J=deltas[i],
                map_delta_J=ds[i],arms={arm:[dict(seat=r["official"]["seat"],margin=r["margin"],U=r["U"],J=r["J"],
                wdl=r["official"]["result"],raw_sha256=r["raw_sha256"]) for r in runs] for arm,runs in rows[seed].items()})
                for i,seed in enumerate(PLAN["train_maps"])])


def released():
    old=FIT/"replication";cfg=load(old/"protocol.json")
    for arm in cfg["arms"]:
        wd=old/arm
        if not (wd/"official_replay.json").exists() or not (wd/"results.json").exists():return False
        rr=load(wd/"results.json");audit=load(wd/"official_replay.json")["runs"][arm]
        exp={(s,p) for s in cfg["maps"] for p in cfg["physical_seats"]}
        assert rr["status"]=="complete" and len(audit["games"])==len(rr["games"])==4
        assert {(g["seed"],g["seat"]) for g in audit["games"]}==exp
        tapes={(g["seed"],g["candidate_seat"]):g for g in rr["games"]}
        for g in audit["games"]:
            r=tapes[g["seed"],g["seat"]]
            assert g["statuses"]==["DONE","DONE"] and g["transcript_sha256"]==r["raw_sha256"]==sha(wd/r["raw_file"])
            assert g["official_reward_by_seat"]==r["official_score_by_seat"] and r["valid"]
    return True


def busy_external_games():
    active=[]
    for cmdfile in Path("/proc").glob("[0-9]*/cmdline"):
        try:a=[b.decode(errors="replace") for b in cmdfile.read_bytes().split(b"\0") if b]
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if not a or "python" not in Path(a[0]).name:continue
        scripts=[Path(x) for x in a[1:4] if x.endswith(".py")]
        if any(p.name=="tree_pack_arena.py" or (p.parent==FIT/"replication" and p.name=="run.py" and "evaluate" not in a[2:]) for p in scripts):
            active.append(int(cmdfile.parent.name))
    return active


def inputs_ready():
    if not (HERE/"older_registry_crosscheck.json").exists() or not (HERE/"clearance.json").exists() or not (HERE/"mechanism.json").exists() or not released():return False
    c=load(HERE/"clearance.json");m=load(HERE/"mechanism.json")
    assert c["protocol_sha256"]==m["protocol_sha256"]==PLAN_SHA and c["verdict"]=="eligible_in_checked_scope"
    assert set(map(int,c["boards"]))==set(PLAN["train_maps"]+PLAN["development_maps"])
    return m["passed"]


def run_stage(stage):
    for i,seed in enumerate(PLAN[stage+"_maps"]):
        tasks=[]
        for j,arm in enumerate(ARMS[stage]):
            for seat in PLAN["seats"]:
                cpu=2*((j+i)%2)+seat
                tasks.append([sys.executable,"-u",str(Path(__file__).resolve()),"game",stage,str(seed),arm,str(seat),str(cpu)])
        with ThreadPoolExecutor(max_workers=4) as pool:
            runs=[pool.submit(subprocess.run,cmd,check=True,env=THREAD_ENV) for cmd in tasks]
            for work in runs:work.result()


def development(cp):
    if not cp["training_gate_passed"]:
        return {"formal":False,"ran":False,"games":0,"selected_sha256":PLAN["parent_sha256"],"reason":"training gate retained byte-identical v1; redundant games skipped"}
    groups=consume("development");m=[]
    for seed,rows in groups.items():
        dm=contrast(rows,"candidate","v1","margin");du=contrast(rows,"candidate","v1","U")
        def brief(r):return {k:r[k] for k in ("official","margin","U","J")}
        m.append(dict(seed=seed,paired_margin_deltas=dm,paired_U_deltas=du,mean_margin_delta=mean(dm),
                      mean_U_delta=mean(du),arms={a:[brief(x) for x in rr] for a,rr in rows.items()}))
    wins={arm:sum(a["official"]["result"]=="win" for z in m for a in z["arms"][arm]) for arm in ARMS["development"]}
    passed=all(v["mean_margin_delta"]>0 and v["mean_U_delta"]>0 for v in m) and sum(u>0 for v in m for u in v["paired_U_deltas"])>=3 and wins["candidate"]>=wins["v1"]
    return dict(formal=False,ran=True,games=8,won_replication_gate=passed,map_results=m,equal_map_mean_margin_delta=mean(v["mean_margin_delta"] for v in m),
                equal_map_mean_U_delta=mean(v["mean_U_delta"] for v in m),wins=wins,selected_sha256=cp["pack_sha256"] if passed else PLAN["parent_sha256"])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command",choices=["run","game","report"]);ap.add_argument("args",nargs="*")
    a=ap.parse_args()
    if a.command=="game":
        stage,seed,arm,seat,cpu=a.args
        return game(stage,int(seed),arm,int(seat),int(cpu))
    source_checks()
    if a.command=="report":
        report();return
    with PLAN_PATH.open("rb") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            if (HERE/"mechanism.json").exists() and not load(HERE/"mechanism.json")["passed"]:
                trial.frozen_file(HERE/"stopped.json",dict(reason="no actual action coverage on the preregistered roots",parent=PLAN["parent_sha256"]));return
            if inputs_ready() and not busy_external_games():break
            time.sleep(12)
        run_stage("train")
        cp=fit();trial.frozen_file(HERE/"checkpoint.json",cp)
        if cp["training_gate_passed"]:
            packed=package(cp["alpha"])
            target=HERE/"candidate.py"
            if target.exists():assert target.read_bytes()==packed
            else:atomic(target,packed)
            run_stage("development")
        result=development(cp)
        trial.frozen_file(HERE/"development.json",result)
        report()


def report():
    cp=load(HERE/"checkpoint.json");out=load(HERE/"development.json")
    assert cp==fit() and out==development(cp)
    lines=["# 跨地图终局回报训练：返库和避险的第二方向","",
           "以冻结的 v1 双网 9c54df17 为中心。正方向加强载货返库与避险，同时降低追逐敌货和继续采收的偏好。两张训练图各自进行正、负扰动的双席完整对局。每局的官方终局分差决定一次回报 J，两席等权得地图差，两图再等权。原版考官温度为 1，搜索受本机实际时钟影响。","",
           "| 训练地图 | 席 | 正扰动分差 / 胜负 | 负扰动分差 / 胜负 | J正−J负 |","|---|---:|---:|---:|---:|"]
    for m in cp["maps"]:
        for s,(p,n) in enumerate(zip(m["arms"]["plus"],m["arms"]["minus"])):
            lines.append(f'| {m["seed"]} | {s} | {p["margin"]:+.2f} / {p["wdl"]} | {n["margin"]:+.2f} / {n["wdl"]} | {m["seat_delta_J"][s]:+.4f} |')
    lines+=["",f'两张训练图的平均 J 差分别为 {cp["map_deltas"][0]:+.4f}、{cp["map_deltas"][1]:+.4f}，等权平均 {cp["mean_map_delta_J"]:+.4f}。斜率估计 {cp["derivative"]:+.4f}，依预注册条件采用的实际位移 α 为 {cp["alpha"]:+.4f}。']
    if not out["ran"]:
        lines+=["","训练信号未达预设的跨图采纳条件，保留与原 v1 完全相同的模型。两张预留开发图没有开赛。"]
    else:
        lines+=["","| 开发图 | 席 | v1分差 | 候选分差 | Δ分差 | ΔU |","|---|---:|---:|---:|---:|---:|"]
        for m in out["map_results"]:
            for s,(a,b) in enumerate(zip(m["arms"]["v1"],m["arms"]["candidate"])):
                lines.append(f'| {m["seed"]} | {s} | {a["margin"]:+.2f} | {b["margin"]:+.2f} | {m["paired_margin_deltas"][s]:+.2f} | {m["paired_U_deltas"][s]:+.4f} |')
        lines+=["",f'两图平均分差改变 {out["equal_map_mean_margin_delta"]:+.2f}，平均 ΔU 为 {out["equal_map_mean_U_delta"]:+.4f}。两图四局胜场数 v1 {out["wins"]["v1"]}、新候选 {out["wins"]["candidate"]}。开发条件达成：{out["won_replication_gate"]}。']
    lines+=["",f'本轮保留的提交包 SHA-256 为 `{out["selected_sha256"]}`。每组只有两张独立地图，双席、正负扰动均嵌套在地图内。以上为开发数据，正式胜率仍需独立考试。',"",
            "[预注册](protocol.json) · [地图隔离](clearance.json) · [正交扰动实际改令](mechanism.json) · [训练检查点](checkpoint.json) · [开发结论](development.json) · [各训练席完整行动与官方核验](train/)"]
    lines+=["","时间口径：验图曾与上一轮第一方向的八局复测共享宿主 4 核总配额。第一轮复测的两臂在同一时间段运行，所有既有结果保留。本轮正负扰动的原版考官对局在前批八份官方核验全部出现后才启动，头两个逻辑核与另两个核在第二张地图交换分组。"]
    atomic(HERE/"README.md",("\n".join(lines)+"\n").encode())
    print("Finished multimap",cp["map_deltas"],cp["training_gate_passed"],out["selected_sha256"],flush=True)


if __name__=="__main__":
    main()
