"""Re-evaluate all three static decision functions on each candidate's own game history."""
from collections import Counter
import fcntl
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

from build import HERE, ROOT, PARENT, read, sha, seal
sys.path.insert(0,str(ROOT/"code"))
from fastsim import FastCarbon


def player(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    policy=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    return policy


def tape(arm, game):
    path=HERE/arm/game["raw_file"]
    if sha(path)!=game["raw_sha256"]:
        raise ValueError("changed game transcript")
    with gzip.open(path,"rt") as stream:
        return json.load(stream)


def different(first, other, seat):
    for t,(a,b) in enumerate(zip(first["actions"],other["actions"])):
        if a!=b:
            return {"step":t,"candidate_seat_differs":a[seat]!=b[seat],"opponent_differs":a[1-seat]!=b[1-seat]}
    return None


def evidence(result):
    games=[]
    for row in result["paired_map_seats"]:
        actual=tape("candidate",row["candidate"])
        controls={arm:tape(arm,row[arm]) for arm in ("parent","control")}
        s=row["seat"]
        sim=FastCarbon(actual["game_configuration"]).reset(seed=row["seed"])
        nets={arm:player(path,"causal_replay_"+arm) for arm,path in [
              ("parent",PARENT),("control",HERE/"control.py"),("candidate",HERE/"candidate.py")]}
        changes=[]
        for t,joint in enumerate(actual["actions"]):
            obs=sim.obs(s)
            moves={arm:p.agent(obs,sim.config) for arm,p in nets.items()}
            if moves["candidate"]!=joint[s]:
                raise ValueError("native candidate command not reproduced")
            edits={}
            for arm in ("parent","control"):
                original=moves[arm]
                edited=[(uid,original.get(uid,"NONE"),joint[s].get(uid,"NONE")) for uid in sorted(set(original)|set(joint[s]))
                        if original.get(uid)!=joint[s].get(uid)]
                if edited:edits[arm]=edited
            if edits:
                changes.append({"step":t,"vs":edits})
            sim.step(*joint)
        if list(sim.cash)!=actual["last_post_cash"] or row["candidate"]["official_scores"]!=actual["official_score_by_seat"]:
            raise ValueError("replay terminal state differs")
        games.append(dict(seed=row["seed"],seat=s,steps=len(actual["actions"]),events=changes,
                          first_actual_trajectory_divergence={arm:different(actual,raw,s) for arm,raw in controls.items()}))
    return games


def main(wait):
    release=HERE.parents[1]/"tree_response_training/supplement_01/economy_slot/released.json"
    with (HERE/".attribution.lock").open("a") as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        while not release.exists() and wait:time.sleep(12)
        if not release.exists():raise ValueError("native evidence is not yet released")
        result=read(HERE/"comparison.json")
        if sha(HERE/"comparison.json")!=read(release)["comparison_sha256"]:
            raise ValueError("native release differs")
        games=evidence(result)
        edits={a:sum(len(event["vs"].get(a,[])) for g in games for event in g["events"]) for a in ("parent","control")}
        changed_steps={a:sum(a in event["vs"] for g in games for event in g["events"]) for a in edits}
        record={"games":games,"total_steps":sum(g["steps"] for g in games),"different_orders_vs":edits,
                "different_steps_vs":changed_steps,"comparison_sha256":sha(HERE/"comparison.json"),
                "native_release_sha256":sha(release),"attribution_code_sha256":sha(Path(__file__))}
        raw=json.dumps(record,allow_nan=False,separators=(",",":")).encode()
        seal(HERE/"actual_attribution.json.gz",gzip.compress(raw,mtime=0))
        lines=["# 经济辅助训练的原版实战结果","",
               "两张开发图，三臂各换席四局。每局以自身物理席的官方终分减另一席计分。","",
               "| 地图 / 席 | 原 v1 分差 | 同训控制分差 | 辅助候选分差 | 较控制变化 |",
               "|---|---:|---:|---:|---:|"]
        for r in result["paired_map_seats"]:
            p,c,a=(r[k]["margin"] for k in ("parent","control","candidate"))
            lines.append(f"| {r['seed']} / {r['seat']} | {p:,.2f} | {c:,.2f} | {a:,.2f} | {a-c:+,.2f} |")
        lines.extend(["","| 臂 | 胜 / 平 / 负 | 官方平均分差 |","|---|---:|---:|"])
        for key,label in (("parent","原 v1"),("control","同训控制"),("candidate","辅助候选")):
            w=result["rollup"][key];avg=sum(r[key]["margin"] for r in result["paired_map_seats"])/4
            lines.append(f"| {label} | {w['win']} / {w['draw']} / {w['loss']} | {avg:,.2f} |")
        verdict=("达到小样本双对照续验条件" if result["verdict"]=="exploratory_followup_signal" else "未达双对照条件，保留 v1")
        lines.extend(["",verdict+"。本场两张图均为开发图。正式指标须另用预留的 100 张全新地图。","",
                f"在候选四局实际走出的 {record['total_steps']} 个公开局面上逐帧重算三臂。相对旧 v1，候选在 {changed_steps['parent']} 步实际改变 {edits['parent']} 条命令；相对同训控制，在 {changed_steps['control']} 步改变 {edits['control']} 条命令。整局还包含随机考官独立作答造成的差异。每局两臂何时首次走入不同轨迹可在归因数据检查。",
                "","[三臂官方对照、逐图聚类区间与预注册判据](comparison.json) · [逐步真实命令差异](actual_attribution.json.gz)"])
        seal(HERE/"RESULTS.md",("\n".join(lines)+"\n").encode())


if __name__=="__main__":
    main("--wait" in sys.argv)
