"""Reconcile terminal margin deltas and identify the concrete costs of the largest cash reversals."""
from collections import Counter
import argparse
import fcntl
from statistics import mean
import time

import analyze as study
from crosscheck import corroborate

HERE, DEV, ARMS = study.HERE,study.DEV,study.ARMS


def cash(g, own):
    s = g["seat"] if own else 1-g["seat"]
    a = g["physical_cash"][s]
    result = {name:sum(a.get(k,0) for k in keys) for name,keys in study.CATEGORIES.items()}
    result["official_settlement"] = g["official_minus_closing_cash"][s]
    return result


def gaps(g):
    mine, opp = cash(g,True), cash(g,False)
    by = {k:mine[k]-opp[k] for k in mine}
    study.check(abs(sum(by.values())+g["cash_leads"][0]-g["official_margin"])<1e-5,"channel delta mismatch")
    return by


def render():
    source = study.read(DEV/"comparison.json")
    evidence = {a:study.read(HERE/(a+".json.gz"))["games"] for a in ARMS}
    corroborate(evidence)
    by = {a:{(g["seed"],g["seat"]):g for g in games} for a,games in evidence.items()}
    pairs = []
    for row in source["paired_map_seats"]:
        key = row["seed"],row["seat"]
        for a in ARMS:
            study.check(row[a]["raw_sha256"]==by[a][key]["raw_sha256"] and
                        row[a]["margin"]==by[a][key]["official_margin"],"official comparison differs")
        delta = {}
        for ref in ("parent","control"):
            c, b = gaps(by["candidate"][key]), gaps(by[ref][key])
            delta[ref] = {k:c[k]-b[k] for k in c}
            study.check(abs(sum(delta[ref].values()) - (row["candidate"]["margin"]-row[ref]["margin"]))<1e-5,
                        "unreconciled change")
        pairs.append(dict(seed=key[0],seat=key[1],arm_official_margins={a:row[a]["margin"] for a in ARMS},
                          candidate_net_changes=delta))
    summed = {}
    for arm,games in evidence.items():
        summed[arm] = {}
        for side in ("own","opponent"):
            stats = Counter()
            for g in games:
                physical = g["seat"] if side=="own" else 1-g["seat"]
                stats.update(g["physical_pressure"][physical])
            summed[arm][side] = dict(stats)
    cases = []
    for g in evidence["candidate"]:
        w = g["extreme_windows"]["loss_10"]
        a, b = w["pre"],w["post"]; s=g["seat"]
        events = {}
        for turn in g["turns"][a:b]:
            acc = turn["cash_gain_by_seat"]
            for k in set(acc[0])|set(acc[1]):
                events[k] = events.get(k,0)+acc[s].get(k,0)-acc[1-s].get(k,0)
        worst = min(events,key=events.get)
        payments = [(abs(e[2]),turn["pre"],e) for turn in g["turns"][a:b]
                    for e in turn["salient_official_payments"]
                    if e[0]==worst and (e[2] if e[1]==s else -e[2]) < 0]
        major = max(payments,key=lambda r:r[0]) if payments else None
        case = {"seed":g["seed"],"seat":s,"window":[a,b],"margin_delta":w["candidate_minus_enemy_cash_change"],
                "channels":events,"largest_adverse_channel":worst,"one_recorded_payment":major,"worker_trace":[]}
        if major:
            _,step,record = major
            wid = record[3]
            for i in (10,5,3,0):
                t = step-i
                if t<0:continue
                stamped = next((x for x in w["context"] if x["pre"]==t), None)
                if not stamped:continue
                assets = stamped["before"]["players"]
                for seat,x in enumerate(assets):
                    if wid in x[2]:
                        case["worker_trace"].append({"pre":t,"physical_seat":seat,"pos_cargo_role":x[2][wid],"order":stamped["orders"][seat].get(wid,"NONE")})
                        break
        cases.append(case)
    capsule = {"implementation_sha256":study.sha(__file_path__),
               "comparison_sha256":study.sha(DEV/"comparison.json"),"independent_cash_custody_sha256":study.sha(HERE/"crosscheck.json"),"source_artifacts_sha256":{a:study.sha(HERE/(a+".json.gz")) for a in ARMS},
               "pairs":pairs,"pressure_totals":summed,"candidate_largest_adverse_ten_turn_windows":cases}
    study.publish(HERE/"summary.json",capsule)
    lines=["# 三臂实战的回款、抢货与护树","",
           "两张相同开发地图，各换两席。每臂在原版 Master3s2 独立采样和搜索下完成四局。所有金额均按物理席重放官方 Board，逐步对上两席现金，并单列终局奖分与账面现金的差额。"
           "[完整原版赛果及逐图差异](../comparison.json) · [收支和实战行为汇总](summary.json) · [独立模拟器逐步复核现金与每次碰撞去向](crosscheck.json)","",
           "| 同图同席 | 候选 − 同训控制，官方分差变化 | 回库净差变化 | 树息净差变化 | 新种及夺树支出净差变化 | 招工净差变化 | 其余及终局调整 |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for r in pairs:
        d=r["candidate_net_changes"]["control"]
        nums=[sum(d.values()),d["deposits"],d["trees"],d["canopy_cost"],d["hires"],d["other"]+d["official_settlement"]]
        lines.append(f"| {r['seed']} / {r['seat']} | "+" | ".join(f"{v:+,.2f}" for v in nums)+" |")
    lines.extend(["","分渠道变化包含我方和考官两边各自得失。三臂各场面对能独立回应和采样的考官，这些金额只能定位当场差别。","","## 抢货和防守机会","",
                  "表中各格汇总四局。一名有货采集工在一步内可能与较轻的敌采集工同格，计一个暴露工回合；紧邻本方基地且同一个基地能被对手一步争夺，才计临门窗口。守树窗口要求对方种树工在可继续持有的己树上，行动前对手能付基本夺树价。货被抢记官方碰撞中直接转往对方采集工的量，另有掉地、入树、入基地等结算。","",
                  "| 臂 / 持货方 | 暴露工回合 / 当步货被敌采集工拿到 | 所有敌方直接缴货量 | 临门窗口 / 对手到该基地 / 实收笔数 | 树可被夺 / 防守采集工可到 / 在树上击杀 / 实际失树 |",
                  "|---|---:|---:|---:|---:|"])
    names={"parent":"父代","control":"同训控制","candidate":"经济辅助"}
    for arm in ARMS:
        for side in ("own","opponent"):
            d=Counter(summed[arm][side])
            groups=[("carrier_turns_exposed","carrier_turns_captured"),("immediate_home_windows","immediate_home_contested","immediate_home_banked"),
                    ("tree_seizure_windows","tree_windows_with_reachable_guard","tree_windows_guard_kill","tree_windows_seized")]
            shown=[" / ".join(str(d[k]) for k in group) for group in groups]
            lines.append(f"| {names[arm]} / {'我方' if side=='own' else '考官'} | {shown[0]} | {d['cargo_captured_by_enemy_collector']:,.2f} | {shown[1]} | {shown[2]} |")
    lines.extend(["","同一工人连续五步面临威胁会贡献五次机会。对手可能原本就会绕路，被威胁后没交货的步数无法直接折成人为封掉的现金。","",
                  "## 对手突然拉开差距时","","每局固定取我方现金差下降最多的连续十步，含两席实际动作、伤亡、回款、树息及此前十步的完整行动和双方各自可见的碳值；同时逐局归档最大单步下跌和对应上涨对照。","",
                  "| 图 / 席 | 前态 → 后态 | 十步净差变化 | 最大不利渠道及净额 |","|---|---:|---:|---:|"])
    labels={"own_delivery":"己方工人回己库","tree_absorption":"树息","plant":"新种支出","seize":"夺树支出","hire_collector":"招采集工",
            "hire_planter":"招种树工","opponent_worker_delivery_to_this_base":"对方交入己库","tree_collision_cash":"碰撞入树现金","base_collision_cash":"碰撞入库现金","rounding":"结算舍入"}
    for c in cases:
        k=c["largest_adverse_channel"]
        lines.append(f"| {c['seed']} / {c['seat']} | {c['window'][0]} → {c['window'][1]} | {c['margin_delta']:+,.2f} | {labels[k]} {c['channels'][k]:+,.2f} |")
    detailed = next((c for c in sorted(cases,key=lambda x:x["margin_delta"]) if len(c["worker_trace"])==4),None)
    if detailed:
        _,t,e=detailed["one_recorded_payment"]
        path="，".join(f"t{x['pre']} 在 {x['pos_cargo_role'][0]} 格持货 {x['pos_cargo_role'][1]:.1f}，执行 {x['order']}" for x in detailed["worker_trace"])
        lines.extend(["",f"图 {detailed['seed']} 席 {detailed['seat']} 的一笔实付涉及 {e[3]}。{path}。t{t} 推进时，物理席 {e[1]} 的 {labels[e[0]]} 记账 {e[2]:+,.2f}。提前改走其他路线后的终局收益需要独立续局验证。"])
    games=evidence["candidate"]
    changed_steps=sum(g["sapling_contribution"]["changed_steps"] for g in games)
    roles=Counter();total=sum(g["steps"] for g in games)
    for g in games: roles.update(g["sapling_contribution"]["changed_orders_by_role"])
    lines.extend(["","## 学生在实战发令中的贡献","",
                  f"候选四局 {total:,} 次真实公开观察均重现记录中的联合己方动作。保持同一个行动前局面、合法前缀解码和返航保护，移去占比 25% 的 Sapling 策略分数后，{changed_steps} 个回合出现不同发令，其中采集工 {roles['collector']} 条、种树工 {roles['planter']} 条、基地 {roles['base']} 条。这个消融只确定输出贡献，不能据此推算胜率。","",
                  "[父代逐步与局面证据](parent.json.gz) · [同训控制](control.json.gz) · [候选及学生消融](candidate.json.gz) · [官方重放和机会定义](analyze.py)"])
    study.publish(HERE/"README.md","\n".join(lines)+"\n")


def main(wait):
    with (HERE/".analysis.lock").open("a") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return
        for arm in ARMS:
            while wait and not (DEV/arm/"official_replay.json").exists():
                time.sleep(10)
            if not (DEV/arm/"official_replay.json").exists():
                return
            out=HERE/(arm+".json.gz")
            if out.exists():
                saved=study.read(out)
                study.check(saved["implementation_sha256"]==study.sha(study.Path(study.__file__)) and saved["official_replay_sha256"]==study.sha(DEV/arm/"official_replay.json"),
                            "stale economic analysis")
            else:
                study.process(arm)
            print(arm,"economic and pressure attribution complete",flush=True)
        while wait and not (DEV/"comparison.json").exists():
            time.sleep(5)
        if (DEV/"comparison.json").exists(): render()


__file_path__ = study.Path(__file__).resolve()
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--wait",action="store_true")
    main(ap.parse_args().wait)
