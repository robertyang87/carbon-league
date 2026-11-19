"""Follow each actual positive capture until the first settlement of the collector that received it."""
from collections import Counter
import argparse
import fcntl
import time
from pathlib import Path

import analyze as study


def custody(g):
    # The engine resolves worker collisions before bank deposits at each joint step.
    open_lots = {}
    finalized = []
    banks = {}
    for row in g["turns"]:
        t = row["pre"]
        for d in row["deaths"]:
            uid = d["worker"]
            sink,receiver,wid = d["transfer_to"]
            label = "carrier_died_"+d["reason"]
            for lot in open_lots.pop(uid,[]):
                lot.update(exit=label,exit_pre=t,exit_cell=d["cell"],death_sink=[sink,receiver,wid])
                finalized.append(lot)
            if sink=="worker" and d["cargo"]>0 and receiver==1-d["owner"]:
                open_lots.setdefault(wid,[]).append({"initial_pre":t,"origin_worker":uid,"carrier":wid,
                                                     "beneficiary":receiver,"cargo_at_capture":d["cargo"],"cell":d["cell"]})
        for kind,owner,amount,wid,cell,_ in row["salient_official_payments"]:
            if kind not in ("own_delivery","opponent_worker_delivery_to_this_base") or amount<=0:
                continue
            lots = open_lots.pop(wid,[])
            if not lots:
                continue
            key=(t,wid)
            study.check(key not in banks,"duplicate bank settlement")
            banks[key]={"pre":t,"worker":wid,"paid_to":owner,"actual_receipt":amount,"cell":cell,
                        "related_captures":len(lots),"kind":kind}
            for lot in lots:
                label = "first_own_bank" if owner==lot["beneficiary"] else "first_opponent_bank"
                lot.update(exit=label,exit_pre=t,exit_cell=cell,bank_key=list(key))
                finalized.append(lot)
    for lots in open_lots.values():
        for lot in lots:
            lot.update(exit="no_bank_before_terminal",exit_pre=len(g["turns"]))
            finalized.append(lot)
    return {"seed":g["seed"],"seat":g["seat"],"source_raw_sha256":g["raw_sha256"],
            "events":finalized,"unique_bank_receipts":list(banks.values())}


def summarize(games,own):
    rows=[]
    for g in games:
        owner = g["seat"] if own else 1-g["seat"]
        rows.extend((g,e) for e in g["events"] if e["beneficiary"]==owner)
    counts=Counter(e["exit"] for _,e in rows)
    cargo=Counter()
    for _,e in rows:
        cargo[e["exit"]]+=e["cargo_at_capture"]
    delays=sorted(e["exit_pre"]-e["initial_pre"] for _,e in rows if e["exit"]=="first_own_bank")
    bank_keys={(g["seed"],g["seat"],*e["bank_key"]) for g,e in rows if e["exit"]=="first_own_bank"}
    amounts=[]
    for g in games:
        amounts.extend(b["actual_receipt"] for b in g["unique_bank_receipts"]
                       if (g["seed"],g["seat"],b["pre"],b["worker"]) in bank_keys)
    study.check(len(bank_keys)==len(amounts),"nonunique settlement")
    return {"events":len(rows),"initial_captured":sum(cargo.values()),"counts":dict(counts),"initial_cargo_by_outcome":dict(cargo),
            "first_own_bank_delay_steps":delays,"unique_actual_own_receipts":len(amounts),"cash_on_those_receipts":sum(amounts)}


def guard_table(games, own):
    counts={"reachable":Counter(),"out_of_reach":Counter()}
    for g in games:
        physical=g["seat"] if own else 1-g["seat"]
        for row in g["turns"]:
            for h in row["tree_exposures"][physical]:
                group="reachable" if h["possible_guard"] else "out_of_reach"
                c=counts[group]
                c["windows"]+=1
                c["guard_arrived"]+=bool(h["actual_guard_at_tree"])
                c["invader_stayed"]+=h["raider_stayed"]
                c["invader_left"]+=not h["raider_stayed"]
                c["tree_seized"]+=h["seized"]
                c["invader_killed_on_tree"]+=h["invader_killed_by_own_collector"]
    return {k:dict(v) for k,v in counts.items()}


def main():
    games={a:study.read(study.HERE/(a+".json.gz"))["games"] for a in study.ARMS}
    records={a:[custody(g) for g in cohort] for a,cohort in games.items()}
    guards={a:{side:guard_table(cohort,side=="own") for side in ("own","opponent")} for a,cohort in games.items()}
    totals={a:{k:summarize(games,k=="own") for k in ("own","opponent")} for a,games in records.items()}
    data={"implementation_sha256":study.sha(Path(__file__)),"game_analysis_sha256":{a:study.sha(study.HERE/(a+".json.gz")) for a in records},
          "summaries":totals,"guard_reach":guards,"games":records}
    study.publish(study.HERE/"capture_settlement.json",data)
    lines=["# 缴来的货有多少真的走到本方基地","",
           "跟踪经济训练三臂 12 局中的每笔实际缴获。仅在官方碰撞把正货量从一方工人转给另一方采集工时开始计一笔，跟到该收货工第一次交库、死亡或终局。一次交库可对应多次缴获，实际到账只按唯一的局、工人和步号计一次。",
           "","| 臂 / 抢货方 | 真实缴获笔数及货量 | 收货工先交本方库，笔数及起始货量 | 相关唯一回款，笔数和混合总额 | 到库间隔大于十步的缴获笔数 |",
           "|---|---:|---:|---:|---:|"]
    for arm in study.ARMS:
        for role in ("own","opponent"):
            r=totals[arm][role]
            lines.append(f"| { {'parent':'父代','control':'同训控制','candidate':'经济辅助'}[arm]} / "
                         f"{'我方' if role=='own' else '考官'} | {r['events']} / {r['initial_captured']:,.2f} | "
                         f"{r['counts'].get('first_own_bank',0)} / {r['initial_cargo_by_outcome'].get('first_own_bank',0):,.2f} | "
                         f"{r['unique_actual_own_receipts']} / {r['cash_on_those_receipts']:,.2f} | "
                         f"{sum(t>10 for t in r['first_own_bank_delay_steps'])} |")
    lines.extend(["","货物会与原有载荷、途中采集和其他缴获混合，也可能在路上被对手树吸走。表中“起始货量”按碰撞瞬间计，“混合总额”按实际充值计，两者不能相加或当成抢这一步带来的新增现金。收货工再次被打掉后在本表截止，货转入哪一方、哪一格留在逐笔数据里。提前打断对手其他回款的价值单凭这些回放无法识别。",
                  "","[每笔双方伤亡、承运工、收款方与实际到账](capture_settlement.json) · [原始现金及机会核验](README.md)"])
    lines.extend(["","## 对手已经站上己方树的时机","","分母为种树工在树格、有足够行动前现金、树未到寿的工树回合。比较一步可到与不可到的采集工；同一工人连续威胁同树可再次计入。","",
                  "| 臂 / 树主 | 一步可守 / 守方抵达 / 来者撤离 / 失树 | 无人一步可守 / 来者撤离 / 失树 |",
                  "|---|---:|---:|"])
    for arm in study.ARMS:
        for side in ("own","opponent"):
            h=guards[arm][side]
            good,bad=Counter(h["reachable"]),Counter(h["out_of_reach"])
            nums=[good["windows"],good["guard_arrived"],good["invader_left"],good["tree_seized"]]
            low=[bad["windows"],bad["invader_left"],bad["tree_seized"]]
            title={"parent":"父代","control":"同训控制","candidate":"经济辅助"}[arm]
            lines.append(f"| {title} / {'我方' if side=='own' else '考官'} | "+
                         " / ".join(map(str,nums))+" | "+" / ".join(map(str,low))+" |")
    lines.extend(["","当两种局面实际失树率差异大时，下一次可在对手种树工抵达树格前一回合，比较守方采集工继续经营与调位防守的真实终局差。来者是否撤走与守方选择在同一时间发生，现有回放没有给出改变守方一步之后对手会怎么走。"])
    study.publish(study.HERE/"CAPTURE_SETTLEMENT.md","\n".join(lines)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait",action="store_true")
    args=parser.parse_args()
    with (study.HERE/".settlement.lock").open("a") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(0)
        while args.wait and not (study.HERE/"candidate.json.gz").exists():
            time.sleep(10)
        main()
