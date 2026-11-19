"""Settle the three packages on each new map and partition the margin at t128 and t160."""
import argparse
from collections import Counter
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile

HERE = Path(__file__).resolve().parent
LANE = HERE.parent
PRIOR = HERE.parents[2] / "fresh_original/economy"
sys.path.insert(0, str(PRIOR))
import settle as S

ARMS = ("parent", "student", "window")
PHASES = ("before", "window", "after")


def snapshot(sim, seat):
    # Only candidates' public team records and public terrain; opening physical hashes are separately verified.
    players = json.loads(json.dumps(sim.obs(seat)["players"]))
    return dict(t=sim.step_count, cash=list(sim.cash), margin=sim.cash[seat]-sim.cash[1-seat],
        public_players=players, scoreboard_by_seat=[
            dict(cash=sim.cash[s], cargo=sum(w[1] for w in p[2].values()),
                 collectors=sum(w[2]=="COLLECTOR" for w in p[2].values()),
                 planters=sum(w[2]=="PLANTER" for w in p[2].values()), trees=len(p[3])) for s,p in enumerate(players)])


def phases(directory, g, whole):
    receipt=S.read(directory/"receipt.json")
    p=directory/"turns.jsonl"
    tape=[json.loads(line) for line in p.read_bytes().splitlines()] if p.exists() else []
    _,cfg=S.live.official()
    sim=S.FastCarbon(cfg).reset(seed=g["seed"])
    sim.cash=S.CashEvents(sim.cash)
    s=g["seat"]
    checkpoints={"0":snapshot(sim,s)}
    ledger={phase:[Counter(),Counter()] for phase in PHASES}
    counts=Counter()
    orders=[]
    for row in tape:
        t=row["step"]
        assert t==sim.step_count
        if any(c["status"]!="ok" for c in row["calls"]):
            break
        joint=[c["response"]["action"] for c in row["calls"]]
        if t in (128,160):
            checkpoints[str(t)]=snapshot(sim,s)
        name="before" if t<128 else "window" if t<160 else "after"
        counts[name]+=1
        sim.cash.events.clear()
        sim.step(*joint)
        orders.append(joint)
        for kind,side,amount,*fields in sim.cash.events:
            slot=ledger[name][side]
            if kind=="tree_absorption":
                slot["ground_rent"]+=amount-fields[2]
                slot["cargo_tax"]+=fields[2]
            else:
                slot[kind]+=amount
    assert len(orders)==g["official_transitions"] and list(sim.cash)==whole["physical_end"]
    checkpoints["end"]=snapshot(sim,s)
    stages={ph:S.balance(v,s) for ph,v in ledger.items()}
    for k in S.MONEY:
        assert abs(sum(v[k] for v in stages.values())-whole["margin_components"]["whole"][k])<1e-6
    for start,end,name in (("0","128","before"),("128","160","window"),("160","end","after")):
        if start in checkpoints and end in checkpoints:
            assert abs(sum(stages[name].values())-(checkpoints[end]["margin"]-checkpoints[start]["margin"]))<1e-6
    return dict(checkpoints=checkpoints,steps=counts,physical_seat_ledger=ledger,margin_components=stages,
                economic_margin_changes={k:sum(v.values()) for k,v in stages.items()}),orders


def divergence(tape1,tape2,seat):
    first={"own":None,"original":None,"joint":None}
    same_until_128=0
    for t,(a,b) in enumerate(zip(tape1,tape2)):
        own=a[seat]!=b[seat];original=a[1-seat]!=b[1-seat]
        if first["joint"] is None and t<128 and not own and not original:
            same_until_128+=1
        for k,changed in (("own",own),("original",original),("joint",own or original)):
            if first[k] is None and changed:
                first[k]=t
    return dict(first_different_public_commands=first,identical_joint_prefix_steps_before_window=same_until_128,
        both_joint_orders_at_first_difference=([tape1[first["joint"]],tape2[first["joint"]]] if first["joint"] is not None else []),
        same_decisions_comparable_only_until_first_joint_difference=True)


def compare(root):
    planfile=LANE/"protocol.json"
    plan=S.read(planfile)
    games={};tapes={}
    with tempfile.TemporaryDirectory() as tmp:
        for reg in S.unpack_archives(root,Path(tmp)):
            r=S.read(reg)
            if r.get("matrix_sha256")!=S.live.sha(planfile):
                continue
            seed,s,arm=r["seed"],r["candidate_seat"],r["arm"]
            key=(seed,s,arm)
            assert seed in plan["seeds"] and arm in ARMS and s in (0,1)
            if key in games:
                assert S.live.sha(reg.parent/"receipt.json")==games[key]["game"]["receipt_sha256"]
                continue
            g=S.verify_game(reg.parent,seed,s,arm,planfile)
            group=S.settle_game(g,reg.parent)
            segmented,orders=phases(reg.parent,g,group)
            group["window_settlement"]=segmented
            group["workflow_run"]=r["workflow_run"]
            games[key],tapes[key]=group,orders
    assert set(games)==set(itertools.product(plan["seeds"],(0,1),ARMS))
    comparisons=[]
    for seed,s in itertools.product(plan["seeds"],(0,1)):
        triple=[games[seed,s,n] for n in ARMS]
        assert len({x["workflow_run"] for x in triple})==1 and len({x["game"]["image_id"] for x in triple})==1
        for baseline in ("parent","student"):
            w=games[seed,s,"window"];b=games[seed,s,baseline]
            pieces={k:w["window_settlement"]["economic_margin_changes"][k]-b["window_settlement"]["economic_margin_changes"][k] for k in PHASES}
            terminal=(w["official_margin"]-b["official_margin"] if all(z["official_margin"] is not None for z in (w,b)) else None)
            C=[z["window_settlement"]["checkpoints"] for z in (w,b)]
            root_delta=C[0]["128"]["margin"]-C[1]["128"]["margin"] if all("128" in k for k in C) else None
            if root_delta is not None:assert math.isclose(root_delta,pieces["before"],abs_tol=1e-6)
            comparisons.append(dict(seed=seed,seat=s,baseline=baseline,terminal_diff=terminal,
                difference_at_t128=root_delta,component_differences=pieces,
                official_adjustment_diff=terminal-sum(pieces.values()) if terminal is not None else None,
                prefix=divergence(tapes[seed,s,"window"],tapes[seed,s,baseline],s)))
    return dict(protocol=S.live.sha(planfile),analysis_sha256=S.live.sha(Path(__file__)),
                reused_settlement_sha256=S.live.sha(Path(S.__file__)),games=[games[k] for k in sorted(games)],comparisons=comparisons)


def d(x):
    return "—" if x is None else f"{x:+,.2f}"


def score(z):
    if not z["scored"]:return "未计"
    return ("平" if z["status"]=="complete" else "双方故障") if z["winner"] is None else ("胜" if z["winner"]==z["game"]["seat"] else "负")


def report(out):
    names={"parent":"父代","student":"全程 H24","window":"窗口 H24"}
    rows=["# 神经权重干预窗口的独立整局结算","","两张预登记开发图，各自双席三臂。对手采用原版 Master3s2 公开入口，双方独立冷启、整局自然采样。窗口臂在 t128–159 使用训练权重，接手前和 t160 起运行父代权重。","",
          "## 全批结果","","|版本|席 0 胜/平/负|席 1 胜/平/负|全场胜/平/负|平均官方分差|","|---|---|---|---|---:|"]
    def wdl(zs):
        c=Counter(score(z) for z in zs);v='/'.join(str(c[k]) for k in ("胜","平","负"));rest=len(zs)-sum(c[k] for k in ("胜","平","负"))
        return v if not rest else v+f"（另有{rest}场，见逐局状态）"
    for n in ARMS:
        games=[r for r in out["games"] if r["game"]["arm"]==n]
        scores=[r["official_margin"] for r in games]
        rows.append(f"|{names[n]}|{wdl([z for z in games if z['game']['seat']==0])}|{wdl([z for z in games if z['game']['seat']==1])}|{wdl(games)}|{d(statistics.mean(scores) if all(x is not None for x in scores) else None)}|")
    rows+=["","|地图|席|版本|状态|己方 / 考官官方分|t128 接手时净现金|窗口 32 步净变动|交还父代后净变动|终局分差|","|---|---:|---|---|---|---:|---:|---:|---:|"]
    for z in out["games"]:
        g=z["game"];s=g["seat"];v=z["window_settlement"];p=v["economic_margin_changes"];scores=g["official_scores_by_seat"]
        pair="—" if scores is None else f"{scores[s]:,.2f} / {scores[1-s]:,.2f}"
        rows.append(f"|{g['seed']}|{s}|{names[g['arm']]}|{score(z)} · {z['status']} · {g['official_transitions']}步|{pair}|{d(v['checkpoints'].get('128',{}).get('margin'))}|{d(p['window'])}|{d(p['after'])}|{d(z['official_margin'])}|")
    rows+=["","## 接手局面与 32 步账目","","接手时的携货和存活树数都取己方减考官。窗口内交付、地租和货税按各自实际入账结算；栽夺和招工为带符号的双方支出净变动。","",
        "|地图|席|版本|t128 携货差 / 树数差|窗口交付净差|窗口地租 / 货税净差|窗口栽夺 / 招工净差|","|---|---:|---|---|---:|---|---|"]
    for z in out["games"]:
        g=z["game"];s=g["seat"];v=z["window_settlement"];p=v["margin_components"]["window"]
        start=v["checkpoints"].get("128");gap="—"
        if start:
            a=start["scoreboard_by_seat"][s];b=start["scoreboard_by_seat"][1-s]
            gap=f"{d(a['cargo']-b['cargo'])} / {a['trees']-b['trees']:+d}"
        rows.append(f"|{g['seed']}|{s}|{names[g['arm']]}|{gap}|{d(p['own_delivery'])}|{d(p['ground_rent'])} / {d(p['cargo_tax'])}|{d(p['plant']+p['seize'])} / {d(p['hire_collector']+p['hire_planter'])}|")
    rows+=["","## 同图同席的窗口臂差值","","每一项均用窗口臂减同图同席的比较臂。接手前的累计差、窗口内新增差、归还后新增差按物理现金分解。每局的官方灭绝计分调整另存。","",
           "|地图|席|比较臂|接手时已有差|32 步新增差|交还后新增差|官方终局差|首个不同己令 / 敌令 t|","|---|---:|---|---:|---:|---:|---:|---|"]
    for x in out["comparisons"]:
        p=x["component_differences"];f=x["prefix"]["first_different_public_commands"]
        rows.append(f"|{x['seed']}|{x['seat']}|{names[x['baseline']]}|{d(x['difference_at_t128'])}|{d(p['window'])}|{d(p['after'])}|{d(x['terminal_diff'])}|{f['own']} / {f['original']}|")
    rows+=["","|比较臂|地图|席位平均终局差|接手时已有差|窗口新增差|交还后新增差|","|---|---|---:|---:|---:|---:|"]
    for base in ("parent","student"):
        all_comp=[x for x in out["comparisons"] if x["baseline"]==base]
        for seed in sorted({x["seed"] for x in all_comp})+[None]:
            xs=[x for x in all_comp if seed is None or x["seed"]==seed]
            def avg(items):
                return d(statistics.mean(items)) if all(x is not None for x in items) else "—"
            vals=[avg([x["terminal_diff"] for x in xs]),avg([x["difference_at_t128"] for x in xs])]
            vals.extend(avg([x["component_differences"][k] for x in xs]) for k in ("window","after"))
            rows.append(f"|{names[base]}|{seed or '两图等权'}|{'|'.join(vals)}|")
    rows+=["","同图同席提供相同开局。考官每场重新自然采样，命令序列可能从开局起就分叉。接手时的现金、携货、存活工人、树产和地图位置因此可能不同。三段差值是观测轨迹的会计分解，无法由此把整局差或某一段的差全部归因于权重的 32 次调用。","",
        "## 外部计时与原版搜索","","|地图|席|版本|本方 P95 / P99 / 最慢 ms|考官 P95 / P99 / 最慢 ms|考官实搜 / 单候选 / 零深|双方慢步|失败请求|","|---|---:|---|---|---|---|---|---|"]
    for z in out["games"]:
        g=z["game"];s=g["seat"];t=z["ladder_clock_by_seat"];q=z["original_search"]["branches"]
        tt=[' / '.join(f"{1000*t[i][k]:.0f}" if k in t[i] else '—' for k in ("p95","p99","max")) for i in (s,1-s)]
        n=" / ".join(str(q.get(k,0)) for k in ("searched","single","zero"))
        rows.append(f"|{g['seed']}|{s}|{names[g['arm']]}|{tt[0]}|{tt[1]}|{n}|{t[s].get('slow',0)} / {t[1-s].get('slow',0)}|{sum(a.get('errors',0)+a.get('timeouts',0) for a in t)}|")
    rows+=["","[逐局现金、接手局面、逐拍搜索与计时](settlement.json.gz) · [矩阵注册](../protocol.json)"]
    (HERE/"README.md").write_text('\n'.join(rows)+'\n')


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("artifacts",type=Path)
    result=compare(parser.parse_args().artifacts)
    S.store(HERE/"settlement.json.gz",result)
    report(result)
