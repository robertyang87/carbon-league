"""Render all eight same-state policy edits and the observed continuations."""
from collections import Counter
from pathlib import Path

from replay import DEV,HERE,check,dump,read,sha


def budget():
    root=DEV.parents[1]/"early_lane_dev/native_response"
    rates=[]; sources={}
    for case in sorted(root.glob("case_*")):
        for arm in ("parent","alternative"):
            p=case/("live_"+arm+".json")
            row=read(p)
            rates.append((row["elapsed"],row["search_calls"]))
            sources[str(p.relative_to(DEV.parents[1]))]=sha(p)
        p=case/"live_shared_init.json"
        sources[str(p.relative_to(DEV.parents[1]))]=sha(p)
    return {"reference_native_arms":len(rates),"elapsed_seconds":sum(r[0] for r in rates),
            "reactive_search_calls":sum(r[1] for r in rates),"weighted_seconds_per_search":sum(r[0] for r in rates)/sum(r[1] for r in rates),
            "range_seconds_per_search":[min(r[0]/r[1] for r in rates),max(r[0]/r[1] for r in rates)],"refs":sources}


def resource_delta(root,seat):
    d=root["candidate_minus_control"]
    resource=[]
    for row in d["workers"]:
        c,n=row["control"],row["candidate"]
        if c and n and abs(c[1]-n[1])<1e-7:
            continue
        resource.append({"type":"worker","unit":row["id"],"seat":row["physical_seat"],
                         "control":c,"candidate":n,"delta_cargo":(n[1] if n else 0)-(c[1] if c else 0)})
    for row in d["trees"]:
        resource.append({"type":"tree",**row})
    return {"cash_by_seat":d["cash_by_physical_seat"],"delta_margin":d["cash_by_physical_seat"][seat]-d["cash_by_physical_seat"][1-seat],
            "resource_changes":resource}


def fmt_state(v):
    return "缺席" if v is None else "/".join(map(str,v))


def main():
    bundle=read(HERE/"one_turn.json.gz")
    check(bundle["implementation_sha256"]==sha(HERE/"replay.py"),"script differs from evidence")
    rates=budget()
    lines=["# 候选与同训控制在相同局面下的八次改令","",
           "在收入辅助候选实际走出的 1,196 个公开局面中，候选与同训控制有八次发令不同，共十条单位命令。用两份冻结比赛包从开局接收同一席的公开历史，各自重算完整联合动作；逐槽验证合法性、单位顺序以及第一次改令前的前缀特征。该步对手指令取候选原局已同时发出的指令。两份联合动作各由官方 Board 从相同盘面、相同工树 ID 计数器推进一次。",
           "","| 地图 / 席 / 步 | 工人角色与命令，控制 → 候选 | 当步现金差改变量 | 实物变化，候选相对控制 | 候选原局继续 3 / 5 / 10 步的现金差变动 |",
           "|---|---|---:|---|---:|"]
    summary=[]
    for g in bundle["games"]:
        s=g["seat"]
        for root in g["edit_roots"]:
            edits=root["edited"]
            effects=resource_delta(root,s)
            desc=[]
            for e in edits:
                p,cargo,_=e["before"];role="采集" if e["role"]=="COLLECTOR" else "种树"
                suffix="，前序已分岔" if not e["same_preceding_orders"] else ""
                desc.append(f"{role} {e['unit'].rsplit('-',1)[-1]}，{p}格／{cargo:.1f}货：{e['control']} → {e['candidate']}{suffix}")
            goods=[]
            for r in effects["resource_changes"]:
                side="我" if r.get("seat",r.get("physical_seat"))==s else "敌"
                if r["type"]=="tree":
                    d="新增持有" if r["control"] is None else "失去持有" if r["candidate"] is None else "树位/树龄变化"
                    goods.append(f"{side}{d}树 {r['id']}")
                else:
                    v=r["candidate"]
                    if r["control"] is None or v is None:
                        goods.append(f"{side}{r['unit'].rsplit('-',1)[-1]} {'仅候选存活' if v else '仅控制存活'}")
                    if abs(r["delta_cargo"])>1.e-7:goods.append(f"{side}{r['unit'].rsplit('-',1)[-1]} 持货 {r['delta_cargo']:+.2f}")
            future=root["recorded_candidate_trajectory_only"]
            offsets=[f"{future[str(h)]['cash_difference_change']:+.2f}" if future[str(h)]["available"] else "已终局" for h in (3,5,10)]
            post=root["outcomes"]["candidate"]["post"];alt=root["outcomes"]["control"]["post"]
            terminal_exact=False
            if root["pre"]==298 and all(p["workers"] or p["trees"] for p in post+alt):
                candidate_final=post[s]["cash"]-post[1-s]["cash"]; other_final=alt[s]["cash"]-alt[1-s]["cash"]
                check(abs(candidate_final-g["recorded_margin"])<1e-6,"official horizon score not reproduced")
                terminal_exact=True
                goods.append(f"正常步限，两种动作的精确终分差 {candidate_final:+.2f} / {other_final:+.2f}（候选 / 控制）")
            lines.append(f"| {g['seed']} / {s} / {root['pre']} | "+"；".join(desc)+f" | {effects['delta_margin']:+.2f} | "+
                         ("；".join(goods) if goods else "两分支仅移动位置或局面资源分布有别，工树与货量相同")+" | "+" / ".join(offsets)+" |")
            summary.append({"seed":g["seed"],"seat":s,"pre":root["pre"],"edits":edits,
                            "exact_post_effect":effects,"normal_terminal_either_action":terminal_exact,
                            "altered_root_one_future_candidate_3_5_10":offsets})
    lines.extend(["","净现金差按本方现金减对手现金。工人货物属于库存，须交入基地等结算才会入账。两条命令同时改变的根只估整份联合动作之差；该步第一次分岔之后，后续工人的实际合法前缀也随之改变。末列始终是候选真实轨迹和它自己行动前现金差的变化。控制替代分支只推进了当拍。","",
                  "## 两臂原生对局何时分开","","| 图 / 席 | 首个不同的联合动作步 | 当步我方改变 | 当步考官改变 | 候选历史上首个网络改令步 |","|---|---:|---|---|---:|"])
    for g in bundle["games"]:
        d=g["first_native_joint_divergence"]["control"]; first=g["edit_roots"][0]["pre"]
        lines.append(f"| {g['seed']} / {g['seat']} | {d['pre']} | {'是' if d['own_changed'] else '否'} | {'是' if d['opponent_changed'] else '否'} | {first} |")
    lines+=["","四对同图同席的原生比赛都由考官先改变行动。当己方网络开始给出不同命令时，考官采样已经让两盘走入了不同局面。因此这四局的主尺差同时包含考官早期采样差异和后续策略互动。","",
            f"在先前八个原版续局臂上，共 {rates['reactive_search_calls']} 次后续搜索、耗时 {rates['elapsed_seconds']:.2f} 秒，按调用数加权 {rates['weighted_seconds_per_search']:.3f} 秒/次。若比较某一步 t 的两个根动作后都改用同一续弈策略，未提前灭绝时两个分支至步限合计还需约 2 × (298−t) 次原版搜索。预算须另计复原公开历史、重放和核验。","",
            "[八根全联合动作、各槽前缀、现金科目及两种结果态](one_turn.json.gz) · [原始三臂结果](../comparison.json)"]
    # Check the largest immediate inventory trade against observed receipts, without projecting the replacement branch.
    target=next(r for r in summary if r["seed"]==621497120 and r["seat"]==1 and r["pre"]==288)
    check(all(abs(v)<1e-8 for r in summary for v in r["exact_post_effect"]["cash_by_seat"]),"observed cash pattern changed")
    assets=target["exact_post_effect"]["resource_changes"]
    lost=sum(v["delta_cargo"] for v in assets if v["type"]=="worker" and v["seat"]==1)
    original=read(DEV/"game_analysis/candidate.json.gz")
    source=next(g for g in original["games"] if g["seed"]==target["seed"] and g["seat"]==target["seat"])
    uids={d["unit"] for d in target["edits"]}
    deposits=[(turn["pre"],e[3],e[2]) for turn in source["turns"][288:298] for e in turn["salient_official_payments"]
              if e[0]=="own_delivery" and e[3] in uids]
    check({u for _,u,_ in deposits}==uids,"missing observed carrier deposit")
    deposit_text="、".join(f"t{t} 的 {u.rsplit('-',1)[-1]} 号 {v:.2f}" for t,u,v in deposits)
    near_end_calls=2*(298-288)
    lines.extend(["","## 是否追加原版续局","",
                  "八根当步两席现金均完全相同，也没有任何工人或树因换令而新增、死亡或易主。t298 两条路已经有精确相同的终局分差。"
                  "其余七根会改变位置，部分会改变下一帧货量，不能据此认定某步硬丢了大笔已到账资源。",
                  "",f"最值得低成本复核的是 t288：候选同时把两名采集工从停采改为移动，当步合计少持 {abs(lost):.2f} 货。"
                  f"候选原局随后实收 {deposit_text}。没有相同根状态下控制命令的完整续局，不能把这三笔到账全算成提早动身的功劳。"
                  f"若需要区分这种临终取舍，优先对此根做一对，两臂后续共 {near_end_calls} 次原版回应，参考耗时约 {near_end_calls*rates['weighted_seconds_per_search']:.1f} 秒搜索；"
                  "另加状态复原、逐步核验及排队，重复不同考官采样的预算须按重复数计。将这项列为低优先定位实验，保留现有原版训练工位。"])
    dump(HERE/"summary.json",{"source_sha256":sha(HERE/"one_turn.json.gz"),"report_code_sha256":sha(Path(__file__)),
                               "cost":rates,"actual_cash_evidence_sha256":sha(DEV/"game_analysis/candidate.json.gz"),"t288_actual_carrier_deposits":deposits,"roots":summary})
    dump(HERE/"README.md","\n".join(lines)+"\n")


if __name__=="__main__":
    main()
