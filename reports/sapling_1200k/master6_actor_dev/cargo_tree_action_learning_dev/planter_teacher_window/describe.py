"""Trace candidate and examiner behavior around the largest realized changes in the paired cash gap."""
from collections import Counter
from pathlib import Path
from run import ACTOR, HERE, PLAN, R, START, END, archive


def public_turns(raw, tape, side):
    sim = R.FastCarbon(raw["game_configuration"]).reset(seed=raw["seed"])
    out = {}
    for t, joint in enumerate(raw["actions"][:START] + tape):
        if t>=START-10:
            view=sim.obs(side)
            players=view["players"]
            out[t]=dict(t=t, orders=joint, seat_public_cash=[p[0] for p in players], workers=[p[2] for p in players],
                trees=[p[3] for p in players], visible_carbon_for_own_seat=view["carbon"])
        sim.step(*joint)
    return out


def tablevalue(v):
    return f"{v:+,.2f}" if abs(v)>1e-9 else "0.00"


def describe():
    summary=R.load(HERE/"results.json")
    history=[]
    text=["# 原版教师只接管本席种植工 16 步","",
          "两张已曝光训练地图从第 128 步各分成两条路径。对照全程用首胜包 8da4b465。教师臂在第 128–143 步由本席原版给种植工提令，其余工人和基地由 Sapling 混合策略按实际前缀重新解码。教师的信念在每一步写入最终实际走位。第 144 步起均交还同一父包，完整原版考官一直按本席观测重新响应到官方终局。","",
          "| 地图、己席 | 父代官方比分（我／敌） | 教师窗口官方比分（我／敌） | 相对终差 | 真改种植令步数／条数 | 中盘纯地租差变化 |",
          "|---|---:|---:|---:|---:|---:|"]
    notes=[]
    for c,row in zip(PLAN["cases"],summary["rows"]):
        s=c["seat"];raw=archive(c);data={k:R.load(HERE/f"{c['seed']}_{s}_{k}.json.gz") for k in ("parent","teacher")}
        ids={k:public_turns(raw,z["tape"],s) for k,z in data.items()}
        last_gap=0;deltas=[]
        for a,b in zip(data["parent"]["cash"]["steps"],data["teacher"]["cash"]["steps"]):
            t=a["t"];assert t==b["t"]
            gap=(b["cash"][s]-b["cash"][1-s])-(a["cash"][s]-a["cash"][1-s])
            delta=gap-last_gap;last_gap=gap
            effects=Counter()
            for sign,r in ((1,b),(-1,a)):
                for k,owner,val,*meta in r["events"]:
                    scale=sign*(1 if s==owner else -1)
                    if k=="tree_absorption":
                        effects["ground_rent"]+=scale*(val-meta[2]);effects["cargo_tax"]+=scale*meta[2]
                    else:effects[k]+=scale*val
            assert abs(sum(effects.values())-delta)<1e-5
            deltas.append(dict(t=t,delta=delta,paired_advantage_after=gap,events=dict(effects)))
        turns=sorted(deltas,key=lambda x:(-abs(x["delta"]),x["t"]))[:3]
        for turn in turns:
            t=turn["t"]
            turn["preceding"]={k:{str(d):[ids[k][h] for h in range(t-d,t+1)] for d in (3,5,10)}for k in ids}
        def first_difference(physical_seat):
            return next((START+i for i,(a,b) in enumerate(zip(data["parent"]["tape"],data["teacher"]["tape"]))
                         if a[physical_seat]!=b[physical_seat]),None)
        first_planter=next((r["t"] for r in data["teacher"]["decisions"] if r["changed_planters"]),None)
        first_ours,first_enemy=first_difference(s),first_difference(1-s)
        timing_note=""
        if first_enemy is not None and first_planter is not None and first_enemy<=first_planter:
            timing_note="同时行动和自适应搜索墙钟使这一敌令差异无法归于当拍种植令。"
        history.append(dict(seed=c["seed"],seat=s,turns=turns,first_real_planter_intervention=first_planter,
                            first_own_between_arm_difference=first_ours,first_rival_between_arm_difference=first_enemy))
        p=row["parent_scores"];q=row["teacher_scores"]
        d=row["cash_delta_by_horizon"]
        text.append(f"| {c['seed']}，席 {s} | {p[s]:,.2f} / {p[1-s]:,.2f} | {q[s]:,.2f} / {q[1-s]:,.2f} | {tablevalue(row['delta_margin'])} | {row['changed_planter_steps']} / {row['changed_planter_slots']} | {tablevalue(d['to199'].get('ground_rent',0))} |")
        notes.append(f"图 {c['seed']}：到终局，教师窗口相对父代改变的本方减对方常规入库为 {tablevalue(d['full'].get('own_delivery',0))}，地面树租 {tablevalue(d['full'].get('ground_rent',0))}，种植及夺树净支出对分差贡献 {tablevalue(sum(d['full'].get(k,0) for k in ('plant','seize')))}。最大单拍现金差变化为第 {turns[0]['t']} 步 {tablevalue(turns[0]['delta'])}。考官两臂实际耗时 {row['arms']['parent']['enemy_wall']:.2f}／{row['arms']['teacher']['enemy_wall']:.2f} 秒，分别记录 {len(data['parent']['enemy_calls'])}／{len(data['teacher']['enemy_calls'])} 次公开入口。教师臂第一次实际改变本方种植令在 t{first_planter}，双臂本方联令首次不同在 t{first_ours}，敌方联令首次不同在 t{first_enemy}。{timing_note}")
    text.extend(["", notes[0], "", notes[1], ""])
    text.append("登记的教师示范筛查结果为 **"+("通过" if summary["teacher_data_signal_met"] else "未通过")+"**。两张图各一组共同起始随机流；行为与搜索深度随轨迹及墙钟演化。这些是神经训练信号的条件对照，新图整场胜率和模型在正常开局中的能力仍须独立衡量。")
    text.extend(["","两个物理席位共验证 32 次父包逐步完全等价；另在 7 个原有局面逐项指定 35 种种植工合法选择，获 28 个实际变化。两个原图的全部档案命令重放到官方终局，现金分类账逐项核平。中盘地租只计第 128–199 步由土壤吸收的本方减对方到账额。每局最大三次现金转折逐笔附了双方行动、前后现金、工人、树和本席能见的碳，并分别回溯前 3、5、10 步。后续全队收益无法归到单一工人一次走位。","",
          "[事前协议](protocol.json) · [实令与护栏的独立检查](controlled_edits.json) · [逐图官方分、账项和胜负输入](results.json) · [现金转折与实际走位](retrospective.json.gz)"])
    R.write(HERE/"retrospective.json.gz",dict(analysis=R.sha(Path(__file__)),rows=history))
    (HERE/"README.md").write_text("\n".join(text)+"\n")


if __name__=="__main__":
    describe()
