"""Make the paired economic results readable while preserving physical seat and time."""
from collections import Counter
from statistics import mean

KEYS=("bank","donation","tree","collision_cash","recruit","plant","seize","round")


def fmt(v):
    return f"{0 if abs(v)<.00001 else v:,.2f}"


def reg(x):
    return "无" if x is None else f"t{x}"


def focus_trace(row,s,ids):
    by={w['uid']:w for w in row["by_seat"][s]["workers"]}
    result=[]
    for uid in ids[:2]:
        w=by.get(uid)
        result.append(f"{uid} 尚未出现或已消失" if w is None else
                      f"{uid} ({w['xy'][0]},{w['xy'][1]})，载 {fmt(w['cargo'])}，{w['move']}")
    return " / ".join(result) or "无在场工"


def mean_component(pairs,k,side):
    return mean(p["candidate"]["by_physical_seat"][p["seat"]^side][k]-
                p["parent"]["by_physical_seat"][p["seat"]^side][k] for p in pairs)


def write_report(data,output):
    pairs=data["pairs"]
    lines=["# 撤离候选的四组同图同席现金复盘","",
           "父代和候选分别在两张开发图、两个物理席与原版 Master3s2 打到正式终局。下表每一行都用已存双方命令重放并逐步核平现金，终分对官方收据。正常回库与树息分列；截获实收和敌货树税分别是相应现金收入的组成项，不能再次相加。[八局正式对照](../comparison.json) · [完整复盘与双方命令依据](games.json.gz)","",
           "| 图 / 我席 | 臂 / 账户 | 官方终分 | 正常回库 | 树收入（其中敌货税） | 抢入采集工 / 最终兑现 | 敌送基地 / 碰撞直冲现金 | 招工 / 种树 / 夺树费用 |",
           "|---|---|---:|---:|---:|---:|---:|---:|"]
    saves=sum(e["saved"] for p in pairs for e in p["candidate"]["edits"])
    max_turns=[p["candidate"]["largest_abs_cash_lead_change"] for p in pairs]
    heavy_steps=sum(w["components"][p["seat"]]["bank"]==0 and w["components"][1-p["seat"]]["bank"]>300
                    for p,w in zip(pairs,max_turns))
    wins=sum(p["candidate"]["terminal"][p["seat"]]>p["candidate"]["terminal"][1-p["seat"]] for p in pairs)
    mean_gap=mean(p["delta_margin"] for p in pairs)
    lines[3:3]=["",f"四盘候选实际获胜 {wins} 场，相对父代平均分差变化 {mean_gap:+,.2f}。"
                     f"单步对照确认 {saves} 次撤离救活了原地会死亡的种植工。"
                     f"候选 {heavy_steps} 盘的最大现金领先跳变都发生在本方回库为零、考官同时入库超过 300 的步上。"
                     "下一项开发实验应保留已验证的即时避险，比较提升己方重货兑现或在敌方入库前争夺回程路段的实际终局收益。"
                     "本次没有独立正式胜率结果。",""]
    for p in pairs:
        for arm,name in (("parent","父代"),("candidate","候选")):
            a=p[arm]
            for rel,player in enumerate(("我","考官")):
                s=p["seat"]^rel;v=a["by_physical_seat"][s]
                lines.append(f"| {p['seed']} / {p['seat']} | {name} / {player} | {fmt(a['terminal'][s])} | {fmt(v['bank'])} | {fmt(v['tree'])}（{fmt(v['enemy_cargo_tax_within_tree'])}） | {fmt(v['captured_into_collector'])} / {fmt(v['captured_then_banked'])} | {fmt(v['donation'])} / {fmt(v['collision_cash'])} | {fmt(-v['recruit'])} / {fmt(-v['plant'])} / {fmt(-v['seize'])} |")
    lines+=["","## 哪些科目随候选一起变化","",
            "| 候选相对父代，每局平均 | 本方变化 | 考官变化 | 本方相对变化 |","|---|---:|---:|---:|"]
    labels=dict(bank="正常回库",tree="树息及敌货税",donation="敌送基地",collision_cash="碰撞直接进现金",
                recruit="招工现金净流",plant="新种现金净流",seize="夺树现金净流",round="结算舍入")
    for k in KEYS:
        a,b=(mean_component(pairs,k,r) for r in (0,1))
        lines.append(f"| {labels[k]} | {fmt(a)} | {fmt(b)} | {fmt(a-b)} |")
    lines+=["","## 双方实际轨迹从何时分开","",
            "| 图 / 我席 | 首次实际联令不同 | 本方首先不同 | 考官首先不同 | 候选在自己公共局面首次改 v1 | 共改 | 救活 / 反伤 / 放弃已可成交 |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    cases=[]
    for p in pairs:
        z=p["divergence"];es=p["candidate"]["edits"]
        cases+=[(p,e) for e in es]
        lines.append(f"| {p['seed']} / {p['seat']} | {reg(z['first_either'])} | {reg(z['first_own'])} | {reg(z['first_examiner'])} | {reg(z['first_local_edit'])} | {len(es)} | {sum(e['saved'] for e in es)} / {sum(e['harmed'] for e in es)} / {sum(e['skipped_valid_purchase'] for e in es)} |")
    early=sum(p["divergence"]["opponent_already_different_before_first_edit"] is True for p in pairs)
    lines+=["",f"{early} 个对照在候选第一次改令前，考官的实际命令就已经分开。原版温度采样与搜索墙钟都保留，回放只能定位实际分歧。一次持续到终局的同席成绩差，无法全数归给其中一笔撤离。单步救活、反伤和放弃支付分别固定同一局面、同一条考官已发命令、其余工人不动，用官方 Board 与 FastCarbon 对两种动作各核一次。","",
            "## 四场候选现金领先变化最大的一步","",
            "统一取本方现金减对方现金的单步变化绝对值最大者，平手取最早。t 行代表发令前，t+1 为结算后。坐标是物理棋盘从左上起计的 (列, 行)。重点跟踪当步实际收付有关的工人，两边没有收付时列当时载货最高的采集工。"]
    for p in pairs:
        r=p["candidate"]["largest_abs_cash_lead_change"];s=p["seat"];t=r["pre"]
        a,b=r["components"][s],r["components"][1-s]
        lines+=["",f"### {p['seed']} 席 {s}，t{t} → t{t+1}","",
                f"本方现金变 {fmt(r['own_delta'])}，考官变 {fmt(r['enemy_delta'])}，本方领先变化 {fmt(r['delta_cash_lead'])}。正常回库分别 {fmt(a['bank'])} / {fmt(b['bank'])}，树收入 {fmt(a['tree'])} / {fmt(b['tree'])}，合计招种夺支出 {fmt(-a['recruit']-a['plant']-a['seize'])} / {fmt(-b['recruit']-b['plant']-b['seize'])}。","",
                "| 发令前 | 本方工号、位置、货、命令 | 考官工号、位置、货、命令 | 双方实际正常回库 → 下一帧 |",
                "|---|---|---|---:|"]
        for snap in r["states"]:
            by=next(x["components"] for x in r["nearby_cash"] if x["pre"]==snap["t"])
            lines.append(f"| t{snap['t']} | {focus_trace(snap,s,r['focus'][s])} | {focus_trace(snap,1-s,r['focus'][1-s])} | {fmt(by[s]['bank'])} / {fmt(by[1-s]['bank'])} |")
        approach=r["largest_rival_delivery_approach"]
        if approach:
            gaps="，".join(f"t{a['t']} 为 {a['gap_to_enemy'] if a['gap_to_enemy'] is not None else '无轻工'}" for a in approach["positions"])
            lines+=["",f"当步最大一笔为 {approach['worker']} 在 {tuple(approach['bank_xy'])} 回库 {fmt(approach['amount'])}。"
                        f"它与本方最近、货更轻的采集工的实际环面距离（格）依次为 {gaps}。"
                        "距离只描述当时的空间关系，在对方基地碰撞还可能把货交给对方。"]
    lines+=["","## 防守的直接收益与代价",""]
    wins=[v for v in cases if v[1]["saved"]]
    if wins:
        p,e=max(wins,key=lambda v:(bool(v[1]["next_actual_purchase_within_30"]),v[1]["living_after_10"],-v[1]["t"]))
        follow=e['next_actual_purchase_within_30']
        observation=(f"这名工在实际后续 t{follow['post']} 又{('种树' if follow['kind']=='plant' else '夺树')}，支付 {fmt(follow['price'])}。"
                     if follow else "实走后 30 步中，这名工没有新的种夺付款记录。")
        lines.append(f"在图 {p['seed']} 席 {p['seat']} t{e['t']}，{e['uid']} 从 {tuple(e['xy'])} 发 {e['move']} 存活，留在原地则当步死亡。{observation}")
    else:lines.append("在这四场候选实战实际触发的改令中，未找到同根一步能证明的救活。")
    negative=[v for v in cases if v[1]["harmed"] or v[1]["skipped_valid_purchase"]]
    if negative:
        p,e=max(negative,key=lambda v:(v[1]["harmed"],v[1]["branches"]["NONE"]["paid"],-v[1]["t"]))
        a=e["branches"]["NONE"];b=e["branches"][e["move"]]
        lines.append(f"反例在图 {p['seed']} 席 {p['seat']} t{e['t']}。{e['uid']} 从 {tuple(e['xy'])} 发 {e['move']} 后"+("死亡" if b["died"] else "存活")+"，同根发 NONE 则"+("死亡" if a["died"] else "存活")+f"、实际种夺付款 {fmt(a['paid'])}。少支出和放弃持有同时发生；这棵未取得的树在另一条未来轨迹上的最终回报未知。")
    else:lines.append("实际改令中未发现同根造成新死亡或放弃本可成交的种夺，四局不足以排除未见反例。")
    lines+=["","每个选定窗口记录回溯时刻双方在场工的完整命令与货量，并逐步列出窗口内每次实际回库。全场命令保存在两个参试包各自的原始对局凭证中。抢到的货沿真实碰撞、树税和回库路径核对，落地散失、被对面再抢与进入现金分开统计。每局两席入账分项在相分相减前独立核平。[逐局现金和改令后果](games.json.gz)",""]
    (output/"README.md").write_text("\n".join(lines))
