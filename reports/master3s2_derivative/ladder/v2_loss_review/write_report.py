"""Render the public review directly from independently reconciled game ledgers."""
import gzip
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

HERE=Path(__file__).resolve().parent
def read(p):
    return json.load(gzip.open(p)) if p.suffix==".gz" else json.loads(p.read_bytes())
def money(n):
    return f"{n:+,.2f}".replace("-", "−")
def when(timestamp):
    return datetime.fromisoformat(timestamp).astimezone(timezone(timedelta(hours=8))).strftime("%H:%M:%S")
def link(card):
    return f"{card['match']}-{card['game']}.ledger.json.gz"


s=read(HERE/"summary.json")
groups=s["groups"]
outs=lambda g,k:g["outcomes"].get(k,0)
rows=[]
def add(txt=""): rows.append(txt)
add("# ds-pro-v2 天梯败局全量复盘")
add()
add(f"截至北京时间 **2026-11-20 {when(s['utc'])}**，按 v2 注册 ID `{s['strategy']}` 读取所有公开赛卡，逐局区分定位与普通天梯。首场创建于 {when(s['first_created'])}，最后一场已知结算于 {when(s['last_done'])}。部署文件 SHA-256 为 `{s['observed_package']}`。[包身份](../ds-pro-v2.json) · [初次赛卡](cards.json.gz) · [第二次读回](readback.json.gz) · [截止时分页读回](final.json.gz) · [赛号、席位、对手注册 ID、UTC 时间及状态](summary.json)")
add()
add("| 赛种 | 已结算 | 胜 / 平 / 负 | 席0胜 / 负 | 席1胜 / 负 | 平均官方分差 | 待结算 |")
add("|---|---:|---:|---:|---:|---:|---:|")
for kind,label in (("newrank-placement","定位"),("newrank-ladder","普通天梯")):
    g=groups[kind];o=g["outcomes"];wins,draws,losses=(outs(g,t) for t in "WDL")
    seat0,seat1=(g["seats"][str(i)] for i in (0,1))
    add(f"| {label} | {g['completed']} | {wins} / {draws} / {losses} | {seat0.get('W',0)} / {seat0.get('L',0)} | {seat1.get('W',0)} / {seat1.get('L',0)} | {money(g['margins_mean'])} | {g['pending']} |")
add()
add(f"定位赛 {groups['newrank-placement']['completed']} 张赛图的五场失利都在席 1。对手、地形及搜索采样没有配对，席别本身的效果须另验。截至该时点共有 {len(s['losses'])} 场已结算败局。对每份败局回放，从 t0 读取官方全盘，只在离线复算中使用，随后逐拍施加双方实发命令。全部败局均为双方 DONE、完整 299 拍，官方模拟的现金、工人、树、全图碳量每拍与回放相等，另用独立实现核对现金事件。[复核程序](review_v2.py)")
add()
add("## 每场的经济差")
add()
add("同一行均以 v2 为正方向。树净已扣种植与夺树开支；截获货进入工人背包，不能与现金再加一次。招募差及舍入补齐官方终局分差。每个对手名指向该局包含全部实发命令、现金曲线、按窗双方收入和逐树持有史的账本。")
add()
add("| 对手 | 赛种 / 席 | 官方比分（v2 在前） | 分差 | 归库差 | 树净差 | 招募开支差 | 截获敌货差 |")
add("|---|---|---:|---:|---:|---:|---:|---:|")
games=[]
for c in s["losses"]:
    g=read(HERE/link(c));games.append(g);d=g["whole"]["difference"];mode="定位" if c["kind"]=="newrank-placement" else "普通"
    assert g["official"]["matched"]==299 and g["official"]["first_divergence"] is None
    assert all(v=="DONE" for v in c["statuses"].values())
    add(f"| [{c['opponent']}]({link(c)}) | {mode} / {c['seat']} | {c['score'][0]:,.2f} / {c['score'][1]:,.2f} | {money(c['margin'])} | {money(d['bank'])} | {money(d['net_tree'])} | {money(d['recruit'])} | {money(d['robbed_cargo'])} |")
add()
add("## 败局在什么时候形成")
add()
add("首次持续落后以连续至少 10 拍为界。各宽度选自身现金相对差下降最多的区间，保留两个时点，不把窗口内所有亏损都归给一名工人。")
add()
add("| 对手 | 首次连续落后 | 最后领先 | 最深落后（时点，分差） | 最差3拍 | 最差5拍 | 最差10拍 |")
add("|---|---:|---:|---:|---:|---:|---:|")
for g in games:
    c=g["card"];w=g["worst"];first=g["first_sustained_deficits"]["10"];cols=[]
    for i in (3,5,10):
        e=g["worst_windows"][str(i)];cols.append(f"t{e['start']}–{e['end']}，{money(e['margin_to']-e['margin_from'])}")
    add(f"| {c['opponent']} | t{first['start']}–{first['end']} | t{g['last_strict_lead']} | t{w['step']}，{money(w['margin'])} | "+" | ".join(cols)+" |")
add()
add("以下每局选上表最差 10 拍内，对方最大的一笔真实回库，沿该工人的实发轨迹向前查。右侧同时列出窗口开始时双方全部在途货及窗口内的己方／对手实际回库。己方可在场上看到双方工人的位置和持货；敌方当拍动作及其未来到账只能用作复盘标签。")
add()
add("| 对手 | 窗口初始在途货（v2 / 敌） | 窗口归库（v2 / 敌） | 对手工人轨迹（格号 · 动作 · 载货） |")
add("|---|---:|---:|---|")
for g in games:
    c=g["card"];b=g["worst_windows"]["10"];before=g["steps"][b["start"]]["teams"];tx=b["returns"]
    totals=[sum(e["amount"] for e in tx if e["seat"]==side) for side in (c["seat"],1-c["seat"])]
    e=max((i for i in tx if i["seat"]!=c["seat"]),key=lambda i:i["amount"], default=None)
    traj="期间无对手回库"
    if e:
        uid=e["worker"].split("-")[-1]
        points=[]
        for h in e["history"]:
            if h["t"] not in [e["t"]-10,e["t"]-5,e["t"]-3] or h["state"] is None:continue
            points.append(f"t{h['t']} 格{h['state'][0]}·{h['next_if_present']}·{h['state'][1]:.2f}")
        traj=f"#{uid} "+" → ".join(points)+f" → t{e['t']} 入账 {e['amount']:.2f}"
    add(f"| {c['opponent']} | {before[0]['cargo']:.2f} / {before[1]['cargo']:.2f} | {totals[0]:.2f} / {totals[1]:.2f} | {traj} |")
add()
add("## 对局给下一版训练的具体约束")
add()
add("1. **区分值得守的树和回不了本的新树。** 对 Fable v7，v2 种 40 棵，23 棵在原持有期回本，17 棵在回本前被夺。树毛收入比对方高 1,036.10，种植多付 1,443.34，扣除夺树差后树净反落后 147.24。对小肥鱼时情况相反：v2 种 12 棵、11 棵回本，对方种 41 棵、38 棵回本；我方省下 1,631.92 种植费，却少取 3,196.77 树毛收入。模型需要同时估计当前种树机会的可守期限、双方种树工抵达时间、已有树的延续租金，以及离开这一格能做的事。仅给种树加权或减权都会碰到现成反例。")
add("2. **预测对手接下来一段时间能交付多少，以及是否有机会阻断。** 对哈基米 v3，t281 现金领先 577.82，己方在途 429.45，对方 1,313.95；接着十拍资金形势逆转 1,350.53。对 stephen，t56–66 的回库差来自敌方四名工人到账，v2 同窗没有一笔。训练输入可使用当前两个基地的路径、两侧工人预计到达及携货、较轻采集工的拦截范围。真实未来的归库和可被原版真实应答验证的改令保留作监督；没有合法拦截机会时，应比较己方加快交付的收益。")
add("3. **原版对手这一局要同时提高经营和回款。** 对天梯注册名 claude-AlphaCarbon-Master3s2，v2 的最后一次领先发生在 t60；t288 我方已经没有树，对手 9 棵，现金仍只落后 154.73，但我方在途 137.90，对手 579.72。第 149、150 拍三笔敌方回库与最后十拍的连续回款均能定位。帐面上树净少 516.04，回库少 225.21，招募费相同。作为同源教师或评测者的身份应另由源码哈希认证，单凭天梯报名名不能证明其与仓内 `master3s2.py` 字节相同。")
add()
add("**同名对手对照。** v2 对同一注册 ID 的 doubao-flash-v1 曾席 0 赢 416.33，席 1 负 1,005.31。胜局回库领先 325.61、树净领先 120.82；负局分别落后 569.35 和 405.82。v1 对同一注册 ID 的 stephen-K4TacSearch-v6 曾在席 1 赢 473.58，v2 在另图同席输 508.89。此处的官方地图哈希不同，不能据此归因某个网络层。[doubao 胜局逐拍账](414c6ee737eec3f8c4ab93e45163ee8f-1.ledger.json.gz) · [v1 已读回赛卡](../placement_refresh/summary.json)")
add()
add("每场回放文件都以比赛 ID 加局号命名。新模型若复用这些轨迹训练或择策，须把其地形列入已曝光集合。")
add()
for g in games:
    c=g["card"];path=f"replays/{c['match']}-{c['game']}.json.gz";add(f"- [{c['opponent']} 官方原回放]({path}) · [全程经济、动作与时点]({link(c)})")
(HERE/"README.md").write_text("\n".join(rows)+"\n")
