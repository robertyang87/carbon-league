"""Render the Chinese decision readout from the completed, unselected hosted matrix."""
from collections import Counter,defaultdict
import gzip
import json
import statistics

import settle as S

NAMES={"parent":"父代","student":"H24"}
CASH={"own_delivery":"己方交回本营","opponent_worker_delivery_to_this_base":"敌方误入存入","ground_rent":"地租",
      "cargo_tax":"货税","tree_collision_cash":"树格碰撞入账","base_collision_cash":"基地碰撞入账",
      "plant":"栽树支出","seize":"夺树支出","hire_planter":"招种植工","hire_collector":"招采集工","rounding":"舍入"}
def fmt(v):
    return "未产生" if v is None else f"{v:+,.2f}"
def ms(v):
    return "—" if v is None else f"{1000*v:,.0f}"
def num(v):
    return "—" if v is None else f"{v:,.1f}"
def average(values):
    return statistics.mean(values) if values and all(v is not None for v in values) else None
def wdl(z):
    if not z['scored']:return '未记'
    if z['winner'] is None:return '平' if z['game']['status']=='complete' else '双方故障'
    return '胜' if z['winner']==z['game']['seat'] else '负'
def settle_display(ledger, side):
    def total(keys):return sum(ledger[side].get(k,0) for k in keys)
    return ' / '.join(f"{v:,.0f}" for v in (total(['own_delivery']),total(['ground_rent']),total(['cargo_tax']),
                -total(['plant','seize']),-total(['hire_collector','hire_planter']),total(['opponent_worker_delivery_to_this_base','tree_collision_cash','base_collision_cash'])))
def report():
    raw=S.HERE/"settled_games.json.gz"
    r=json.loads(gzip.decompress(raw.read_bytes()))
    games=sorted(r['games'],key=lambda z:(z['game']['seed'],z['game']['seat'],z['game']['arm']))
    assert len(games)==8
    lines=["# H24 与父代在两张新开发图上的整局攻防与现金结算","",
        "预先登记地图 621758352、621758353，各打一遍双席，H24 与父代各四局。每局独立冷启，考官为仓内原版 Master3s2，保留原生温度、搜索截止和自然采样。",
        "", "## 整批结果", "", "|候选|席 0 胜/平/负|席 1 胜/平/负|合计胜/平/负|已登记场次|平均官方分差|", "|---|---|---|---|---:|---:|"]
    groups=defaultdict(list)
    for z in games:groups[z['game']['arm']].append(z)
    def counts(rows):
        c=Counter(wdl(z) for z in rows)
        s='/'.join(str(c[k]) for k in ('胜','平','负'))
        return s if len(rows)==sum(c[k] for k in ('胜','平','负')) else s+f"（另{len(rows)-sum(c[k] for k in ('胜','平','负'))}场见状态）"
    for arm,ms_ in groups.items():
        lines.append(f"|{NAMES[arm]}|{counts([z for z in ms_ if z['game']['seat']==0])}|{counts([z for z in ms_ if z['game']['seat']==1])}|{counts(ms_)}|{len(ms_)}|{fmt(average([z['official_margin'] for z in ms_]))}|")
    average_change=average([z["official_margin"] for z in groups["student"]])
    average_base=average([z["official_margin"] for z in groups["parent"]])
    if average_change is not None and average_base is not None:
        stage=[]
        for period in ("0_99","100_199","200_298"):
            def mean_diff(fields):
                return statistics.mean(sum(z["margin_components"][period][k] for k in fields) for z in groups["student"]) - statistics.mean(sum(z["margin_components"][period][k] for k in fields) for z in groups["parent"])
            stage.append((mean_diff(CASH), mean_diff(["own_delivery"]),mean_diff(["ground_rent"]),mean_diff(["plant"])))
        lines.extend(["",f"H24 平均官方分差较父代变化 **{average_change-average_base:+,.3f}**。按游戏阶段拆开，前 100 拍为 **{fmt(stage[0][0])}**、中间 100 拍为 **{fmt(stage[1][0])}**、最后 99 拍为 **{fmt(stage[2][0])}**。前 100 拍的双方净地租变化为 {fmt(stage[0][2])}，交付净差变化为 {fmt(stage[0][1])}，栽树开支净变化为 {fmt(stage[0][3])}。"])
    lines.extend(["","|地图|候选席位|版本|全场状态|本方 / 考官官方分|分差|末段交付净差|全局地租净差|","|---|---:|---|---|---|---:|---:|---:|"])
    for z in games:
        g=z['game'];s=g['seat'];scores=g['official_scores_by_seat']
        scores_text=f"{scores[s]:,.2f} / {scores[1-s]:,.2f}" if scores is not None else "未产生"
        lines.append(f"|{g['seed']}|{s}|{NAMES[g['arm']]}|{wdl(z)} · {z['status']} · {g['official_transitions']} 步|{scores_text}|{fmt(z['official_margin'])}|{fmt(z['margin_components']['200_298']['own_delivery'])}|{fmt(z['margin_components']['whole']['ground_rent'])}|")
    lines.extend(["","末段指 t=200–298 的最多 99 次联合行动。交付净差按双方工人向各自基地的实际交款相减。树木吸收地面碳和向对方载货工收取的税分别结算。逐场文件同时保留实际最后 99 拍，在提前结束时标注有效步数。","",
        "## 收入与支出去向","",
        "各栏为同一方整局的交付 / 地租 / 货税 / 栽夺树支出 / 招工支出 / 碰撞与敌方基地交付入账，单位为游戏分，支出列为正。原始精度与舍入项见逐局证据。","",
        "|地图|席|版本|本方|考官|","|---|---:|---|---|---|"])
    for z in games:
        g=z['game'];s=g['seat'];l=z['money_by_period_by_seat']['whole']
        lines.append(f"|{g['seed']}|{s}|{NAMES[g['arm']]}|{settle_display(l,s)}|{settle_display(l,1-s)}|")
    lines.extend(["","## 原版搜索与平台时钟","","每步耗时从外部发出公开请求计至取得并解析响应。P95/P99 用最近秩，表内时间单位为毫秒。慢步按超过 1 秒计，缺测及执行故障保留。","","|地图|席|版本|考官完成搜索 / 单候选 / 零深回退|考官完成深度（次数）|本方 P95 / P99 / 最慢|考官 P95 / P99 / 最慢|慢步 本方 / 考官|整场秒|","|---|---:|---|---|---|---|---|---|---:|"])
    for z in games:
        g=z['game'];s=g['seat'];q=z['original_search']
        depths=', '.join(f"{d}层×{n}" for d,n in sorted(q['depth'].items(),key=lambda p:int(p[0])))
        br=q['branches'];b=' / '.join(str(br.get(k,0)) for k in ('searched','single','zero'))
        if br.get('unresolved_no_search'):b+=f"（另{br['unresolved_no_search']}未归类）"
        tt=z['ladder_clock_by_seat']
        clocks=[' / '.join(ms(tt[i].get(k)) for k in ('p95','p99','max')) for i in (s,1-s)]
        lines.append(f"|{g['seed']}|{s}|{NAMES[g['arm']]}|{b}|{depths or '无'}|{clocks[0]}|{clocks[1]}|{tt[s].get('slow',0)} / {tt[1-s].get('slow',0)}|{num(z['whole_wall_seconds'])}|")
    lines.extend(["","## 同图同席的差值","","H24 相对父代的差值均取各自独立整局。末段只累计 t=200–298；树支出为带符号的双方栽夺支出净变化，正值有利于 H24。","","|地图|席|终局分差改善|末段交付净差改善|全局地租净差改善|全局树支出净改善|","|---|---:|---:|---:|---:|---:|"])
    for seed in sorted({z["game"]["seed"] for z in games}):
        for seat in (0,1):
            pair={z["game"]["arm"]:z for z in games if (z["game"]["seed"],z["game"]["seat"])==(seed,seat)}
            v=pair["student"];b=pair["parent"]
            def component(period,fields):
                return sum(v["margin_components"][period][k]-b["margin_components"][period][k] for k in fields)
            change=None if v["official_margin"] is None or b["official_margin"] is None else v["official_margin"]-b["official_margin"]
            lines.append(f"|{seed}|{seat}|{fmt(change)}|{fmt(component('200_298',['own_delivery']))}|{fmt(component('whole',['ground_rent']))}|{fmt(component('whole',['plant','seize']))}|")
    lines.extend(["","下表按阶段列出 H24 相对父代在双方现金净变动上的变化。负数表示 H24 在该段多失分。","","|地图|席|t0–99|t100–199|t200–298|","|---|---:|---:|---:|---:|"])
    for seed in sorted({z["game"]["seed"] for z in games}):
        for seat in (0,1):
            pair={z["game"]["arm"]:z["margin_components"] for z in games if (z["game"]["seed"],z["game"]["seat"])==(seed,seat)}
            delta=[sum(pair["student"][k].values())-sum(pair["parent"][k].values()) for k in ("0_99","100_199","200_298")]
            lines.append(f"|{seed}|{seat}|{'|'.join(fmt(x) for x in delta)}|")
    lines.extend(["","逐拍证据包含实际候选数、完成深度、原生截止、整段零深回退和测得的公开入口 CPU 与墙钟耗时。回退场仍纳入全批。每局冷启、累计超额和资源峰值见同一证据文件。","",
                  "## 最大单步现金损失之前发生了什么","","每场取首个使本方相对现金下降最多的联合行动。表中列出该拍最大的两项现金变动，符号表示本方减考官。3/5/10 为回溯相应拍数再计入转折拍的累计净变动。前序双方逐工人指令、携货、基地、树位和现金均保留，可用来检查布局和应对。","",
                  "|地图|席|版本|转折 t|该拍净变动|两项主要变动|回溯 3 / 5 / 10|","|---|---:|---|---:|---:|---|---|"])
    for z in games:
        g=z['game'];s=g['seat'];p=z['largest_observed_cash_loss']
        if not p:
            lines.append(f"|{g['seed']}|{s}|{NAMES[g['arm']]}|—|—|未产生有效转移|—|");continue
        b=S.balance(p['from_t_minus_10'][-1]['money'],s)
        keys=sorted((k for k in b if abs(b[k])>.001),key=lambda k:-abs(b[k]))[:2]
        lines.append(f"|{g['seed']}|{s}|{NAMES[g['arm']]}|{p['t']}|{fmt(p['change'])}|{', '.join(CASH[k]+' '+fmt(b[k]) for k in keys)}|{' / '.join(fmt(p['lookbacks'][str(h)]['cumulative_margin_change']) for h in (3,5,10))}|")
    lines.extend(["","## 最重的连续十拍亏损","","|地图|席|版本|最差十拍|现金净变化|最大两项来源|","|---|---:|---|---|---:|---|"])
    for z in games:
        g=z["game"];p=z["largest_losing_spans"].get("10")
        if p:
            a=p["component_diff"];ks=sorted((k for k in a if abs(a[k])>.001),key=lambda k:-abs(a[k]))[:2]
            lines.append(f"|{g['seed']}|{g['seat']}|{NAMES[g['arm']]}|{p['first_t']}–{p['last_t']}|{fmt(p['net_change'])}|{', '.join(CASH[k]+' '+fmt(a[k]) for k in ks)}|")
    lines.extend(["","逐局证据也列出最重的连续三拍、五拍，以及这些时段内双方每个单位的行动和状态。","",
                  "这批只有两张地图。每张地图上的多步决策和两席共享地图条件，无法从八局估出可靠的总体胜率。各局的考官独立自然采样，单局现金轨迹提供行动归因线索，无法确定单个动作的反事实终局收益。","",
                  f"来源为独立重放后核对的八套容器工件。矩阵 SHA-256 为 `{r['matrix_sha256']}`。[逐局结算与时钟](settled_games.json.gz) · [矩阵注册](../protocol.json)"])
    (S.HERE/"README.md").write_text('\n'.join(lines)+'\n')
if __name__=="__main__":
    report()
