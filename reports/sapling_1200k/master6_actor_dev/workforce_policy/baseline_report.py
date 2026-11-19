"""Summarize complete v1 games separately from the previously exposed parent continuation branches."""
import gzip,json
from collections import Counter
from hashlib import sha256
from pathlib import Path
from statistics import median

HERE=Path(__file__).resolve().parent
BANDS=((0,59),(60,149),(150,219),(220,259),(260,279),(280,289),(290,298))
def desc(h):
    n=len(h);return {'hires':n,'visible_after_first_step':sum(w['birth_visible_after_step'] for w in h),
        'direct_lifetime_positive':sum(w['lifetime_net']>0 for w in h),'direct_10step_recovered_fee':sum(w['paid_back_by_10'] for w in h),
        'gross_bank':round(sum(w['bank'] for w in h),3),'gross_tree':round(sum(w['tree'] for w in h),3),
        'plant_seize_cost':round(sum(w['tree_cost'] for w in h),3),'hire_cost':round(sum(w['price'] for w in h),3),
        'lifetime_direct_net':round(sum(w['lifetime_net'] for w in h),3),'median_worker_net':round(median(w['lifetime_net'] for w in h),3) if n else None}
def audit_group(games,side,lo=0,hi=298,pred=lambda r:True):
    rows=[(g,r) for g in games for r in g['base_rows'] if r['seat']==(side^g['seat']) and max(lo,g['source'].get('root',0))<=r['t']<=hi and pred(r)]
    legal=[(g,r) for g,r in rows if r['choices']]
    clear=[(g,r) for g,r in legal if not r['own_landing'] and not r['opponent_in_one_step']]
    ws=[g['workers'][r['uid']] for g,r in rows if 'uid'in r]
    cmd=Counter(r['command'] for g,r in rows);requested=cmd['RECCOLLECTOR']+cmd['RECPLANTER']
    return {'slots':len(rows),'at_capacity_slots':sum(r['remaining_capacity']==0 for g,r in rows),
            'insufficient_to_hire_either_slots':sum(r['pre_cash']<min(r['prices'].values()) for g,r in rows),
            'capacity_and_either_affordable':len(legal),'capacity_and_affordable_none':sum(r['command']=='NONE' for g,r in legal),
            'clear_capacity_affordable':len(clear),'clear_none':sum(r['command']=='NONE' for g,r in clear),
            'clear_none_distinct_map_seat_turns':len({(g['seed'],g['seat'],r['t']) for g,r in clear if r['command']=='NONE'}),
            'attempts':requested,'paid':len(ws),'denied':requested-len(ws),'visible_after_first_step':sum(w['birth_visible_after_step'] for w in ws),
            'paid_with_under_next_hire_left':sum(r['after_hire_cash']<min(r['prices'].values()) for g,r in rows if 'uid' in r),
            'paid_with_under_first_plant_or_seize_left':sum(not r['would_leave_plant_or_seize'][g['workers'][r['uid']]['role']] for g,r in rows if 'uid' in r),
            'on_friendly_landing_paid':sum(r['own_landing'] for g,r in rows if 'uid' in r),
            'threat_adjacent_paid':sum(r['opponent_in_one_step'] for g,r in rows if 'uid' in r),
            'C':desc([w for w in ws if w['role']=='C']),'P':desc([w for w in ws if w['role']=='P'])}
def main():
    file=HERE/'baseline_games.json.gz';data=json.loads(gzip.decompress(file.read_bytes()));gs=data['games'][:6];branches=data['games'][6:]
    if len(gs)!=6 or len(branches)!=2 or data['source_sha256']!=sha256((HERE/'baseline.py').read_bytes()).hexdigest():raise ValueError('source changed')
    result={'evidence_sha256':sha256(file.read_bytes()).hexdigest(),'sides':{},'games':[],'examples':[]}
    for side,key in ((0,'v1'),(1,'Master3s2')):
        result['sides'][key]={'all':audit_group(gs,side),'by_phase':{f'{a}-{b}':audit_group(gs,side,a,b) for a,b in BANDS},
                'already_two_planters_pre220':audit_group(gs,side,0,219,lambda r:r['workersP']>=2),
                'all_hires_60_219':audit_group(gs,side,60,219),'late_280_plus':audit_group(gs,side,280,298),
                'shared_branch_post_root_only':audit_group(branches,side)}
    for g in gs+branches:
        sv,op=g['summary'][g['seat']],g['summary'][1-g['seat']]
        result['games'].append({'group':g['source']['group'],'seed':g['seed'],'seat':g['seat'],'source':g['source']['source'],
                                'branch_root':g['source'].get('root'),'v1_bank':sv['bank'],'enemy_bank':op['bank'],
                                'v1_tree':sv['tree'],'enemy_tree':op['tree'],'scores':g['source']['official_rewards']})
    examples=[]
    for g in gs:
        for r in g['base_rows']:
            if r['seat']!=g['seat'] or 'uid' not in r:continue
            w=g['workers'][r['uid']]
            if r['t']<220 and w['role']=='P' and r['workersP']>=2:key='third_planter'
            elif r['t']>=280 and w['role']=='C':key='late_collector'
            else:continue
            if w['lifetime_net']==0:continue
            kind=key+('_returned' if w['lifetime_net']>0 else '_lost')
            if kind not in {e['type'] for e in examples}:
                examples.append({'type':kind,'seed':g['seed'],'seat':g['seat'],'root':r,'worker':{k:v for k,v in w.items() if k!='events'},
                                 'first_deposit_at':next((t for t,k,_ in w['events'] if k=='own_delivery'),None)})
    result['examples']=examples
    (HERE/'baseline_summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    def fmt(n):return f'{n:,.0f}'
    u,v=result['sides']['v1'],result['sides']['Master3s2']
    lines=['# v1 对原版 Master3s2 的招工账','',
       '六盘主样本来自冻结模型 `9c54df17…` 对完整原版 Master3s2 的三张已见开发图，每张换席。另复原 621461321 席0 与 621461322 席1 的原策略续弈，分别与主样本共享 t91、t103 之前的轨迹，分开列账。较早 416cc31b 双网、训练扰动和对其他考官的八局均排除。复算只读取既有命令，无新对局。[来源、分段与逐工收据](baseline_summary.json) · [完整逐基地/工人记录](baseline_games.json.gz)','',
       f'在三个地图对上，v1 共招 {u["all"]["C"]["hires"]} 名采集工、{u["all"]["P"]["hires"]} 名种植工，考官分别招 {v["all"]["C"]["hires"]}、{v["all"]["P"]["hires"]} 名。差异集中在时机和当时已有劳动力。t0–219 己方已有至少两名种植工时，v1 又招了 {u["already_two_planters_pre220"]["P"]["hires"]} 名，考官为 {v["already_two_planters_pre220"]["P"]["hires"]} 名；两边后来靠本人种植/夺取后持有期树息覆盖所有相关费用的分别有 {u["already_two_planters_pre220"]["P"]["direct_lifetime_positive"]} 和 {v["already_two_planters_pre220"]["P"]["direct_lifetime_positive"]} 名。六盘中 v1 的实际树收入逐盘均低于考官。这个差额也可能来自站位、存活、夺树或守树，单凭人数无法识别原因。','',
       '| 购买决策步 | v1 采工（付费/直接净正） | 实际净额 | v1 种工（付费/直接净正） | 实际净额 | 空闲可付且安全/其中未招 |','|---|---:|---:|---:|---:|---:|']
    for label,b in u['by_phase'].items():
        c,p=b['C'],b['P'];lines.append(f'| {label} | {c["hires"]}/{c["direct_lifetime_positive"]} | {fmt(c["lifetime_direct_net"])} | {p["hires"]}/{p["direct_lifetime_positive"]} | {fmt(p["lifetime_direct_net"])} | {b["clear_capacity_affordable"]}/{b["clear_none"]} |')
    lines+=['',f'在 t280–298，v1 的 10 个新采集工仅 1 人直接净正，合计 −172；t260–279 的 13 人中有 9 人净正，合计 +650。提前关闭整个尾盘招工会错过已实现的收益。建议先验证两类具体选择：已有多种工时继续招同工种、改招采工或保现金，以及最后十几步把剩余回程与可交货量算入招工价值。这里的开口和收益来自观察样本，替换指令的终局收益需要同根实际续弈。','',
       '| 全局基地槽/资源 | v1 | Master3s2 |','|---|---:|---:|']
    for name,k in [('总决策槽','slots'),('已满员','at_capacity_slots'),('两工种均付不起','insufficient_to_hire_either_slots'),('尚有工位且至少付得起一类','capacity_and_either_affordable'),('上述可招但未招','capacity_and_affordable_none'),('其中无友方同时落点且无一邻格敌工','clear_capacity_affordable'),('安全开放槽未招','clear_none'),('实际招工指令','attempts'),('实付并生成ID','paid'),('新工在该次推进后仍在场','visible_after_first_step'),('指令因钱/容量未执行','denied'),('实付后不足下次最低招工价','paid_with_under_next_hire_left'),('实付后不足最低种植或夺树投入','paid_with_under_first_plant_or_seize_left')]:
        lines.append(f'| {name} | {u["all"][k]} | {v["all"][k]} |')
    lines+=['','安全开放是以行动前公开资产和己方已定的工人动作计算：该基地不在己方工人的落点，敌方没有一步可到达的任何工人，钱与工位按真实基地执行顺序核算。这是保守的几何范围，不能保证未来安全。一个回合可能有两个可招基地；钱或工位只够一个时，两个空过各自仍代表一个可选动作。每个实际新工身份由模拟器生成计数、真实支付事件和下一帧存活分别核验。','',
        f'三图六盘的 v1 种植工收入按真实树的归属期归因：即使支付种/夺费的种植工已死，树只要仍由本方持有，其收益继续记入同一笔投资。132 名种植工合计直接净额 {fmt(u["all"]["P"]["lifetime_direct_net"])}，考官 122 名为 {fmt(v["all"]["P"]["lifetime_direct_net"])}。t0–219 已有至少两名种工仍加招时，两边该组净额分别为 {fmt(u["already_two_planters_pre220"]["P"]["lifetime_direct_net"])}、{fmt(v["already_two_planters_pre220"]["P"]["lifetime_direct_net"])}。8 条完整或接续的命令路径均检查所有进账都有明确付款/回库工，未分配金额为零。','',
        '| 地图/物理席 | 主体与考官回库 | 主体与考官树收入 |','|---|---:|---:|']
    for g in result['games']:
        label='续弈自 t'+str(g['branch_root']) if g['branch_root'] is not None else '完整'
        lines.append(f'| {g["seed"]} / {g["seat"]}（{label}） | {fmt(g["v1_bank"])} / {fmt(g["enemy_bank"])} | {fmt(g["v1_tree"])} / {fmt(g["enemy_tree"])} |')
    bx=u['shared_branch_post_root_only']
    lines+=['',f'两条分支的整场经济表保留共享早段。从分支起点之后另计，v1 支付并生成 {bx["C"]["hires"]} 名采集工和 {bx["P"]["hires"]} 名种植工。它们不能充当另外两张开发图。','',
        '下列例子按原存档顺序，在同类现象中各取首个直接净正和净负的实招。金额为该 ID 直到终局实际交库，加上其本人建/夺的每一棵树在随后本方持有期间入账的树息，扣招工费及该 ID 的种/夺成本。直接收入没有给阻断敌人、护树或掩护队友定价；不同工种间的直接净额也不等同于替换后的终局差。','',
        '| 实例 | 地图/席/步 | 新工 ID | 决策前采/种 | 付款后现金 | 该工回库/树息/树投资 | 直接净额 |','|---|---|---|---:|---:|---:|---:|']
    for e in examples:
        w,r=e['worker'],e['root'];lines.append(f'| {e["type"]} | {e["seed"]}/{e["seat"]}/{r["t"]} | {w["uid"]} | {r["workersC"]}/{r["workersP"]} | {r["after_hire_cash"]:.2f} | {w["bank"]:.2f}/{w["tree"]:.2f}/{w["tree_cost"]:.2f} | {w["lifetime_net"]:+.2f} |')
    lines+=['','历史 V4 在高手 454 个可选基地动作中给出 346 次招种工，真实示范仅 81 次。对原版四局，V3 到 V4 的平均树收入从 129 升到 1,372，但正常交货从 5,313 降至 950，四局分差均恶化。V4 在另外 16 局规则对手卷上种下 1,011 棵，601 次投资后连基础招工 30 也不够。该批旧策略与本次 v1 的六局分开统计。[历史训练审计](../../TREE_V4_TRAINING.md) · [原版考试与反例](../../MASTER3S2_90.md)','',
      '从当前目录依次运行 `python baseline.py` 和 `python baseline_report.py` 即可复算。两程序不运行模型推理或新比赛。[逐步重演](baseline.py) · [汇总程序](baseline_report.py)','']
    (HERE/'baseline.md').write_text('\n'.join(lines))
if __name__=='__main__':main()
