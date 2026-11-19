"""Render the timing analysis, including conditional denominators and reconstructed fork witnesses."""
import gzip
import json
from pathlib import Path
from raid_timing import HERE,DEST
p=HERE
load=lambda name:json.loads((gzip.decompress((p/name).read_bytes()) if name.endswith('.gz') else (p/name).read_bytes()))
d=load('raid_timing.json.gz');lead=load('raid_timing_lead.json');cost=load('raid_timing_cost.json')['groups'];cases=load('raid_timing_examples.json.gz')
z=d['stages'];g=z['eligible']
def rate(n,r): return 1000*n/max(r,1)
def fmt_e(q):return f"{rate(q['cashed'],q['pair_roots']):.2f}"
def ci(q):
 v=q['cluster_ci'].get('banking_events_per_1000',[]);return f"[{v[0]:.2f}, {v[1]:.2f}]" if v else '—'
def table(label,q):
 return f"| {label} | {q['pair_roots']:,} | {q['selected']:,} | {q['won']:,} | {q['cashed']:,} | {fmt_e(q)} | {ci(q)} |"
heading=['| 行动前可见条件 | 工人对次 | 我方实际选交点 | 真正取得货 | 至少部分入账 | 入账笔／千对次 | 95% 区间 |','|---|---:|---:|---:|---:|---:|---:|']
txt=['# 抢货该在哪一步动手','',
'**最值得动手的窗口，是对方货车已没有安全出口、我方有敌基地以外的交点时。出手前还要算最短回库步数。** 在 2,928 次这种零出口机会中，168 笔最终兑现，每千次 57.38 笔。还剩一个出口时降到 3.59，两个及以上约 0.81。这个格局适合提高候选优先级；只准在零出口时出手，会漏掉 1,009 笔实收中的 841 笔。','',
'一次真实例子发生在 M9XJ 对 Master5s。撞击前三步，货车仍有两个安全落点；前一步所有出口被覆盖。M9XJ 带 44.03 从 157 格向上，Master5s 带 624.55 从 143 格向左，两人同时进 142 格，624.55 后来全额进了 M9XJ 的现金。相反，it720 有一次面对 714.26 碳的货车，双方交换位置互相穿过；只让 it720 原地不动即会收到这笔货。下面区分每步之前看得到的线索与事后才知道的对手行动。','',
'## 一次机会的定义和结果','',
f"全库 1,324 局、{d['n_unique_seeds']:,} 个地图种子，两席都扫。对局第 t 帧必须已有我方采集工，并有对手带至少 50 碳、比我方更重。若两人各走一步（含停留）的可达格相交，就记一个“工人对／决策时刻”。同一目标可对应多名追兵，连续多步也分别计。903,398 对次满足几何条件，其中 3,065 对次唯一交点都在对方基地；其余 900,333 对次可在基地外取得货。四局没有达到该入组条件，仍保留零根收据。",'',
f"在有非敌基地交点的 {g['pair_roots']:,} 对次里，我方选择合法交点 {g['selected']:,} 次，未选 {g['not_selected']:,} 次。实际在合法交点相遇 {g['selected_and_met']:,} 次，拿到目标货 {g['won']:,} 次，其中 {g['won_without_cash']:,} 次最终未给自己增加现金，{g['cashed']:,} 次至少部分入账，现金共 {g['banked_carbon']:,.2f}。选择交点包括停留或恰好同路，回放不能判定选手主观上是否想打劫。相对于全库各局两席所有正货量的 4,225 笔，此处只计满足轻重、货≥50、原工人在场等预先条件的 1,511 笔。按局、步、双方工号逐笔与全量攻防账复核，这 1,511 笔覆盖账上全部 ≥50 碳的采集工截获，货量与最终入库金额逐笔相符。",'',
f"以事后实际对手命令固定，出现 {g['missed_single_actor']:,} 个可改单名我方工人、即刻能从原动作未取得货的目标收到货的对次。这个数字没有考虑对手因新策略改变前序部署，也不推断随后能汇款。训练输入严格截止两边本步落子之前。",'',
'## 看盘顺序','',
'1. 先画双方各自的下一步格，排除敌基地。碰撞获胜后在那里缴货，收益会回到对手。然后以全体我方较轻或等重采集工的一步覆盖图，统计目标还能选几个不受威胁的非我方基地落点。覆盖表示能到达，各个威胁格并非可以由同一工人在同一步全占。','',
'2. 有真正封口时把截获列入比较，核算哪名轻工出手、哪名继续挡逃路。对手连续向家移动只是辅助信息：连靠近 0、1、2、3 步时，最终收入分别只有每千对次 1.10、1.32、1.02、0.95 笔。把对手惯性当下一步确定方向，容易扑空。','',
'3. 评估回程现金，而非只比较撞击时的货值。第 280 步起出现 554 笔实际取得，只有 61 笔在终局前至少部分兑现；第 150–239 步为 452 笔中 447 笔兑现。直接算步限更有效：6,855 个即使走最短路也送不回的机会里，有 473 笔真抢货，最后己方现金入账为零。恰能踩线返库的 18 笔真抢货中有 7 笔入账；留足至少 3 步余量的 995 笔中有 984 笔入账。候选里应保留己方直接进库、保护重货与继续采集的动作。只靠自己货空也不足以判定值得追。','',
'4. 防守时从货车自己的视角反向计算安全落点和回库路径。若有出口，观察敌方能否覆盖下一步；若所有出口将封死，应在封口前一两步改变路线。可躲开的位置并不保证未来安全，卡路径的动作也未必需要本回合真正交锋。','',
'## 时段、距离与持货条件','',*heading]
labels={'enemy_safe_exits':'目标安全出口','step':'决策步','enemy_home':'目标距自己的基地','return_streak':'目标连续向基地靠近','our_home':'我方距自己的基地','cargo':'敌人工人当前货量','cargo_ratio_percent':'我方货占对方百分数','escort_needed':'附近有受威胁的己方重货','could_bank_now':'我方带货可一步交库'}
for dimension,title in labels.items():
 for key,q in z.items():
  if key.split(' ')[0]==dimension and q['pair_roots']:
   a=key.split(' ')[1]
   if dimension=='return_streak' and a=='3..3':a='3+'
   txt.append(table(title+' '+a,q))
for key,label in [('zero_exits_no_prior_pair','当前封口／上一帧同对未入组'),('zero_exits_already_boxed','当前封口／上一帧已封口'),('zero_exits_was_one','当前封口／上一帧有1出口'),('zero_exits_was_two_plus','当前封口／上一帧有2+出口'),('scheduled_bank_slack_impossible','到步限前按最短路也送不回'),('scheduled_bank_slack_exact','按最短路送回恰好到时'),('scheduled_bank_slack_one_two','最短路送回后余1至2步'),('scheduled_bank_slack_three_plus','最短路送回后余至少3步')]:
 txt.append(table(label,cost[key]))
txt+=['','回库余量 = 299 −（当前帧 + 1）− 从所有非敌基地交点到最近己方基地的最小步数。这个上界只用行动前的布局和赛程，提前灭绝、敌人拦截、我方走较远交点都会缩小实际余量。前一帧出口只在同一对工人也满足入组条件时比较。','区间在地图种子层做 600 次重抽，每次保留同图全部席和所有重叠工人对，以入账笔数／对次数计算。区间描述这些对局中的条件频率；这些条件不是随机分派给选手的。','',
'## 有多早能看见','',
'以下用根时刻以前已知的信息入组，标签问“同一名追兵与同一货车，在随后 H 个真实联合步中是否取得一笔当局最终兑现的货”。每个 H 只取距离终结至少 H 步的根。H 扩大时，不同起点可能指向同一笔货。','',
'| 当前可见状况 | 未来 1 步／千根 | 未来 3 步／千根 | 未来 5 步／千根 | 未来 10 步／千根 |','|---|---:|---:|---:|---:|']
for key,label in [('any','全部机会'),('0 exits','零出口'),('1 exit','一个出口'),('2+ exits','至少两个出口'),('overlap first turn','刚进入可接触范围'),('overlap second turn','连续可接触第二步'),('overlap 3-5 turns','连续可接触 3–5 步'),('overlap 6+ turns','连续可接触至少 6 步')]:
 vals=[lead['prospective'][f'cash_{h}'][key] for h in (1,3,5,10)]
 txt.append('| '+label+' | '+' | '.join(f"{rate(v['observed_positive'],v['eligible']):.2f}" for v in vals)+' |')
b=lead['retrospective']['banked']
txt+=['',f"反向查看 1,009 笔最终兑现：把撞后第一帧定义为 T，T−1 即下命令前局面。同一追兵与目标在 T−10、T−5、T−3、T−1 已符合几何与持货条件的笔数依次是 {b['10']['same_pair_was_threat']}、{b['5']['same_pair_was_threat']}、{b['3']['same_pair_was_threat']}、{b['1']['same_pair_was_threat']}。当时即完全封口的只有 {b['10']['same_pair_then_zero_exits']}、{b['5']['same_pair_then_zero_exits']}、{b['3']['same_pair_then_zero_exits']}、{b['1']['same_pair_then_zero_exits']}。这支持远期观察、多次重算，临近时才承诺具体落点。",'',
'## 己方交付代价与对手差异','',*heading]
for key,label in [('own_bank_available','自己带货且有一步进库选项'),('own_bank_and_at_least_one_bank_crossing','存在同时能进自己基地的交点'),('own_bank_but_all_crossings_require_leaving_delivery','所有交点都会放弃本步进库'),('actually_selected_crossing_away_from_own_direct_delivery','实际放弃本步进库并选择交点')]:txt.append(table(label,cost[key]))
txt+=['',f"最后一行覆盖的己方原持货累计 {cost['actually_selected_crossing_away_from_own_direct_delivery']['own_pre_cargo_exposure']:,.2f} 碳。这是工人对次的风险暴露，不能加总成真实交付损失：同工同一步可能对多个目标，且之后仍可返库。是否因脱离护送造成队友损失，需要另做双方响应的分支对照。",'',
'| 对手 | 零出口：兑现／机会 | 零出口／千根 | 至少两出口：兑现／机会 | 至少两出口／千根 |','|---|---:|---:|---:|---:|']
ranknames=['榜外','Master5s','Master6','it720-preview','M9XJ-630','ClaudeFable5.2-v7']
for i,name in enumerate(ranknames):
 q=cost[f'opponent_{i}_boxed'];v=cost[f'opponent_{i}_free']
 txt.append(f"| {name} | {q['cashed']}/{q['pair_roots']:,} | {fmt_e(q)} | {v['cashed']}/{v['pair_roots']:,} | {fmt_e(v)} |")
txt+=['','对手分组的地图区间、己方排名分组、敌我距离与逃路联合分层均保存在数据文件中。','',
'## 五种实战走法','',
'坐标以原回放 15×15 世界坐标记，(x,y)=(格号 mod 15, 格号整除 15)，上移让 y 减一，边界环绕。表中箭头为“该帧实际选择的下一步”；对方当步实际命令只在事后呈现，在线比较器不能读取。失败与成功均按该类别的大货样例展示。','']
names=['停','上','右','下','左'];act={'UP':'上','RIGHT':'右','DOWN':'下','LEFT':'左',None:'停'}
xy=lambda p:f'({p%15},{p//15})'
spoken={'banked':'拿到且兑现','won_unbanked':'赢了碰撞、没有入账','refunded':'撞到敌基地、货全还给对方','missed':'两人交换格，错过原地截获','forecast_wrong':'选了交点，对手从另一格离开'}
for k in ('banked','won_unbanked','refunded','missed','forecast_wrong'):
 e=cases[k][0];r=e['root'];frame=r['step']+1
 txt+=[f"### {spoken[k]}",'',f"{e['id'][:8]}，{e['hunter_name']} 对 {e['victim_name']}。决策帧 {r['step']}，下一帧 T={frame}。最后下令时双方距离 {r['distance']}，目标重 {r['their_cargo']:.2f}，还剩 {r['safe_exits']} 个安全落点；原动作实得 {r['captured']:.2f}，最终入账 {r['banked']:.2f}，立即交还敌基地 {r['enemy_base_refund']:.2f}。",'','| 帧 | 追兵坐标、货和下一步 | 目标坐标、货和下一步 | 目标安全出口 |','|---|---|---|---:|']
 def w(row,chosen):
  return f"{xy(row[0])}／{row[1]:.1f}／{act[chosen]}" if row else '已离场或尚未出生'
 for tr in e['trace']:
  if tr['relative_contact'] not in (-10,-5,-3,-1,0):continue
  p0=tr['relative_contact'];label='T' if p0==0 else f'T{p0}'
  txt.append(f"| {label} | {w(tr['hunter'],tr['hunter_next'])} | {w(tr['victim'],tr['victim_next'])} | {len(tr['safe_target_cells']) if 'safe_target_cells' in tr else '—'} |")
 if k=='banked':txt+=['','T−3 时双方已相距两格，对手仍能往两个安全格走；到 T−1 安全出口变零。封口和最后方向都出现后，才形成高收益的碰撞。']
 if k=='won_unbanked':txt+=['','这是最后一个联合步。对手少了这份在途资产，但不能直接认为自己多了等额终局现金；要另核对手本可兑现的金额。']
 if k=='refunded':txt+=['','Master6 用更轻的工人赢了撞击，交锋地点是对方基地。比赛规则按基地所有者结算，所以这里没有己方现金收益。']
 if k=='missed':
  cf=e['fixed_current_opponent_branch'];txt+=['',f"固定其余所有真实命令，只把追兵改成{act.get(cf['alternate'],'停')}，仓内 FastCarbon 复跑当步在 {xy(cf['landing'])} 实收 {cf['transferred_now']:.2f} 碳，当步没有入账。保留原动作的对照逐值复现了真实下一帧。该反例用来检验是否漏列“停留”候选，不能事先认定对手必走原路。"]
 if k=='forecast_wrong':txt+=['','虽然我方走入两人都能进的一格，但目标在真实对局里去了另一个格子；上一步看到目标在射程内，不等于这一步能接到货。']
 txt+=['']
txt+=['## 训练接口','',
'每行对应一个工人对和一个决策时刻。输入只能取字段表中 pre_action_columns，以及补充表的 visible_consecutive_overlap_turns；不把地图索引、工人 ID、真实命令或未来结果作为策略输入。可用标签分成“是否同格／取得”“税后最终入账金额”“未兑现／终局仍持有”和 1、3、5、10 步的同人追踪。用现金目标和夺走但未兑现的目标分训，避免模型以为所有成功碰撞都会带来同额现金。','',
'未选交点、已选但扑空、已取得而未兑都需保留。自然轨迹只给出当时策略的结果；把未执行动作直接标成劣行动会引入选择偏差。若要估计改动作的因果收益，须以独立的双方策略续局验证。所有这些图都已用于研究，不得再转作正式未见地图。','',
'[逐根字段与分层](raid_timing.json.gz) · [90 万行身份与数值](raid_timing_rows.npz) · [前瞻窗口数据](raid_timing_lead.json) · [对应逐根标签](raid_timing_lead.npz) · [交付代价与对手分组](raid_timing_cost.json) · [15 个大货实例的轨迹和三条错过截获的当步复跑](raid_timing_examples.json.gz)','']
(HERE/'raid_timing.md').write_text('\n'.join(txt))
