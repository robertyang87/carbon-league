"""Summarize a completed intake, keeping source results, conditional training effects, and held-out status separate."""
from collections import Counter
from pathlib import Path
from statistics import mean
from select_roots import HERE,load,sha
from next_queue import next_phase

def make():
    labels=load(HERE/'training_labels.json')
    position=load(HERE/'position_coverage.json')['total']
    selection=load(HERE/'selection.json')
    source=load(HERE/'source_receipt.json')
    later=load(HERE/'next_intake.json')
    roots=labels['pairs']
    positive=labels['positive_labels'];negative=labels['negative_labels']
    waiting=next_phase(later)
    nstates=position['all_active_planter_roots']
    agree=sum(r['observed_stratum']==r['requested_stratum'] for r in roots)
    support={k:sorted({r['seed'] for r in roots if r['pay_intent_filled'] and (r['target_delta_official_margin']>0 if k=='positive' else r['target_delta_official_margin']<0)}) for k in ('positive','negative')}
    lines=['# Sapling 的付树款与留现金该怎么学','',
           f"在两张新训练图的两席对局上，取得 **{len(roots)} 对真实原版终局训练标签**。原地付款相对暂缓移动，有 {positive} 对改善终局分差、{negative} 对恶化，{len(roots)-positive-negative} 对持平。{labels['paid']} 对确实发生指定付款。四条源局、两臂每个回合的实际命令和官方终分均已保存。[逐根完整标签](training_labels.json) · [预注册](protocol.json)",'',
           '| 训练图 | 席 / 步 | V1 本来会 | 实际树款 | 单树首持净收入 | 本方终分变化 | 对手终分变化 | 训练标签 Δ分差 |',
           '|---:|---:|---|---:|---:|---:|---:|---:|']
    for r in roots:
        a=r['pay']
        source_action='支付' if r['pay_is_v1'] else '移动'
        lines.append(f"| {r['seed']} | {r['seat']} / {r['t']} | {source_action} | {a['root_debit']:.2f} | {a['first_tenure_net']:+.2f} | {r['delta_own_official_score']:+,.2f} | {r['delta_rival_official_score']:+,.2f} | **{r['target_delta_official_margin']:+,.2f}** |")
    lines+=['','训练标签的符号始终是“支付臂减暂缓臂”。当原 v1 本来会付时，另一臂只把这名工人改为一个合法的移动命令；原 v1 本来会离开时，另一臂只把它留下。双方其他同回合命令不变，包括对手已同时提交的真命令。后续双方根据各自新局面重新决策直到官方终局。因碰撞或支付条件变化而未实际扣款的根仍计入意图干预，只在实付分组中单列。每个根的两个结果来自各一条共同起始随机流，并非该行动期望价值的精确估计。[双网前缀核验](select_roots.py) · [分支与独立计分](label.py)','',
           '## 去哪里和要不要种，是两个学习问题','',
           f"四条真实源局共有 {nstates:,} 个种植工决策、{nstates*5:,} 个物理方向。在 {position['on_promising_cell']} 个决策中，工人已经站在可支付、已知邻碳覆盖当期价格、敌采至少两步且敌种至少三步的空地，实际付款 {position['on_promising_cell_realized_payment']} 次、转身离开 {position['on_promising_cell_v1_moved']} 次。另有 {position['promising_neighbor_only']} 个决策是当前格不满足而相邻落点满足，我方只选择走入该邻格 {position['chose_to_move_to_promising_neighbor']} 次。其余 {position['neither_current_nor_adjacent_promising']:,} 个根周围没有满足这一筛选的格子。该筛选只读本席已见碳与当前价格。移动一格不保证下步仍安全或合算，付款缺席也不等于漏赚。[逐根所有五方向和前 3/5/10 步线路](opportunities.jsonl.gz) · [机会分项](position_coverage.json)",'',
           '## 地图支持与迭代边界','',
           f"预设八种重货、己方树区、敌种威胁组合，{agree}/{len(roots)} 对精确覆盖指定三位组合。实际覆盖 {len(labels['context_counts'])} 种。敌方近家重货高低在这批根上与地图重合，不能区分重货效应和地图效应。两张图不能训练并验证可推广的干预头。此前用过的开发图和刚登记的两张新开发图均未进入训练，正式考试没有开局。[各根候选集合与选取顺序](selection.json) · [新图隔离](clearance.json)",'']
    for key,g in labels['map_summary'].items():
        lines.append(f"- 训练图 {key} 的 {g['pairs']} 个条件根，支付相对暂缓的平均局部终局分差为 {g['mean_local_effect']:+,.2f}。")
    rr=source['games'];source_cost=sum(r['wall_seconds'] for r in rr)
    branch_cost=sum(r[arm]['elapsed'] for r in roots for arm in ('pay','defer'))
    calls=sum(r[arm]['examiner_calls'] for r in roots for arm in ('pay','defer'))
    examples=[f"地图 {r['seed']} 席 {r['seat']} 为 {r['official_margin']:+,.2f}" for r in rr]
    lines+=['','冻结 v1 从开局对完整原版的四局分别为'+'，'.join(examples)+'。比较根上的分差改善不能换算为从开局的胜率。'+
           f"四局源游戏累计 {source_cost/3600:.3f} worker 小时，局部响应累计 {branch_cost/3600:.3f} worker 小时、{calls:,} 次后续真实考官调用。后者不含初始化、原轨迹前缀回放和事后核分。[源局与时间](source_receipt.json) · [响应核分配](response_resources.json)",'']
    if waiting:
        minimum=later['remaining_minimum']
        lines+=['## 仅登记的下一批','',
                f"实付且验证到终局的正向标签覆盖 {len(support['positive'])} 张图，负向标签覆盖 {len(support['negative'])} 张图。达到至少 8 张独立训练图、32 个实付有效对照，还缺 {minimum['new_training_maps']} 张图和 {minimum['new_actually_paid_pairs']} 个实付对照。另行登记了六张主训练图及按实付数量启动的填充图。源局、席位和窗口均有不可重复的任务 ID。这是首批签收时的去重清单，每一续组先核查最新地图曝光，再按已有证据只补缺项。两张独立开发图仍锁定。[下一批恢复清单](next_intake.json) · [只读查询入口](next_queue.py)"]
    (HERE/'RESULTS.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    make()
