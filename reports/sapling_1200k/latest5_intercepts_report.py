"""Tabulate the per-seat ledger and pre-contact histories after a verified replay audit."""
import argparse
from collections import Counter
import json
from pathlib import Path


def fmt(v):
    return f'{v:,.2f}'


def make(data,first_ids):
    blocks=[('github_top5_external','GitHub 新收录，头部对外'),('live_top5_external','平台增量，头部对外'),('live_top5_mutual','平台增量，头部互打')]
    rows=data['games']
    def totals(group):
        c=Counter()
        for g in group:
            for z in g['seat_totals']:
                c.update(z)
        return c
    txt=['# 最新 Top5 回放的打劫、防守与真实入账','',
         '按实际比赛指令逐步复现每一帧的现金、单位、树、步数和完整地图碳。追踪每笔从对方流入的碰撞货，区分留在携货工身上、被敌树吸收、在银行兑现及再度丢失。按持有人变更计一次流转，同一份货被夺回时开始新一笔。','',
         '| 分组 | 局数 | 联合步 | 采集工实得敌货 | 已进己方现金 | 后续交敌树 | 被对方再次拿走¹ | 留在工人至终局 | 有安全格时选安全格² |',
         '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for code,label in blocks:
        group=[g for g in rows if g['cohort']==code]
        c=totals(group)
        again=c['collector_stolen_opponent_worker']+c['collector_stolen_opponent_cash']+c['collector_stolen_ground_or_destroyed']
        txt.append(f"| {label} | {len(group)} | {sum(g['n_steps'] for g in group):,} | {fmt(c['acquired_collector'])} | {fmt(c['collector_stolen_banked'])} | {fmt(c['collector_stolen_tax_paid_to_enemy_trees'])} | {fmt(again)} | {fmt(c['collector_stolen_carried_at_terminal'])} | {c['chose_geometrically_safe_move']:,}/{c['loaded_stay_under_threat_with_safe_move']:,} |")
    txt+=['','¹ 此列合并被对方工人抢回、流向对方现金、落地或溢出的未保住部分。逐笔 JSON 分开记录。若落地，无法在此之后区分该份碳和原有地面资源。','',
          '² 重载定义为持货至少 50。原地可被对方较轻或等重采集工触及，同时有至少一个不受这种一步触及的非敌基地落点。安全格是几何指标，未计队友占位、树税或基地招募。不同单位、同局连续决定均相关。','']
    first=totals([g for g in rows if g['id'] in first_ids])
    all_lots=[e for g in rows for e in g['lots'] if e['type']=='collector']
    heavy=[e for e in all_lots if e['mass']>=50]
    mutual=next(g for g in rows if g['cohort']=='live_top5_mutual')
    both=totals([mutual])
    txt+=['## 收货与防守的落点','',
          f"在最先按更新卡片发现的 {len(first_ids)} 张 GitHub 图上，实际取得敌货 {fmt(first['acquired_collector'])}，之后收现 {fmt(first['collector_stolen_banked'])}。完整核对仓内回放后，另有八张起初归到线上的图已在仓内；上表按最终来源分组，同一对局只计一次。",'',
          f"全部 {len(all_lots)} 笔采集工实得敌货中，{sum(e['fate'].get('banked',0)>0 for e in all_lots)} 笔至少部分进了收货方现金。{len(heavy)} 笔涉及至少 50 碳的重载工，其中 {sum(e['approach']['target_move']!='NONE' for e in heavy)} 个目标在碰撞当步已经移动，{sum(bool(e['approach']['safe_victim_landings']) for e in heavy)} 个目标当时有其他几何安全格。只瞄准停采的工人会漏掉大部分实收样本；应按双方移动预测相遇，并扣除战利品返程的敌树税。",'',
          f"唯一头部互打的一局共 {mutual['n_steps']} 个联合步。较轻工与对方重工的可达工人对次 {both['lighter_vs_loaded_reachable_pairs']}，实际落点接触 {both['chosen_landing_met_enemy']}，两席实收敌货都为零。两席合计在危险局面有安全格时，实际选了 {both['chose_geometrically_safe_move']}/{both['loaded_stay_under_threat_with_safe_move']} 次。单图结果只覆盖这次对阵。",'']
    for chosen in data['mutual_examples']:
        g=next(x for x in rows if x['id']==chosen['game']);e=chosen['lot'];f=e['fate'];p=e['beneficiary']
        if e['type']=='enemy_base_refund':
            cash_story=f"争到的 {fmt(e['mass'])} 碳在碰撞格直接存进受害方基地，获胜采集工实际获利 0。"
        else:
            cash_story=f"从对手取得 {fmt(e['mass'])} 碳。"
        txt+=['## 头部对打的逐步回溯','',
              f"{g['names'][0]}（席零）对 {g['names'][1]}（席一），终局 {g['rewards'][0]:,.2f} 比 {g['rewards'][1]:,.2f}。第 {e['t']} 帧，{g['names'][p]} 的 {e['winner']} 在格 {e['pos']} 碰到携货的 {e['victim']}，{cash_story}该笔五个主要去向依次为入账 {fmt(f.get('banked',0))}、敌树 {fmt(f.get('tax_paid_to_enemy_trees',0))}、敌方持货或现金 {fmt(f.get('opponent_worker',0)+f.get('opponent_cash',0))}、落地或销毁 {fmt(f.get('ground_or_destroyed',0))}、己方终局持有 {fmt(f.get('carried_at_terminal',0))}。",
              '', '| 距接触帧 | 抢方位置／货量 | 防方位置／货量 | 双方距离 | 抢方下一步 | 防方下一步 |','|---:|---|---|---:|---|---|']
        def worker(w):
            return f'{w[0]}／{fmt(w[1])}' if w else '已离场或尚未出生'
        for r in chosen['trace']:
            if r['relative'] in (-10,-5,-3,-2,-1,0,1,3,5) or r['state']==e['fate_last_transition'].get('banked'):
                txt.append(f"| {r['relative']:+} | {worker(r['hunter'])} | {worker(r['target'])} | {r['dist'] if r['dist'] is not None else '—'} | {r['hunter_move'] or ('原地' if r['hunter'] else '—')} | {r['target_move'] or ('原地' if r['target'] else '—')} |")
        txt.append('')
    sample=data['example']
    g=next(x for x in rows if x['id']==sample['game'])
    e=sample['lot']
    txt+=['## 外战中确实拿到现金的一笔','',
          f"{g['id'][:8]}，{g['names'][e['beneficiary']]} 对 {g['names'][1-e['beneficiary']]}。第 {e['t']} 帧于 {e['pos']} 号格抢得 {fmt(e['mass'])}，第 {e['fate_last_transition'].get('banked')} 帧前累计入账 {fmt(e['fate'].get('banked',0))}。抢前的两方位置与货量如下。",'',
          '| 接触相对帧 | 抢方位置／货量 | 防方位置／货量 | 距离 |','|---:|---|---|---:|']
    for r in sample['trace']:
        if r['relative'] in (-10,-5,-3,-1,0):
            def seen(w):return f"{w[0]}／{fmt(w[1])}" if w else '已离场或尚未出生'
            txt.append(f"| {r['relative']:+} | {seen(r['hunter'])} | {seen(r['target'])} | {r['dist'] if r['dist'] is not None else '—'} |")
    txt+=['','## 每局每席','',
          '“实得、入账、敌树、再失、终携”都只计收货方采集工确实接到的敌货。己方工人把货交给敌基地的全部金额另计在捐赠列，混战货落地和格容量溢出按失货席计。机会列是本方较轻工与敌方重工相距至多两格的工人对次，接触按双方实际落点相同计。一次群体碰撞可以包含多对。','',
          '| 局／席 | 玩家 | 机会／接触 | 实得 | 入账 | 敌树 | 再失 | 终携 | 捐敌基地 | 混战落地／溢出 | 安全／可避 |',
          '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for code,label in blocks:
        for g in rows:
            if g['cohort']!=code:continue
            for i,z in enumerate(g['seat_totals']):
                c=Counter(z);again=sum(c[k] for k in ('collector_stolen_opponent_worker','collector_stolen_opponent_cash','collector_stolen_ground_or_destroyed'))
                txt.append(f"| {g['id'][:8]}／{i} | {g['names'][i]} | {c['lighter_vs_loaded_reachable_pairs']}/{c['chosen_landing_met_enemy']} | {fmt(c['acquired_collector'])} | {fmt(c['collector_stolen_banked'])} | {fmt(c['collector_stolen_tax_paid_to_enemy_trees'])} | {fmt(again)} | {fmt(c['collector_stolen_carried_at_terminal'])} | {fmt(c['all_enemy_base_donations'])} | {fmt(c['mixed_cargo_left_on_ground'])}／{fmt(c['mixed_cargo_destroyed_by_cell_cap'])} | {c['chose_geometrically_safe_move']}/{c['loaded_stay_under_threat_with_safe_move']} |")
    c=totals(rows)
    txt+=['','## 核账口径','',
          f"上述 {len(rows)} 局中，采集工刚抢到货、却立即按所在基地规则把该货交还给受害方的总额为 {fmt(c['enemy_base_refund_at_contact'])}。凡由树、基地自动获得的敌货与采集工获得的货单列，金额分别为 {fmt(c['acquired_tree'])} 和 {fmt(c['acquired_base'])}。两席合计的混战落地 {fmt(c['mixed_cargo_left_on_ground'])}，格容量溢出销毁 {fmt(c['mixed_cargo_destroyed_by_cell_cap'])}。","",
          '可从清单中的压缩包 SHA-256 和解压后的规范 JSON SHA-256 重现本报告。逐笔数据包括归属变化、现金入账帧与扣税帧，原始清单保存完整 match ID、顺序和双方分数。[逐笔记录](latest5_intercepts.json) · [回放清单](latest5_inventory.json)','']
    return '\n'.join(txt)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('out',type=Path)
    p.add_argument('--first-manifest',type=Path,default=Path(__file__).with_name('latest5_inventory_github.json'))
    a=p.parse_args()
    first_ids={g['id'] for g in json.loads(a.first_manifest.read_text())['games']}
    a.out.write_text(make(json.loads(a.source.read_text()),first_ids))
