"""Generate the prespecified full-panel comparison from deduplicated signed game receipts."""
import statistics
from reconcile import HERE, EXAM, read, save

def main():
    seeds=read(EXAM/'protocol.json')['seeds']; arms=('parent','candidate')
    games={}; traces={}
    for seed in seeds:
        for seat in (0,1):
            for arm in arms:
                k=(seed,seat,arm);stem=f'{arm}_{seed}_{seat}'
                games[k]=read(HERE/(stem+'.json'))
                traces[k]=read(HERE/(stem+'_turns.json.gz'))
                assert traces[k]['source']==games[k]['raw_sha256']
    margins={k:g['sides']['own']['terminal_score']-g['sides']['opponent']['terminal_score'] for k,g in games.items()}
    delta={seed:[margins[seed,s,'candidate']-margins[seed,s,'parent'] for s in (0,1)] for seed in seeds}
    per_map={k:statistics.mean(v) for k,v in delta.items()}
    combined=statistics.mean(per_map.values())
    sums={}
    for arm in arms:
        for side in ('own','opponent'):
            rows=[r['sides'][side] for k,r in games.items() if k[2]==arm]
            t=[v for k,p in traces.items() if k[2]==arm for v in p['safe_rich_purchases_t90_189'] if v['side']==side]
            out={'mid_soil_t100_199':sum(r['mid_soil_t100_199'] for r in rows),
                 'safe_rich_purchases_t90_189':len(t),'safe_rich_recoup':sum(v['net25']>0 for v in t),
                 'safe_rich_net25':sum(v['net25'] for v in t)}
            for f in ('normal_deliveries','late_t200_298'):
                a={m:sum(r[f][m] for r in rows) for m in ('n','sum','large_n','large_sum')};a['mean']=a['sum']/a['n'];out[f]=a
            sums[arm+'/'+side]=out
    save(HERE/'paired_summary.json',{'paired_margin_by_seed_and_seat':delta,'paired_seat_mean_by_seed':per_map,'paired_two_seed_mean':combined,
                                      'games':len(games),'maps':len(seeds),'pooled_economics':sums})
    lines=['# 对原版 Master3s2 的经济账',
           '',f'两个新地图、双席共八局。父代和联合候选各一胜三负，双方在全部八局都存活至状态 t299。候选相对父代的双席平均分差变化在 345 图为 {per_map[seeds[0]]:+.2f}，346 图为 {per_map[seeds[1]]:+.2f}，两图等权 {combined:+.2f}。较大的单笔回库在部分场次增加，实际中盘地租下降；两项经济目标尚未同时实现。原版考官保留自然采样，同图同席控制了地图和席位，各场对手仍可能走不同的棋。[配对差和两方汇总](paired_summary.json)',
           '', '## 每局实付现金', '',
           '单元格按“本方 / 对手”排列。普通交仓只计自己的工人返回自己基地的正入库。中盘树租是 t100–199 所有持有树木从地面吸收的现金，逐棵扣除了来自敌方工人的税。每张原始现金写入都和模拟器逐步余额核对；正常回库和全程树木吸收另与场务回执核对。双方终局现金与官方重放相符。','',
           '| 图 | 席 | 版本 | 正常交仓笔数 | 平均每笔 | ≥200 笔数 | 中盘地面树租 | 终局现金 |',
           '|---|---:|---|---:|---:|---:|---:|---:|']
    for seed in seeds:
        for seat in (0,1):
            for arm in arms:
                k=(seed,seat,arm);r=games[k];a,b=[r['sides'][s] for s in ('own','opponent')]
                bank=lambda s,m:s['normal_deliveries'][m]
                lines.append(f"| {str(seed)[-3:]} | {seat} | {'父代' if arm=='parent' else '候选'} | {bank(a,'n')}/{bank(b,'n')} | {bank(a,'mean'):.2f}/{bank(b,'mean'):.2f} | {bank(a,'large_n')}/{bank(b,'large_n')} | {a['mid_soil_t100_199']:.2f}/{b['mid_soil_t100_199']:.2f} | {a['terminal_score']:.2f}/{b['terminal_score']:.2f} |")
    p,c=sums['parent/own'],sums['candidate/own'];bp,bc=p['normal_deliveries'],c['normal_deliveries'];lp,lc=p['late_t200_298'],c['late_t200_298']
    lines.extend(['','## 两项训练目标','',
        f"四个同图同席对照中，全程 ≥200 的本方实际入库依次由 6→10、7→7、8→6、8→10 笔。合计 {bp['large_n']}→{bc['large_n']} 笔，其中残局 t200–298 为 {lp['large_n']}→{lc['large_n']}，残局交仓金额 {lp['sum']:,.2f}→{lc['sum']:,.2f}。全程普通交仓金额却由 {bp['sum']:,.2f} 降至 {bc['sum']:,.2f}，均笔 {bp['mean']:.2f}→{bc['mean']:.2f}。因此增加的大单和尾段入账并没有形成全局交仓净增。", '',
        f"四局本方中盘地面树租由 {p['mid_soil_t100_199']:,.2f} 降至 {c['mid_soil_t100_199']:,.2f}，两个地图各自一席上升、一席下降。另逐棵照训练标签核验，t90–189 买入时四邻可见碳 ≥120、敌工至少 2 格远的安全富格新树由 {p['safe_rich_purchases_t90_189']} 减至 {c['safe_rich_purchases_t90_189']}，首次连续持有期最多 25 步内仅靠地面收入回本的树由 {p['safe_rich_recoup']} 减至 {c['safe_rich_recoup']}。按已买入树计算的回本比例会上升，覆盖减少。四局聚合中的这两种变化方向相反。", '',
        '## 胜局和反例','',
        '346 图席 1 的候选赢 493.63，父代同席也赢 94.62。两盘对手的现金相差 1,056.76，本方现金相差 −657.75，合成对父代的分差增益 399.01。候选整场交仓 85 笔，对手 60 笔；本方单笔均值、大单数与中盘树租仍较对手低。单盘实际收益可以还原，对手采样和后续状态都不同，无法把 399.01 归给某条改令。', '',
        '按八局统一口径，找出每局最终领先方最后一次从非领先转为持续领先的现金过账。346 席 1 的候选在 t42 动作前落后 6.73，动作后领先 91.14，此后一直在前。己方采集工 worker-2 在 t32（前 10 步）从格 184 空载向下，最近等载敌采集工距 1；t37（前 5 步）从 198 格载 65.70 向上，较轻敌工距 3；t39（前 3 步）在 183 格载 92.45 停留，较轻敌工距 4；t42 从 153 格载 111.18 向右进入 154 的基地，此时较轻敌工距 8。该步 3s2 也交仓 24.80；双方各花 30 招工。交仓差 +86.38 与土壤地租差 +11.49 解释这一步的现金转正。统一取终局胜方在残局十步内最大的现金波动，候选 t218–227 拉开 772.43，其中正常交仓净差 +663.05。[该盘的可见状态、全队实令与逐项现金](candidate_621758346_1_turns.json.gz)', '',
        '反例有两个方向。345 席 1 的候选在 t120 开头尚领先 231.91，3s2 单笔正常交仓 367.33，结算后候选落后 108.32，最终输 1,576.25。346 席 0 候选买了 9 棵安全富格树、8 棵回本，纯地租 1,274.44 高于 3s2 的 990.51；但全程本方大单仅 6 笔、对方 12 笔，仍输 1,660.54。每局均按同样定义留有前 3、5、10 步的双方可见状态与真实联合作用，现金转折属于赛后解释。[345 席 1](candidate_621758345_1_turns.json.gz) · [346 席 0](candidate_621758346_0_turns.json.gz)', '',
        '从仓库根目录依次运行本目录内的 reconcile.py、turns.py、report.py，即可重新生成现金明细、转折点和本表。文件名中的 own 与 opponent 始终相对考生席位。普通交仓保留每笔时间、工人身份与金额。逐树记录保留观测到的富度、敌距、实付树价和回收期。每场均校验原件哈希与原版裁判 DONE 结果。[逐场配对差](summary.json)'])
    (HERE/'README.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':
    main()
