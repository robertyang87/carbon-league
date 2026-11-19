"""Re-read all eight official terminals, cash accounts and observed ten-step swings on new maps."""
import gzip
import json
from pathlib import Path
from statistics import mean
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from build import digest,seal
from economy import audit


def money(x):
    return f'{x:+,.2f}'


def main():
    c=json.loads((HERE/'comparison.json').read_bytes())
    incomes=json.loads((HERE/'income.json').read_bytes())
    if c['completed_native_games']!=8 or len(incomes['games'])!=8:
        raise ValueError('complete set of eight official games required')
    totals={};windows={};gaps=[]
    output=['# 两张新开发图的同期原版复测','']
    for arm in ('parent','candidate'):
        totals[arm]=[{} for _ in c['pairs']]
        for index,pair in enumerate(c['pairs']):
            seed,seat=pair['seed'],pair['seat'];a=pair['arms'][arm]
            raw=HERE/'games'/arm/f'raw_{seed}_{seat}.json.gz'
            z=audit(raw,a['raw_sha256'])
            old=next(q for q in incomes['games'] if (q['arm'],q['seed'],q['seat'])==(arm,seed,seat))
            accounts_ok=all(actual.keys()==frozen.keys() and all(abs(actual[k]-frozen[k])<=1e-7 for k in actual)
                            for actual,frozen in zip(z['by_physical_seat'],old['by_seat']))
            if not accounts_ok or z['official_score']!=old['official_score_by_seat']:
                raise ValueError('independent income or original reward mismatch')
            totals[arm][index]={'own':z['by_physical_seat'][seat],'rival':z['by_physical_seat'][1-seat]}
            windows[f'{arm}/{seed}/{seat}']=z['windows']
    raw_windows=(json.dumps(windows,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    seal(HERE/'turning_points.json.gz',gzip.compress(raw_windows,mtime=0))
    candidate=c['arms']['candidate'];parent=c['arms']['parent']
    output.append(f'本轮复测共八局。Echo4＋Sapling 在两张新图、两个物理席位取得 **{candidate["wins"]}/4 胜**，平均官方分差 **{money(candidate["mean_margin"])}**。同期 it50＋Sapling 父代为 {parent["wins"]}/4 胜、{money(parent["mean_margin"])}。两臂都保留同一份真实 Sapling 和同一组回库残差。四个同图同席配对全部纳入，平均差 {money(c["paired_mean_margin_difference"])}。两张地图只提供两个独立簇。策略采样及搜索的自然时钟持续演化，同地图配对仍有续局噪声。[注册单](protocol.json) · [官方终分](comparison.json)')
    output.extend(['','| 地图 | 物理席 | 候选分差 | 同期父代分差 | 候选改善 |','|---|---:|---:|---:|---:|'])
    for pair in c['pairs']:
        output.append(f'| {pair["seed"]} | {pair["seat"]} | {money(pair["arms"]["candidate"]["margin"])} | {money(pair["arms"]["parent"]["margin"])} | {money(pair["delta_margin"])} |')
        s=pair['seat']; a=pair['arms']['candidate']['official_score_by_seat'];b=pair['arms']['parent']['official_score_by_seat']
        gaps.append((a[s]-b[s],a[1-s]-b[1-s]))
    dm=', '.join(f'{k} 为 {money(v)}' for k,v in c['delta_margin_by_map'].items())
    output.extend(['',f'两图各自的平均配对变化为 {dm}。候选物理席 0、1 分别赢 {candidate["wins_by_seat"][0]}/2、{candidate["wins_by_seat"][1]}/2。以地图为簇的均差标准误 {c["map_cluster_se"]:.2f}。每臂只有四个结果，无法估计总体是否接近九成胜率。此前的两张已曝光图另报 1/4 胜，未并入本表。[旧图对照](../README.md)','',
        f'按官方分数拆开，相对父代，本方平均 {money(mean(g[0] for g in gaps))}、对手平均 {money(mean(g[1] for g in gaps))}。下表列每局平均现金项变化，支出为带符号的净变动。两侧各自从期初现金对回期末现金，正式分差仍独立使用官方收益。','',
        '| 现金来源或支出 | 候选本方 − 父代本方 | 候选对手 − 父代对手 |','|---|---:|---:|'])
    names={'own_delivery':'己方工人回本方基地','opponent_worker_delivery_to_this_base':'敌工向本方基地交付','tree_absorption':'树吸收','tree_collision_cash':'树格碰撞到账','base_collision_cash':'基地格碰撞到账','hire_collector':'招聘采集工','hire_planter':'招聘种植工','plant':'新植付款','seize':'夺树付款','rounding':'逐步舍入'}
    for key,name in names.items():
        diff=[mean(a[r].get(key,0)-b[r].get(key,0) for a,b in zip(totals['candidate'],totals['parent'])) for r in ('own','rival')]
        output.append(f'| {name} | {money(diff[0])} | {money(diff[1])} |')
    output.extend(['','每条轨迹另取现金领先增加最多、收窄最多的连续十步，并在窗口开始前回看 3、5、10 步双方存货、现金、采集工、种植工与持树数。这些局面描述当场经过，不能把后续终分归因给某个单独工人。[逐局现金结算](income.json) · [八场领先转折](turning_points.json.gz) · [实际命令与官方重放](games)',''])
    seal(HERE/'RESULTS.md','\n'.join(output).encode())
    print('verified all 8 official income tapes',digest(HERE/'RESULTS.md'))


if __name__=='__main__':main()
