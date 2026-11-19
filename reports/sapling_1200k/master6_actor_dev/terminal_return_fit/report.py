from collections import Counter
import gzip
import json
from pathlib import Path
from statistics import mean

import trial
from build import HERE, digest


def read(p):
    return json.loads(p.read_bytes())


def public_amount(row,k):
    return '/'.join(f'{row["final"][side][k]:,.0f}' for side in ('candidate','opponent'))


def fmt(n):
    return f'{n:+,.2f}'


def earnings_change(old,new):
    return ', '.join(f'{label}{fmt(new["final"][side][key]-old["final"][side][key])}'
                     for label,side,key in (
                         ('我方回库','candidate','own_positive_delivery_carbon'),
                         ('对手回库','opponent','own_positive_delivery_carbon'),
                         ('我方树收入','candidate','cumulative_tree_absorption_income'),
                         ('对手树收入','opponent','cumulative_tree_absorption_income')))


def write():
    fit=read(HERE/'learned_checkpoint.json')
    dev=read(HERE/'development.json')
    if fit!=trial.compute_fit() or digest((HERE/'learned_checkpoint.json').read_bytes())!=dev['checkpoint_sha256']:
        raise ValueError('fit and evaluation differ')
    p=HERE/'development_actions.json.gz'
    trace=json.loads(gzip.decompress(p.read_bytes()))
    if trace['agent_sha256']!=fit['learned_pack']['pack_sha256'] or len(trace['games'])!=2:
        raise ValueError('root attribution bound to different package or incomplete')
    arms={}
    rel={'正扰动':'train_plus','负扰动':'train_minus','开发父代':'dev_parent','开发候选':'dev_learned'}
    for name in rel.values():
        arms[name]={g['candidate_seat']:g for g in read(HERE/name/'results.json')['games']}
    for g in trace['games']:
        if g['raw_sha256']!=arms['dev_learned'][g['seat']]['raw_sha256']:
            raise ValueError('root attribution transcript mismatch')
    nwin=sum(g['result']=='win' for g in dev['candidate']);npwin=sum(g['result']=='win' for g in dev['parent'])
    nom=mean(g['margin'] for g in dev['candidate'])
    d=dev['delta']
    lines=['# 采集策略终局回报实验结果','',
           f'留出的开发图同图换席，学成候选 {nwin}/2 胜、平均官方分差 {fmt(nom)}，原父代 {npwin}/2 胜。两个席位相对父代分差分别为 {fmt(d[0]["margin"])} 和 {fmt(d[1]["margin"])}，平均 {fmt(dev["mean_margin_delta"])}；按冻结的验证条件，本次为**{"通过单图复核" if dev["replication_signal"] else "未通过单图复核"}**。这张图已进入开发数据，不能再进入正式考试。尚无多图独立胜率或平均分差证据。[原版核分](development.json)','','## 更新来自哪四局','',
           '| 训练臂 | 物理席 | 官方分差 | 终局 U | 优化回报 J |','|---|---:|---:|---:|---:|']
    for arm,label in [('plus','正扰动'),('minus','负扰动')]:
        for g in fit['training'][arm]:
            lines.append(f'| {label} | {g["seat"]} | {fmt(g["margin"])} | {g["U"]:.5f} | {g["J"]:+.5f} |')
    jdelta=[a['J']-b['J'] for a,b in zip(fit['training']['plus'],fit['training']['minus'])]
    lines+=['',f'训练图 621461120 的正负两臂共四局全负。两个席位的正负 J 差分别为 {jdelta[0]:+.5f}、{jdelta[1]:+.5f}，平均 {fit["mean_J_plus_minus"]:+.5f}。零点的一次成对扰动把四个系数更新为 [{", ".join(f"{v:+.5f}" for v in fit["theta"])}]。两臂都没有舍弃任何一席，第二席对更新量影响较大。方向固定且只采了一张图，无法单独估计每个系数的效应。[冻结参数与训练终局](learned_checkpoint.json)','',
        f'可运行学成单包 SHA-256 为 `{fit["learned_pack"]["pack_sha256"]}`，其余网络、权重和合法前缀与基线相同。[冻结策略包](../../../../data/weights/sapling_1200k/terminal_return_fit_v1.py)','','## 开发局的现金来源','',
        '| 模型 | 物理席 | 本方得分 | 对方得分 | 分差 | U | 回库（本/对） | 树收入（本/对） | 步 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm,label in [('parent','父代'),('learned','学成候选')]:
        for seat,g in arms['dev_'+arm].items():
            u,_=trial.terminal(g['margin']);scores=g['official_score_by_seat']
            lines.append(f'| {label} | {seat} | {scores[seat]:,.2f} | {scores[1-seat]:,.2f} | {fmt(g["margin"])} | {u:.5f} | {public_amount(g,"own_positive_delivery_carbon")} | {public_amount(g,"cumulative_tree_absorption_income")} | {g["actual_transitions"]} |')
    lines+=['']
    for seat in (0,1):
        old,new=arms['dev_parent'][seat],arms['dev_learned'][seat]
        lines.append(f'- 席 {seat}，{earnings_change(old,new)}。')
    lines+=['','上表的回库、树收入是经济分项，终分还含招工、种植与夺树支出等账项。原版对手采用自身未定种子的温度采样，同图的两臂可走出不同局面。单个动作的收益无法从两场终分直接归因。[父代逐局](dev_parent/results.json) · [候选逐局](dev_learned/results.json)','','## 在真实开发轨迹的每一步，残差改变了什么','',
           '| 物理席 | 检查根数 | 实际改令根 | 直接改令槽 | 前缀传导改令槽 | 联合安全回退根 | 两网络分歧中采用 PPO / Sapling / 其他 |','|---|---:|---:|---:|---:|---:|---:|']
    samples=[]
    for g in trace['games']:
        seat=g['seat'];c=Counter(g['counts'])
        totals=[sum(c['actual_'+key+'_'+role] for role in ('collector','planter','base')) for key in ('expert','sapling','other')]
        lines.append(f'| {seat} | {c["all_roots"]} | {c["residual_changed_roots"]} | {c["direct_residual_changes"]} | {c["downstream_prefix_changes"]} | {c["safety_veto_roots"]} | {totals[0]} / {totals[1]} / {totals[2]} |')
        slots=[(t['t'],v) for t in g['turns'] for v in t['changes']]
        if not slots:continue
        # Fixed descriptive examples: first changed slot and largest logged pre-action cargo.
        for kind,(t,v) in [('首次',slots[0]),('载货最高',max(slots,key=lambda row:row[1]['public_cargo']))]:
            samples.append(f'- 席 {seat} {kind}改令发生在 t{t}，`{v["uid"]}` 带货 {v["public_cargo"]:.2f}，从格 {v["physical_cell"]} 将 {v["parent"]} → {v["actual"]}，对应落点 {v["parent_geometry_on_enacted_prefix"]["landing"]} → {v["actual_geometry_on_enacted_prefix"]["landing"]}。')
    lines+=['','在候选每一局实际遇到的公开状态上，追踪器重算两网并逐槽重放，包括改令导致的后续前缀变化。所有返回动作逐项匹配真实提交记录。分歧计数按真正执行的前缀取两网各自首选，显示两者仍影响动作。表中「直接」指实际动作与同前缀、未加残差的混合首选不同；「前缀传导」指当前槽仍跟随该前缀的未加残差首选，但因前面采集工已改令，最终与父代原指令不同。','','按固定口径展示每席最早和载货最高的改令。原指令作为同一状态的反事实，这两个备选动作的两条后续轨迹未在这里重演。','',*samples,'','全部根、工人、原令/新令、载货、去向、基地距离、树耗与威胁见压缩逐步证据。[实际动作归因](development_actions.json.gz) · [复算脚本](trace_development.py)','']
    times=[g['wall_seconds'] for part in arms.values() for g in part.values()]
    lines += [f'四个训练与四个开发对局的累计 worker 墙钟为 {sum(times)/3600:.3f} 小时，单局范围 {min(times):.1f}–{max(times):.1f} 秒。核验和离线归因另计。[原版训练正臂](train_plus/official_replay.json) · [原版训练负臂](train_minus/official_replay.json) · [原版开发父代](dev_parent/official_replay.json) · [原版开发候选](dev_learned/official_replay.json)','']
    (HERE/'RESULTS.md').write_text('\n'.join(lines))


if __name__=='__main__':write()
