"""Describe signed original games only; this report performs zero agent calls."""
from collections import Counter
from pathlib import Path
import math
import statistics
import sys
import numpy as np
from native import HERE, OUT, PLAN, read, sha, require, save


def report():
    summary=read(HERE/'comparison.json')
    rows=summary['games']; total=summary['total']; p=read(PLAN)
    require(len(rows)==p['planned_games']==16,'incomplete report')
    measurements=[]; errors=[0,0]; wall_spans=[]; calls=[]
    for r in rows:
        d=OUT/'candidate';s=r['seed'];seat=r['seat'];z=read(d/f'row_{s}_{seat}.json')
        tr=read(d/f'wall_{s}_{seat}.json.gz');raw=read(d/f'raw_{s}_{seat}.json.gz')
        require(r['raw_sha256']==sha(d/f'raw_{s}_{seat}.json.gz') and r['wall_sha256']==sha(d/f'wall_{s}_{seat}.json.gz') and
                z['agent_errors_by_seat']==[0,0] and not z['audit_messages'],'signed source mismatch')
        calls.extend(tr['calls'])
        for i,x in enumerate(z['agent_errors_by_seat']):errors[i]+=x
        stamps=tr['segments']
        wall_spans.extend([a['start']['monotonic_seconds'],a['end']['monotonic_seconds']] for a in stamps)
        throttled=sum(v['end']['cgroup_cpu']['throttled_usec']-v['start']['cgroup_cpu']['throttled_usec'] for v in stamps)/1e6
        measurements.append(dict(seed=s,seat=seat,steps=len(tr['calls']),own_seconds=z['call_seconds_by_seat'][seat]['sum'],
                                 shared_group_throttled_observed_seconds=throttled,actual_depth8_calls=sum(c['max_depth']==8 for c in tr['calls']),
                                 examiner_seconds=z['call_seconds_by_seat'][1-seat]['sum'],opponent_cpu_seconds=sum(v['process_cpu_seconds'] for v in tr['calls']),
                                 searched=sum(c['kind']=='searched' for c in tr['calls']),uncompleted=sum(c['kind'] not in ('searched','single') for c in tr['calls']),
                                 opponent_depth_histogram=dict(Counter(c['max_depth'] for c in tr['calls']))))
    depth=Counter(c['max_depth'] for c in calls); kinds=Counter(c['kind'] for c in calls)
    study={'games':measurements,'opponent_action_count':len(calls),'opponent_depth_histogram':dict(depth),'opponent_return_types':dict(kinds),
           'opponent_search_only_mean_completed_depth':statistics.mean(c['max_depth'] for c in calls if c['kind']=='searched'),
           'public_original_timing_ms':dict(zip(['median','p95','max'],np.quantile([c['public_wall_seconds']*1000 for c in calls],[.5,.95,1]).tolist())),
           'opponent_cpu_hours':sum(x['opponent_cpu_seconds'] for x in measurements)/3600,
           'candidate_public_call_hours':sum(x['own_seconds'] for x in measurements)/3600,
           'two_core_batch_wall_minutes':(max(e for _,e in wall_spans)-min(s for s,_ in wall_spans))/60,
           'reported_errors_by_physical_seat':errors,'first_win_receipt':read(HERE/'first_win.json') if (HERE/'first_win.json').exists() else None}
    save(HERE/'runtime.json',study)
    out=['# Sapling 对原版 Master3s2 的八张新开发图','',
         f"冻结的 Echo4 75%＋Sapling 25% 联合策略取得 **{total['wins']} 胜、{total['draws']} 平、{total['losses']} 负**，观察胜率 **{total['observed_win_rate']:.1%}**，平均官方终局分差 **{total['mean_official_margin']:+,.2f}**。八张图各换一次席，预定的 16 局全部计分。[完整逐局表](comparison.json) · [事前登记](protocol.json)",'']
    first=study['first_win_receipt']
    if first:
        r=first['game'];s=r['seed'];seat=r['seat']
        out.extend([f"首个通过官方解释器复核的胜局是地图 **{s}**、**席 {seat}**。我方 **{r['own']:,.2f}**，Master3s2 **{r['opponent']:,.2f}**，分差 **{r['margin']:+,.2f}**，共推进 {r['steps']} 步。签收该胜时已核 {first['signed_completed_when_reported']}/16 局，之后按原批次继续。[首胜签收](first_win.json) · [该局双方全部实际指令](games/candidate/raw_{s}_{seat}.json.gz)",''])
    else:
        out.extend(['八张图都没有观察到胜局。仅按独立抽图、同图两席可相关计算，八次抽图均没有任一席胜时，“新图至少一席胜出”的单侧 95% 精确上限约为 31.2%。',''])
    out.extend(['| 开发图 | 席0 我方／考官 | 席0 分差 | 席1 我方／考官 | 席1 分差 |','|---|---:|---:|---:|---:|'])
    for s in p['seeds']:
        a,b=[r for r in rows if r['seed']==s]
        def cell(r):
            val=f"{r['own']:,.2f} / {r['opponent']:,.2f}"
            return f"[{val}](games/candidate/raw_{s}_{r['seat']}.json.gz)"
        out.append(f"| {s} | {cell(a)} | {a['margin']:+,.2f} | {cell(b)} | {b['margin']:+,.2f} |")
    ci=summary['map_bootstrap_percentile_95']; w=ci['observed_win_rate'];m=ci['mean_official_margin']
    out.extend(['',f"按物理席统计，席 0 赢 {summary['by_seat']['0']['wins']}/8、平均分差 {summary['by_seat']['0']['mean_official_margin']:+,.2f}，席 1 赢 {summary['by_seat']['1']['wins']}/8、平均分差 {summary['by_seat']['1']['mean_official_margin']:+,.2f}。整张图带两个席一起有放回重抽 50,000 次，胜率百分位区间 [{w[0]:.1%}, {w[1]:.1%}]、平均分差区间 [{m[0]:+,.2f}, {m[1]:+,.2f}]。仅有八个地图簇，区间对未见地形的外推精度有限。独立正式考卷使用另 100 张图。[逐图及区间](comparison.json) · [种子和物理初盘审计](clearance.json)",'',
        f"原版对手在自己的真实观察上出令 {len(calls):,} 次，温度 1、自然随机、原生 0.58 秒搜索墙钟。已完成至少一层的搜索 {kinds.get('searched',0):,} 拍，单候选 {kinds.get('single',0):,} 拍。已完成搜索的平均最大深度为 {study['opponent_search_only_mean_completed_depth']:.2f}。全部调用入口中位数 {study['public_original_timing_ms']['median']:.1f} ms、p95 {study['public_original_timing_ms']['p95']:.1f} ms、最大 {study['public_original_timing_ms']['max']:.1f} ms。双核依图换席，在首末签收段之间历时 {study['two_core_batch_wall_minutes']:.2f} 分钟。原版入口累计 {study['opponent_cpu_hours']:.3f} CPU 小时，我方入口累计 {study['candidate_public_call_hours']:.3f} 墙钟小时。双方内部错误均为 0。[逐局深度、耗时及成本](runtime.json) · [运行驱动](native.py)",
        '',"候选文件为 c5ad3f43，含自有 Echo4 it20 和 Sapling Master6 actor V1 两套实际推理，解码使用 75% / 25% logits 和回款修正。考官完整原件为 65f268b2。每局均由仓内官方解释器以双方实令复演至终结，计分含按规则灭绝；逐局行动、规则配置、墙钟与收据保存在同一目录。[模型来源和权重身份](../actor_partner_dev/release.json) · [绑定件](games/binding.json) · [实录目录](games/candidate)",''])
    out.extend(['## 各图真实搜索与机器状态','',
                '两席并行。每个格子依次为原考官达到深度 8 的拍数、入口进程 CPU 秒／墙钟秒、同时段整组 cgroup 累计受限进程秒。cgroup 两席的采样窗口可重叠，其增量不能相加，也无法归因给某一局的考官。保留全部地图。第三张图开局后，独立离线复盘调整至 CPU2，与其他离线任务共用 CPU2/3。','',
                '| 图 | 席0 深8拍，CPU/墙钟秒，组受限秒 | 席1 深8拍，CPU/墙钟秒，组受限秒 |','|---|---:|---:|'])
    for seed in p['seeds']:
        v=sorted((x for x in measurements if x['seed']==seed),key=lambda x:x['seat'])
        cells=[f"{x['actual_depth8_calls']}，{x['opponent_cpu_seconds']:.2f}/{x['examiner_seconds']:.2f}，{x['shared_group_throttled_observed_seconds']:.2f}" for x in v]
        out.append(f"| {seed} | {cells[0]} | {cells[1]} |")
    out.append('')
    (HERE/'RESULTS.md').write_text('\n'.join(out))
if __name__=='__main__':report()
