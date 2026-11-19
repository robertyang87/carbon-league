"""Render the paired original-opponent receipts and separate previously exposed games."""
import gzip
import json
from pathlib import Path
from statistics import mean

from run import HERE, trial
from income_target import reconstruct


def write_report():
    comparison=trial.read(HERE/'comparison.json')
    rows=comparison['new_map_seats']
    maps=comparison['map_clusters']
    progress=comparison['prior_activation_map_reported_separately']
    identity=trial.read(HERE/'protocol.json')['pack_identity']
    arms=comparison['replication_arms']
    economy={}
    timing=[]
    for arm in ('parent','learned'):
        scoreboard=trial.read(HERE/arm/'results.json')
        by_seat=[]
        for r in scoreboard['games']:
            raw=json.loads(gzip.decompress((HERE/arm/r['raw_file']).read_bytes()))
            ledger=reconstruct(r,raw)
            me=r['candidate_seat']
            counts=r['search']['depth']
            steps=sum(counts.values())
            search_times=r['search']['time']
            timing.append({'arm':arm,'seed':r['seed'],'seat':me,'affinity':r['cpu_affinity'],
                           'wall_seconds':r['wall_seconds'],'examiner_external_ms':1000*r['call_seconds_by_seat'][1-me]['mean'],
                           'search_invocations':r['search_invocations'],'search_returns':r['search_return_types'],
                           'completed_depth_histogram':counts,'completed_depth_mean':sum(int(k)*v for k,v in counts.items())/steps if steps else None,
                           'aborted':r['search']['aborted'],'search_mean_ms':1000*mean(search_times) if search_times else None})
            by_seat.append({'seed':r['seed'],'seat':me,'raw_sha256':r['raw_sha256'],'own':ledger['cash_entries_by_seat'][me],
                            'examiner':ledger['cash_entries_by_seat'][1-me],
                            'official_scores_by_seat':r['official_score_by_seat']})
        economy[arm]=by_seat
    trial.frozen_file(HERE/'income.json',economy)
    trial.frozen_file(HERE/'timing.json',{'machine_cpu_max':'400000 100000','concurrency':'parent 0/1 and learned 2/3; shared four-core cgroup including concurrent background map checks on logical CPU 4','observed_worker_cpu_percent_first_map':'98.8–99.4% at 05:16:24 UTC','observed_worker_cpu_percent_second_map':'79.3–79.8% at 05:19:43 UTC','games':timing})
    content=['# 终局回报模型在两张新图对原版 Master3s2 的复查','']
    margin=comparison['mean_new_map_margin_delta']
    low,high=comparison['illustrative_t95_map_interval']
    rule='满足' if comparison['both_map_directional_rule'] else '未满足'
    content += [f"新图共 8 局，父子各打 4 局。训练后候选取得 {arms['learned']['all']['w_d_l'][0]} 胜、{arms['learned']['all']['w_d_l'][1]} 平、{arms['learned']['all']['w_d_l'][2]} 负，"
                f"场均官方分差 {arms['learned']['all']['mean_margin']:+,.2f}。同地图同席父代场均 {arms['parent']['all']['mean_margin']:+,.2f}，候选改善 {margin:+,.2f}。"
                f"两张图上分别获得正的平均配对改善、且每个物理席 U 均提高的登记条件{rule}。[冻结方案](protocol.json) · [完整复查结果](comparison.json)",'',
                '| 新图 | Sapling 席位 | 父代分差 | 候选分差 | 候选−父代 | 父代 U → 候选 U | 候选 / 对手官方得分 |',
                '|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        l,p=r['learned'],r['parent']
        s=r['seat'];score=l['official_scores_by_seat']
        content.append(f"| {r['seed']} | {s} | {p['margin']:+,.2f} | {l['margin']:+,.2f} | {r['margin_delta']:+,.2f} | {p['U']:.4f} → {l['U']:.4f} | {score[s]:,.2f} / {score[1-s]:,.2f} |")
    content+=['',f"两席等权后，各图配对改善分别为 {maps[0]['mean_margin_delta']:+,.2f}、{maps[1]['mean_margin_delta']:+,.2f}。"]
    for s in (0,1):
        b=arms['parent']['by_physical_seat'][str(s)];a=arms['learned']['by_physical_seat'][str(s)]
        content.append(f"席 {s} 的父代 / 候选平均分差为 {b['mean_margin']:+,.2f} / {a['mean_margin']:+,.2f}，候选胜平负 {a['w_d_l'][0]}/{a['w_d_l'][1]}/{a['w_d_l'][2]}。")
    content+=['',f"地图数为 2。条件于地图效应独立且近似正态，配对改善的示意性 95% t 区间为 [{low:+,.2f}, {high:+,.2f}]，仅一个自由度。地图上的四个席位结果不能给出可靠的总体胜率。原版考官沿用搜索、温度 1 与默认随机流，配对控制了地图与席位，未控制采样流。",
              '', '## 最初的信号与本次复查','',
              f"训练图 {comparison['prior_training_map']} 仅提供四个终局奖励。随后激活复查的图 {progress['map']} 上，两席的分差改善为 {progress['delta'][0]['margin']:+,.2f}、{progress['delta'][1]['margin']:+,.2f}，均值 {progress['mean_margin_delta']:+,.2f}。该图与训练图均排除在上面的新图数字之外。所有四张图都要排除在正式考试之外。[最初开发结果](../development.json) · [地图核查](map_clearance.json)",
              '', '## 收入从哪里来','',
              '下表为新图每局平均到账。树吸收、己方工人到本方基地正常交货和碰撞现金分别计量。后者合并基地碰撞与直接落入树主账户的货物。逐局账额由两席真实命令重放，并与对应原始终局状态哈希核对。[每局双方经济帐](income.json)','',
              '| 角色 | 正常回存 | 树吸收 | 碰撞现金 | 雇工与种夺树合计支出 |','|---|---:|---:|---:|---:|']
    for arm,label in [('parent','父代'),('learned','候选')]:
        for key,side in [('own','本方'),('examiner','考官')]:
            entries=[r[key] for r in economy[arm]]
            def amount(*names):
                return mean(sum(e.get(n,0) for n in names) for e in entries)
            content.append(f"| {label}局的{side} | {amount('own_delivery'):,.2f} | {amount('tree_absorption'):,.2f} | {amount('tree_collision_cash','base_collision_cash'):,.2f} | {-amount('plant','seize','hire_collector','hire_planter'):,.2f} |")
    content+=['','## 考官搜索与机器负载','',
              '父代两席固定到逻辑核 0/1，候选两席固定到 2/3。四个考场进程与逻辑核 4 的地图检查共享同一总额 4 核的 cgroup。05:16:24 UTC 四名考场 worker 约为 98.8%–99.4% CPU，05:19:43 的窗口约为 79.3%–79.8%。考官按墙钟动态控制搜索。负载会改变搜索预算，不能把每处比分差都归给模型。所有 8 局均纳入，下面列每局实际证据。[逐局深度直方图、调用类型及耗时](timing.json)','',
              '| 图、席 | 我方包 | 考官平均调用 ms | 完成比较的平均深度 | 8 层次数 / 比较次数 | 实场秒 |','|---|---|---:|---:|---:|---:|']
    for t in timing:
        depth=t['completed_depth_histogram']
        content.append(f"| {t['seed']}、席 {t['seat']} | {'父代' if t['arm']=='parent' else '候选'} | {t['examiner_external_ms']:.1f} | {t['completed_depth_mean']:.2f} | {depth.get('8',0)}/{sum(depth.values())} | {t['wall_seconds']:.1f} |")
    content+=['','## 身份与证据','',
              f"父代提交包 SHA-256 为 `{identity['parent']['pack_sha256']}`。终局回报候选为 `{identity['learned']['pack_sha256']}`。两者沿用 it50 / Sapling 的 75/25 混合。候选只应用在训练图冻结的四参数。复查前先将完整比赛包和权重清单逐字核对，逐场验证考官默认设置、每步搜索记录与官方计分。[已冻检查点](../learned_checkpoint.json) · [复算入口](run.py) · [父代官方回放](parent/official_replay.json) · [候选官方回放](learned/official_replay.json)"]
    (HERE/'README.md').write_text('\n'.join(content)+'\n')


if __name__=='__main__':
    write_report()
