"""Aggregate the complete four registered replay receipts without dropping slow steps."""
import hashlib
import json
import gzip
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent
p=json.loads((H/'protocol.json').read_text())
assert hashlib.sha256((H/'replay.py').read_bytes()).hexdigest()==p['client']
cards=[]
all_steps=[]
for raw in p['raw']:
    path=H/raw.replace('raw_','timing_')
    with gzip.open(path) as f:
        doc=json.load(f)
    steps=doc.pop('steps')
    row={k:v for k,v in doc.items() if k!='ready'}
    row.update(first_step_s=steps[0]['external_s'],container_peak_mib=row['peak_rss_kib']/1024,
               python=doc['ready']['python'],numpy=doc['ready']['numpy'])
    with gzip.open(H.parent/'direct_original/games/candidate'/raw) as f:
        signed=json.load(f)
    assert hashlib.sha256((H.parent/'candidate.py').read_bytes()).hexdigest()==doc['ready']['sha256']==p['candidate']
    conditions=dict(
        count=len(steps)==doc['n']==signed['steps']==len(signed['actions'])==299,
        all_actions_match=all(r['matches_tape'] for r in steps),
        every_index=[r['t'] for r in steps]==list(range(299)),
        rss=max(r['peak_rss_kib'] for r in steps)<=512*1024,
        package_bytes=(H.parent/'candidate.py').stat().st_size<=10_000_000,
        environment=row['python'].startswith('3.12.') and row['numpy']=='1.26.4',
        measured_times=bool(doc['local_thresholds_pass']),
        pin_and_as=doc['ready']['affinity']==[2] and doc['ready']['rlimit_as']==[512*1024**2]*2)
    row['verified_local_replay_gates']=conditions
    cards.append(row)
    all_steps.extend(r['external_s'] for r in steps)
agg=dict(n=len(all_steps),maps=len({v['seed'] for v in cards}),receipts=cards,
         external_step_s={k:float(np.percentile(all_steps,q)) for k,q in [('p50',50),('p95',95),('p99',99),('max',100)]},
         matching=sum(c['matches_tape'] for c in cards),size=(H.parent/'candidate.py').stat().st_size,
         formal_resource_qualification=False,formal_live_match=False,
         local_all_thresholds_pass=len(cards)==4 and all(all(c['verified_local_replay_gates'].values()) for c in cards))
(H/'summary.json').write_text(json.dumps(agg,indent=2)+'\n')
rows=[]
for c in cards:
    rows.append(f"| {c['seed']} | {c['seat']} | {c['n']} | {c['init_external_s']:.2f} | {1000*c['first_step_s']:.1f} | {1000*c['p95_s']:.1f} | {1000*c['p99_s']:.1f} | {1000*c['max_s']:.1f} | {c['slow_steps']} | {c['sum_excess_s']:.2f} | {c['container_peak_mib']:.1f} | {c['wall_start_to_process_exit_s']:.2f} |")
(H/'README.md').write_text(f"""# 首胜 Sapling 包的持久进程计时复演

包 SHA-256 为 `{p['candidate']}`，共 {agg['size']:,} 字节。两张已曝光地图各跑双席，每次都重新启动进程，从 step 0 连续运行到历史终局。**{len(all_steps)} 次公开入口调用中有 {agg['matching']} 次与原始动作完全相同**，四份终局现金均对齐。超 1 秒的步数合计 {sum(c['slow_steps'] for c in cards)}，所有调用的最大外部耗时 {max(all_steps)*1000:.1f} ms。

| 地图 | 席 | 调用数 | 冷启 s | 第0步 ms | P95 ms | P99 ms | 最慢 ms | >1s 次数 | 超额 s | 峰值 RSS MiB | 复演全程 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
"""+'\n'.join(rows)+"""

[平台公开指南](https://117.72.210.134/static/agent-guide.md) 规定每步基准 1 秒，含通信，单步上限 3 秒，每局每席最多 3 次慢步，累计超额最多 5 秒，初始化最多 20 秒，完整赛事最多 1500 秒。单策略最多 1.5 核、512 MiB，单文件至多 10,000,000 字节。两个已知服务地址的指南本次读回 SHA-256 相同，相关原文见 [live_rules.json](live_rules.json)。

**计时边界。** 外部时钟从序列化公开观测之前，经过 Docker stdin/stdout，到父进程解析回令完成。每步都单独结算，不允许从快步积攒余额。冷启从启动 Docker 命令到 READY，包含容器启动、Python 启动、文件读取、源码编译、初次导入，没有进程内暖机。机器文件页缓存可能已热。复演全程计至进程退出，包含历史动作推进，不含现场对手推理、双策略调度或平台队列。

**资源与证书边界。** 本机 rootless Docker 的 cgroup driver 为 none。CLI 虽接受配额，容器内读回 cpu.max 为 400000/100000、memory.max 为 17179869184、cpuset 为 0–4，继承外层计算环境。策略进程在模块加载前自行锁定 CPU2 和 512 MiB 地址空间限额，主控在 CPU3，BLAS 单线程，无网络，Python 3.12.16、NumPy 1.26.4。地址空间限额不等同完整容器内存计账，峰值 RSS 是该策略进程的 ru_maxrss；单核亲和性也不能复现真实 1.5 核 CFS 的调度抖动。该结果支持这四条完整公开轨迹的每步耗时，天梯配额正式认证和 1500 秒完整互动赛事资格仍待真实部署环境验证。

**入口选择。** 仓内 `code/certify_pack.py` 将环境推进和策略放在同一子进程中，父进程发 step 索引，按内部推进耗时修正外部测量。本次需验证公开观测和动作的完整 IPC 开销，故使用只挂载策略及协议 worker 的持久容器。候选包不读取历史对方动作；只有主控据此重建下一个己席公开观测。

[预登记与原始包身份](protocol.json) · [汇总](summary.json) · [外部主控](replay.py) · [策略进程](worker.py)。四份 timing_*.json.gz 含全部逐步延迟、回令一致性及内存高水位，保留两图双席全部结果。
""")
