"""Read back every closed row. Timing is a property of the signed package under this runner."""
import gzip,json
import numpy as np
from run import HERE,EXP,OLD,bench,write,encode
import hashlib

def main():
    prot=json.loads((HERE/'protocol.json').read_text());p=prot['baseline']
    rec={arm:[] for arm in ('parent','candidate')};times={arm:[] for arm in rec}
    same_env=[];table=[];changed=0;diff_slots={}
    for i,fn in enumerate(sorted(p['raw'])):
        pair={}
        for arm in rec:
            out=HERE/(arm+'_'+fn.replace('raw_','timing_'));d=json.loads(gzip.decompress(out.read_bytes()))
            assert d['arm']==arm and d['transcript_sha256']==p['raw'][fn] and d['ready']['numpy']=='1.26.4'
            assert d['ready']['python'].startswith('3.12.')
            assert d['ready']['sha256']==(prot['candidate'] if arm=='candidate' else p['candidate'])
            assert d['ready']['affinity']==[2] and d['ready']['rlimit_as']==[536870912]*2
            assert d['actions_sha256']==hashlib.sha256(encode(d['public_actions'])).hexdigest()
            sec=[s['external_s'] for s in d['steps']]
            assert [s['t'] for s in d['steps']]==list(range(len(sec))) and len(sec)==299
            assert abs(d['sum_excess_s']-sum(max(t-1,0) for t in sec))<1e-9 and d['slow_steps']==sum(t>1 for t in sec)
            assert d['max_s']==max(sec) and d['local_thresholds_pass']==(d['init_external_s']<=20 and max(sec)<=3 and d['slow_steps']<=3 and d['sum_excess_s']<=5 and d['wall_start_to_process_exit_s']<=1500)
            base={k:d[k] for k in ('seed','seat','init_external_s','max_s','p95_s','p99_s','slow_steps','sum_excess_s','peak_rss_kib','n','local_thresholds_pass','matches_tape','wall_start_to_process_exit_s')}
            base.update(step0_s=sec[0],sha256=bench.digest(out),cgroup_readback=d['ready']['cgroup'])
            rec[arm].append(base);times[arm]+=sec;pair[arm]=d
            table.append(f"| {d['seed']} | {d['seat']} | {arm} | {d['init_external_s']:.2f} | {sec[0]*1000:.1f} | {d['p95_s']*1000:.1f} | {d['p99_s']*1000:.1f} | {max(sec)*1000:.1f} | {d['slow_steps']} | {d['sum_excess_s']:.3f} | {d['peak_rss_kib']/1024:.1f} | {d['wall_start_to_process_exit_s']:.2f} |")
        a,b=pair['parent'],pair['candidate']
        assert a['final_cash']==b['final_cash'] and a['ready']['cgroup']==b['ready']['cgroup']
        same_env.append(a['ready']['cgroup']);changed+=sum(x!=y for x,y in zip(a['public_actions'],b['public_actions']))
        assert a['matches_tape']==len(a['public_actions']) and sum(x!=y for x,y in zip(a['public_actions'],b['public_actions']))==299-b['matches_tape']
        for x,y in zip(a['public_actions'],b['public_actions']):
            for uid in set(x)|set(y):
                if x.get(uid)!=y.get(uid):
                    kind='worker' if 'worker' in uid else 'base'
                    diff_slots[kind]=diff_slots.get(kind,0)+1
    assert len(rec['candidate'])==4
    totals={a:dict(steps=len(v),aggregate_p95_s=float(np.percentile(v,95)),aggregate_p99_s=float(np.percentile(v,99)),max_s=max(v),
        slow=sum(x>1 for x in v),sum_excess_s=sum(max(x-1,0) for x in v),max_init_s=max(g['init_external_s'] for g in rec[a]),
        peak_rss_mib=max(g['peak_rss_kib'] for g in rec[a])/1024,all_local_limits=all(g['local_thresholds_pass'] for g in rec[a])) for a,v in times.items()}
    caps=same_env[0]
    packaged=bench.digest(HERE.parent/'candidate.py');weight=bench.digest(HERE.parent/'candidate.npz')
    c=totals['candidate']
    old=json.loads((OLD/'live_rules.json').read_text());guide=json.loads((HERE/'live_rules.json').read_text());assert old['sha256']==guide['sha256']
    write(HERE/'summary.json',dict(candidate=packaged,weights=weight,parent=p['candidate'],guide_sha256=guide['sha256'],hosted_qualified=False,
        diagnostic_scope='Eight cold-container replays of four signed histories; both policies committed isolated actions, environment advanced the archived joint actions.',
        timing_environment= caps, changed_joint_actions=changed,changed_commands=diff_slots,totals=totals,receipts=rec))
    text=f"""# H24 首胜父代同机逐步计时

最终 H24 单文件 `{packaged}`，{(HERE.parent/'candidate.py').stat().st_size:,} 字节；训练权重 `{weight}`。同机交替运行首胜父包 `{p['candidate']}` 与 H24。地图 621758345 和 621758346 的两个席位各冷启动一次，每包 1,196 个公开请求。H24 在这批回放中超 1 秒 {c['slow']} 次，最慢 {1000*c['max_s']:.1f} ms，汇总 P95 / P99 为 {1000*c['aggregate_p95_s']:.1f} / {1000*c['aggregate_p99_s']:.1f} ms，冷启上界 {c['max_init_s']:.2f} 秒，策略进程峰值 RSS {c['peak_rss_mib']:.1f} MiB。

| 已用图 | 席位 | 包 | 冷启 s | 第0步 ms | P95 ms | P99 ms | 最慢 ms | 慢步 | 超额 s | 峰值 RSS MiB | 单席复演全程 s |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
"""+'\n'.join(table)+f"""

每个观测从序列化前计到收到并解析动作。最终 H24 和父包在相同的公开请求上有 {changed} 次联合动作不同，涉及工人命令 {diff_slots.get('worker',0)} 次和基地命令 {diff_slots.get('base',0)} 次。历史终局按原始双方动作重放，只用于检查回放完整性。表中耗时涵盖单席全轨迹，真实双方响应及比赛胜负需另测。

各臂复用同一标准镜像和原有完整 IPC 客户端，Python 3.12、NumPy 1.26.4、无网络、BLAS 单线程。主控和策略同绑 CPU 2，策略设 512 MiB 地址空间限额。容器内实读 cpu.max 为 `{caps['cpu.max']}`，memory.max 为 `{caps['memory.max']}`，可见 CPU 列表 `{caps['cpuset.cpus.effective']}`。以上结果是本地回放诊断。正式天梯资格仍需真实 1.5 核 / 512 MiB 配额和完整互动比赛的执行收据。

protocol.json 在计时前固定两包权重、客户端、worker、镜像和四局输入。八份壓缩收据包含每步耗时、返回的真实物理命令和 RSS；summary.json 绑定逐件 SHA，可用 summarize.py 重新生成。
"""
    (HERE/'README.md').write_text(text.replace('壓缩','压缩'))
if __name__=='__main__':main()
