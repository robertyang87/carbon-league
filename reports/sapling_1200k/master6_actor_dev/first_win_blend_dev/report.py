"""Read all first-win follow-up tapes through a separate run of the unmodified official game interpreter."""
from collections import Counter
from pathlib import Path
import json
import statistics
from native import HERE,OUT,OLD,ROOT,PLAN,REG,read,require,sha,save,jobs,follow,summary

def run():
    binding=read(OUT/'binding.json');require(binding['registered_sha256']==sha(PLAN)==REG,'frozen receipt mismatch')
    for rel,h in binding['checked_driver_sha256'].items():
        require(sha(ROOT/rel)==h,'driver changed '+rel)
    results=[]
    files=[HERE/p for p in ('native.py','report.py','protocol.json','clearance.json','preflight.json','comparison.json','games/binding.json')]
    clocks=[]
    for job in jobs(binding):
        r=follow.check(job)
        results.append(r)
        d=OUT/'candidate';s=r['seed'];seat=r['seat']; documents=[d/f'{name}_{s}_{seat}.{ext}' for name,ext in
            [('raw','json.gz'),('row','json'),('receipt','json'),('ticket','json'),('wall','json.gz')]]
        files.extend(documents);z=read(documents[4]);rv=read(documents[1])
        require(rv['agent_errors_by_seat']==[0,0] and not rv['audit_messages'],'internal error')
        seg=z['segments']
        clocks.append(dict(seed=s,seat=seat,n=len(z['calls']),depth=dict(Counter(c['max_depth'] for c in z['calls'])),
                           cpu_seconds=sum(c['process_cpu_seconds'] for c in z['calls']),wall_seconds=sum(c['public_wall_seconds'] for c in z['calls']),
                           sapling_seconds=rv['call_seconds_by_seat'][seat]['sum'],returns=dict(Counter(c['kind'] for c in z['calls'])),
                           cgroup_throttled=sum(x['end']['cgroup_cpu']['throttled_usec']-x['start']['cgroup_cpu']['throttled_usec'] for x in seg)/1e6,
                           first_start=seg[0]['start']['monotonic_seconds'],last_end=seg[-1]['end']['monotonic_seconds']))
    a=read(HERE/'comparison.json')
    require(json.loads(json.dumps(summary(sorted(results,key=lambda r:(r['seed'],r['seat'])))))==a,'fresh or historical totals differ')
    save(HERE/'runtime.json',clocks)
    files.append(HERE/'runtime.json')
    fresh=a['fresh'];ex=a['exposed_maps'];x=fresh['total'];y=ex['total'];p=read(PLAN)
    lines=['# 更换 Sapling 权重后的原版实战','',
           f"在四张新开发图换席对完整 Master3s2，融合训练版取得 **{x['wins']} 胜、{x['draws']} 平、{x['losses']} 负**，平均官方分差 **{x['mean_official_margin']:+,.2f}**。同一个单文件另在两张已经用过的图换席得到 {y['wins']} 胜、{y['draws']} 平、{y['losses']} 负，平均分差 {y['mean_official_margin']:+,.2f}。[事前登记](protocol.json) · [官方逐局结果](comparison.json)",'']
    for group,obj in [('fresh',fresh),('exposed',ex)]:
        f=HERE/f'first_{group}_win.json'
        if f.exists():
            d=read(f);r=d['game'];require(r in obj['games'] and r['result']=='win','first win differs')
            files.append(f)
            typ='新图' if group=='fresh' else '已曝光对照图'
            lines+= [f"首个{typ}胜局为 {r['seed']} 席 {r['seat']}，我方 {r['own']:,.2f}、原版 {r['opponent']:,.2f}，分差 **{r['margin']:+,.2f}**，{r['steps']} 步。签收时本轮全部 {d['official_checks_completed']}/12。[胜局收据]({f.name})",'']
    lines+=['| 地图 | 组别 | 席0 我方／考官 | 席0 分差 | 席1 我方／考官 | 席1 分差 |','|---|---|---:|---:|---:|---:|']
    for s in p['seeds']+p['historical_control_maps']:
        rows=[r for r in results if r['seed']==s];parts=[]
        for r in rows:
            parts.extend([f"[{r['own']:,.2f}／{r['opponent']:,.2f}](games/candidate/raw_{s}_{r['seat']}.json.gz)",f"{r['margin']:+,.2f}"])
        lines.append('| '+' | '.join([str(s),'新图' if s in p['seeds'] else '已曝光']+parts)+' |')
    means=[r['new_minus_historical_margin'] for r in a['historical_pairs']]
    ci=fresh['map_bootstrap_percentile_95']
    lines+=['',f"新图席 0 与席 1 分别赢 {fresh['by_seat']['0']['wins']}/4 和 {fresh['by_seat']['1']['wins']}/4。按四张地图整簇重抽，胜率百分位区间 [{ci['observed_win_rate'][0]:.1%}, {ci['observed_win_rate'][1]:.1%}]，平均分差区间 [{ci['mean_official_margin'][0]:+,.2f}, {ci['mean_official_margin'][1]:+,.2f}]。仅四个地图簇，百分位区间无法反映未出现过的地图结果。四次抽图均无任一席胜时，该类地图比例的单侧 95% 精确上界约为 52.7%。",'',
            f"两张已用地图在相同地图与物理席上的四个分差，相对先前冻结的 c5ad3f43 平均变化 {statistics.mean(means):+,.2f}。对照来自历史轨迹，原考官有自然温度采样与墙钟搜索，不同批次没有配对随机流。新图四张都被排除出独立正式考卷。[对照逐局与两臂原始身份](comparison.json) · [既有考生](../first_win_dev/RESULTS.md)",'',
            f"12 局共记录 {sum(g['n'] for g in clocks):,} 次原版实际出令，原版入口 CPU {sum(g['cpu_seconds'] for g in clocks)/3600:.3f} 小时、入口墙钟 {sum(g['wall_seconds'] for g in clocks)/3600:.3f} 小时。两核换席从第一局入口到最后一局退出共 {(max(g['last_end'] for g in clocks)-min(g['first_start'] for g in clocks))/60:.2f} 分钟。下面的组限流数记录同一 cgroup，各席时间窗可能重叠。双方内部错误均为零。[逐局计时直读](runtime.json) · [回放目录](games/candidate)",'',
            '| 图 | 席0 深8拍／CPU秒／入口墙钟秒／组受限秒 | 席1 深8拍／CPU秒／入口墙钟秒／组受限秒 |','|---|---:|---:|']
    for seed in p['seeds']+p['historical_control_maps']:
        vals=[g for g in clocks if g['seed']==seed]
        lines.append('| '+str(seed)+' | '+' | '.join(f"{g['depth'].get(8,0)}／{g['cpu_seconds']:.1f}／{g['wall_seconds']:.1f}／{g['cgroup_throttled']:.1f}" for g in vals)+' |')
    lines+=['','候选在每拍实际运行自有 Echo4 与融合训练后的 Sapling，按 75% 与 25% 合成合法前缀上的动作评分，并执行相同回款修正。候选包 85c47d5d，学生权重 2c65117a，原考官 65f268b2。此前独立于这四张新图的旧两图收益是本次选它的依据。[此前训练、动作与证据](../blended_actor_arena/RESULTS.md) · [本次运行驱动](native.py)','']
    (HERE/'RESULTS.md').write_text('\n'.join(lines));files.append(HERE/'RESULTS.md')
    save(HERE/'verification.json',{'official_replayed':len(results),'completed_new_maps':len(p['seeds']),
         'artifacts':{str(f.relative_to(HERE)):{'sha256':sha(f),'bytes':f.stat().st_size} for f in sorted(files)}})
if __name__=='__main__':run()
