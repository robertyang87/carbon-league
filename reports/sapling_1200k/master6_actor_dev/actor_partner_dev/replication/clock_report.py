"""Check actual opponent wall-clock search, with per-call returns and full-game depth distributions."""
from collections import Counter
import gzip
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from build import HERE as STAGE,ROOT,digest,seal


def load(p):
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())


def require(cond,msg):
    if not cond:raise ValueError(msg)


def latency(x):
    a=np.asarray(x,dtype=np.float64)
    return {'n':len(a),'sum_s':float(a.sum()),'mean_s':float(a.mean()),'median_s':float(np.median(a)),
            'p95_s':float(np.quantile(a,.95)),'max_s':float(a.max()),'gt_580ms':int((a>.58).sum()),'gt_1s':int((a>1).sum())}


def main():
    complete=load(HERE/'comparison.json')
    require(complete['completed_native_games']==8,'native jobs incomplete')
    samples=load(HERE/'live_clock.json.gz')
    rows=[];refs={};plan=load(STAGE/'protocol.json')
    for cohort,arm,folder in [('new','parent',HERE/'games/parent'),('new','candidate',HERE/'games/candidate'),
                              ('screen','candidate',STAGE/'native/candidate'),('old_parent','parent',STAGE.parent/plan['parent_source'][str(plan['maps'][0])])]:
        if cohort=='old_parent':
            games=load(folder/'results.json')['games'];refs[str((folder/'results.json').relative_to(ROOT))]=digest(folder/'results.json')
        else:
            receipts=sorted(folder.glob('receipt_*.json'))
            require(len(receipts)==4,'wrong signed cardinality');games=[load(p)['game'] for p in receipts]
            refs.update({str(p.relative_to(ROOT)):digest(p) for p in receipts})
        for g in games:
            seed,s=g['seed'],g['candidate_seat'];tape=load(folder/g['raw_file']);ss=g['search']
            key=f'{arm}/progress_{seed}_{s}.json.gz';cfg=tape['configuration'];hist={int(k):v for k,v in ss['depth'].items()}
            calls=tape['search_calls'];rt=Counter(x[1] for x in calls);success=ss['searched']
            require(not tape['errors'] and not tape['audit_messages'] and g['valid'] and cfg['_TEMP']==1 and cfg['_SEED'] is None
                and cfg['_S_ENGINE'] and cfg['_S_ABORT']==.58 and cfg['_S_D2']==8 and cfg['_S_K']==32 and rt==g['search_return_types']
                and rt['searched']==success==sum(hist.values())==len(ss['time']) and rt['single']==ss['single']
                and len(calls)==tape['steps']==g['call_seconds_by_seat'][1-s]['n'] and [k for k,_ in calls]==list(range(len(calls))),
                'incomplete natural search record')
            times=iter(ss['time']);turns=[{'t':t,'return':state,'seconds_at_original_stats_site':next(times) if state=='searched' else None} for t,state in calls]
            partial=samples['per_arm'].get(key,[]) if cohort=='new' else [];exact=0
            for item in partial:
                j=item['transition']-1
                if 'exact_depth_if_observed' in item:
                    require(turns[j]['return']==item['last_return_label'],'live clock points to wrong turn')
                    turns[j]['depth_from_adjacent_snapshots']=item['exact_depth_if_observed'];exact+=1
            external=samples['external_call_seconds_by_physical_seat'].get(key,[]) if cohort=='new' else []
            if external:
                require(len(external)==2 and len(external[0])==len(external[1])<=len(calls),'wrong external turn count')
                for j,(a,b) in enumerate(zip(*external)):
                    turns[j]['external_examiner_s']=(a,b)[1-s];turns[j]['external_tested_agent_s']=(a,b)[s]
            rows.append(dict(cohort=cohort,arm=arm,seed=seed,seat=s,source_raw=g['raw_sha256'],physical_cpu=g['cpu_affinity'],
                config=cfg,types=dict(rt),stats_aborted=ss['aborted'],stats_changed=ss['changed'],network_backend=ss['net'],
                rollout_backend=ss['rnet'],mean_depth=sum(k*v for k,v in hist.items())/success,depth_hist=hist,
                root_candidates=ss['cands'],rolls=ss['rolls'],opponent_step_internal=latency(ss['time']),
                full_external_calls_by_seat=g['call_seconds_by_seat'],wall_seconds=g['wall_seconds'],
                exact_observed_turn_depths=exact,observed_per_turn_external=len(external[0]) if external else 0,turns=turns))
    data={'clock_semantics':'Original public entry perf_counter to stats append after simulations, searched returns only. External call includes end-of-agent work. Not the search inner block alone. Original .58 does not imply every external call is <= .58.',
          'inputs':refs,'code_sha256':digest(Path(__file__)),'live_trace_sha256':digest(HERE/'live_clock.json.gz'),'games':rows}
    seal(HERE/'clock.json.gz',gzip.compress(json.dumps(data,separators=(',',':'),sort_keys=True,allow_nan=False).encode(),mtime=0))
    fresh=[r for r in rows if r['cohort']=='new']
    md=['# 同期八局的考官墙钟和实际搜索','',
        '每次调用沿用原版入口单调时钟，温度 1、随机种子未设。预算 0.58 秒从入口计；原版按剩余时间和实测后端速度决定候选数与深度。单根候选和零深度可能直接返回。保留原始最大候选 32、深度 8、规划预算比例 0.85、安全系数 1.3。逐回合核对考官真实返回类别、官方终局与统计样本。[完整记录](clock.json.gz)','',
        '| 图/席 | 臂 | 核 | 搜索/单候选/零深 | 深8占比 | 平均深度 | 截止计数 | 改令 | 入口至搜索统计 P50/P95，ms | 完整外部调用平均/最大，ms |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in sorted(fresh,key=lambda n:(n['seed'],n['seat'],n['arm'])):
        s=r['seat'];q=r['types'];a=r['opponent_step_internal'];ext=r['full_external_calls_by_seat'][1-s]
        md.append(f"| {r['seed']}/{s} | {r['arm']} | {r['physical_cpu'][0]} | {q.get('searched',0)}/{q.get('single',0)}/{q.get('depth_zero_return_base',0)} | {r['depth_hist'].get(8,0)/q['searched']:.1%} | {r['mean_depth']:.2f} | {r['stats_aborted']} | {r['stats_changed']} | {a['median_s']*1000:.0f}/{a['p95_s']*1000:.0f} | {ext['mean']*1000:.0f}/{ext['max']*1000:.0f} |")
    md.extend(['','原版 aborted 同时计零深度退出和完成部分层后停止加深；后一类仍计成功搜索。完整深度与根候选数直方图逐局保存。成功搜索样本的次序可与逐拍返回类别一一匹配。早期记录未保留每拍深度，单靠最终直方图无法恢复。'])
    for arm in ('parent','candidate'):
        group=[r for r in fresh if r['arm']==arm];den=sum(r['types']['searched'] for r in group)
        deep=sum(r['depth_hist'].get(8,0) for r in group);exact=sum(r['exact_observed_turn_depths'] for r in group)
        md.extend(['',f'{arm} 共 {den} 次成功搜索，{deep} 次到 8 层（{deep/den:.1%}）。末段从相邻两份逐步检查点恢复了 {exact} 个拍次的实际深度或明确的无深度返回；其他拍次留空。'])
    md.extend(['','两臂在同图同席并行，每个工作进程的亲和掩码固定在逻辑核 0 或 1，下一图或下一席对调。启动时 OMP、OpenBLAS、MKL 均限 1 线程。主机总配额约 4 核，其他工作共享该配额，整机无法认定独占。实际原版搜索支付时间和达到深度均按当场结果报告。','',
        '机器数据内另有旧开发四局与历史亲本两组，分别标记 screen 和 old_parent。两组与本次同期场分列。其他机器的早期比赛没有逐步计时记录，无法据此追认其搜索量。',''])
    seal(HERE/'CLOCK.md','\n'.join(md).encode())
    print('clock audited',len(fresh),sum(len(r['turns']) for r in fresh))


if __name__=='__main__': main()
