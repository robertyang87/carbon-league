"""Verify signed score-action data and summarize credit, including losses and both seats."""
import math
import numpy as np
from experiment import FIT, HERE, ROOT, read, sha, write

def run():
    f=read(HERE/'fit.json'); protocol=read(HERE/'protocol.json')
    summaries=[]
    for g in read(HERE/'scores.json'):
        seed,s=g['seed'],g['seat'];other=1-s
        control=next(c for c in protocol['historical_controls'] if (c['seed'],c['seat'])==(seed,s))
        origpath=ROOT/control['raw'];newpath=HERE/'games/candidate'/f'raw_{seed}_{s}.json.gz'
        sources=[origpath,newpath]
        if any(sha(x)!=d for x,d in zip(sources,[control['sha256'],g['raw']])):raise ValueError('unbound game')
        info={}
        for name,p,score in zip(('control','train'),sources,(control['scores'],g['score'])):
            raw=read(p)
            wall=read(p.with_name(f'wall_{seed}_{s}.json.gz'))
            receipt=read(p.with_name(f'receipt_{seed}_{s}.json'))
            if (wall['raw_sha256']!=sha(p) or receipt['game']['raw_sha256']!=sha(p) or
                receipt['official']['official_reward_by_seat']!=score or len(wall['calls'])!=raw['steps']):
                raise ValueError('timing/score drift')
            info[name]=dict(score=score,own=score[s],opp=score[other],margin=score[s]-score[other],
                steps=raw['steps'],extinction=raw['extinct_by_seat'],depth8=sum(c['max_depth']==8 for c in wall['calls']),
                original_wall=sum(c['public_wall_seconds'] for c in wall['calls']),
                original_cpu=sum(c['process_cpu_seconds'] for c in wall['calls']),
                cutoffs=sum(c['kind']=='depth_zero_return_base' for c in wall['calls']), tape=sha(p))
        trace=[read(HERE/'traces'/f'{seed}_{s}'/f"{a['t']:03d}.json") for a in g['events']]
        supported=[e for e in trace if e['supported']]
        info.update(seed=seed,seat=s,supported=len(supported),sampled_non_greedy=sum(t['choice'] for t in supported),
            action_credits=dict(collector=sum(e['role']==0 for e in supported),planter=sum(e['role']==1 for e in supported)))
        summaries.append(info)
    stable=f.get('gate')=='eligible_for_old_root_response'
    def metric(name):
        v=[r[name] for r in summaries]
        return dict(wins=sum(e['margin']>0 for e in v),draws=sum(e['margin']==0 for e in v),losses=sum(e['margin']<0 for e in v),
            margin=float(np.mean([e['margin'] for e in v])),own=float(np.mean([e['own'] for e in v])),opp=float(np.mean([e['opp'] for e in v])),
            original_wall=float(np.sum([e['original_wall'] for e in v])),d8=sum(e['depth8'] for e in v),cutoffs=sum(e['cutoffs'] for e in v))
    totals={n:metric(n) for n in ('train','control')}
    write(HERE/'batch_audit.json',dict(protocol=sha(HERE/'protocol.json'),fit_sha=sha(HERE/'fit.json'),maps=summaries,totals=totals))
    a,b=totals['train'],totals['control']
    lines=['# 用 Sapling 对原版真实整局的回报更新动作网络','',
           f"四张已曝光开发图各换席一次，采样中的首胜版 **{a['wins']} 胜 {a['draws']} 平 {a['losses']} 负**，平均官方分差 **{a['margin']:+.2f}**。相同地图和席位的历史确定性首胜版 **{b['wins']} 胜 {b['draws']} 平 {b['losses']} 负**，平均分差 **{b['margin']:+.2f}**。下表八个终局均用于训练。导出的贪心学生需另测。",'',
           '| 地图 | 席 | 实际采样／选第二令 | 采样方官方分 | 原版官方分 | 分差 | 历史同席分差 | 原版公开调用秒 | 原版搜到八层的步 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        q=r['train'];lines.append(f"| {r['seed']} | {r['seat']} | {r['supported']}／{r['sampled_non_greedy']} | {q['own']:.2f} | {q['opp']:.2f} | {q['margin']:+.2f} | {r['control']['margin']:+.2f} | {q['original_wall']:.2f} | {q['depth8']} |")
    lines+=['','原版沿用自然温度和 0.58 秒搜索预算。每次出令都留有真实墙钟、CPU、深度与中止记录，并以动作链和官方积分复核绑定全场。历史参照独立运行，双方时钟和原版抽样可有差异。地图数为四。','',
            '每盘只在事前固定的八个时刻尝试探索。对一个工人的两个合法动作分别完整重解队令，核实安全处理后均可实发且彼此不同，再按两种命令的精确概率抽样。全队累计回报来自官方胜负与终局分差。减去该地图和席位的历史参照后，将差值乘以全场每次实采的 log 概率梯度并求和。一个终局可能同时给多次早中晚动作分配同一方向的信用，归因方差较大。参照的采样噪声也会进入梯度。[预登记](protocol.json) · [真实命令及概率](traces) · [官方逐局证据](games/candidate) · [比赛明细](batch_audit.json)','']
    if 'selected' in f:
        ch=f['held_public_command_changes']
        lines+=[f"真正进入下一包的是 Sapling 工人读出矩阵及偏置，发生数值变化 {f['parameter_count']} 项。FP32 更新 L2 范数 {f['step_l2']:.5f}，训练时任一实际二选一 logit 差最大改变 {f['max_abs_paired_logit_shift']:.5f}。留出的 343/344 两图按冻结时刻复演，席 0/1 各有 {ch['0']}／{ch['1']} 个联合命令相对首胜版改变。实际公开部署包 SHA-256 为 `{f['selected']}`，长 {f['file_bytes']:,} 字节，只含原贪心推理入口及新张量。按预登记判据，"+('可进入旧图的原版真实响应复验。' if stable else '停止在离线门。'),'',
                '[训练器](fit.py) · [梯度与训练结果](fit.json) · [学生权重](candidate.npz) · [普通部署包](candidate.py) · [公开命令对照](held_commands.json.gz)']
    else:lines+=['有效探索量未达到预登记最小值，本次没有导出新包。[训练门](fit.json)']
    if (HERE/'response/result.json').exists():
        rr=read(HERE/'response/result.json');lines+=['','## 未用于更新的两张开发图','',
             '每张固定图仅干预第一处符合登记的真实不同命令，随后双方均使用首胜版作为本方的续弈策略。']
        for r in rr['rows']:lines.append(f"- {r['id']} 第 {r['t']} 步，改令后官方终差 {r['student']:+.2f}，父令 {r['parent']:+.2f}，增量 {r['delta']:+.2f}。")
        lines+=['',('两图均严格改善，可考虑在独立开发地图整场复测。' if rr['both_strictly_positive'] else '两图同时严格改善的条件未满足，停止本候选。')+'[终局与逐臂记录](response/result.json)']
    (HERE/'README.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':run()
