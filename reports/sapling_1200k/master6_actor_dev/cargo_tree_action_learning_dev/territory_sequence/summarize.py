"""Summarize frozen checks across physical maps, including failed behavior and support."""
from collections import Counter,defaultdict
import gzip
import json
from pathlib import Path
import numpy as np

P=Path(__file__).resolve().parent
def load(name):
    path=P/name
    return json.loads(gzip.decompress(path.read_bytes()) if name.endswith('.gz') else path.read_bytes())

def report():
    q=load('qualification.json'); f=load('fit.json'); rows=load('held_actions.json.gz')
    rng=np.random.default_rng(78362); detail={}
    for source in ('all','Master3s2','Top5'):
        group=[r for r in rows if source=='all' or r['source']==source]
        maps=defaultdict(list)
        for r in group:maps[r['map']].append(r)
        entry={}
        for opp in ('parent','control'):
            diffs=[]
            for mm in maps.values():
                diffs.append(float(np.average([r['candidate_nll']-r[opp+'_nll'] for r in mm],weights=[r['w'] for r in mm])))
            draws=rng.choice(diffs,(10000,len(diffs)),replace=True).mean(1)
            entry[opp]=dict(candidate_minus_baseline_nll=float(np.mean(diffs)),maps_positive_for_candidate=int((np.asarray(diffs)<0).sum()),
                maps=len(diffs),bootstrap_percentile_95=list(np.quantile(draws,[.025,.975])))
        entry['actual_by_role']={}
        for role in ('COLLECTOR','PLANTER'):
            rr=[r for r in group if r['role']==role]
            entry['actual_by_role'][role]=dict(rows=len(rr),candidate_changed=sum(r['parent_cmd']!=r['candidate_cmd'] for r in rr),
                candidate_became_expert=sum(r['parent_cmd']!=r['truth']==r['candidate_cmd'] for r in rr),
                candidate_left_expert=sum(r['parent_cmd']==r['truth']!=r['candidate_cmd'] for r in rr))
        detail[source]=entry
    pr=load('pairs.json.gz'); income={}
    for typ in ('train','hold'):
        rr=[r for r in pr if r['split']==typ and all(v['prefix'] for v in r['arms'].values())]
        y=np.array([np.subtract(r['arms']['stay']['ys'],r['arms']['move']['ys']) for r in rr]);final=np.array([r['arms']['stay']['official']-r['arms']['move']['official'] for r in rr])
        income[typ]=dict(rows=len(rr),map_count=len({r['map'] for r in rr}),nonzero_H24=int(np.count_nonzero(y[:,1])),
            agreement_H24_and_official_margin_among_nonzero=int(np.sum((y[:,1]*final)>0)))
    a=load('held_native_pairs.json.gz');s=load('sequence.json.gz');top=load('sources.json')
    doc=dict(observed_samples={sp:dict(Counter(r['role'] for r in s if r['split']==sp)) for sp in ('train','val','test')},training_by_source=dict(Counter(r['source'] for r in s if r['split']=='train')),within_fit_exclusion=sorted({r['map'] for r in pr if r['split']=='hold'}),held=detail,real_pair_signal=income,held_exact_changes=q['outcome_supported_changes'])
    (P/'analysis.json').write_text(json.dumps(doc,indent=2,allow_nan=False)+'\n')
    def fmt(sp):
        return ' | '.join(f"{q['metrics'][sp][a]['nll']:.4f}" for a in ('parent','control','candidate'))
    role=detail['all']['actual_by_role']
    result='通过离线门' if q['eligible_for_response'] else '离线门未通过'
    readme=f"""# Sapling 中盘领地与回款序列学习

**{result}。** 从首胜包 `8da4b465` 训练出 H24 候选 `{q['candidate'][:12]}` 与 H4 同数据对照 `{q['control'][:12]}`。两臂更新 Sapling 的单位查询 U 与动作嵌入 V，包括偏置，共 3,616 个量化后有变化的参数。每次部署仍走原来的观测、联合解码与守卫，冻结的 it50 占四分之三，Sapling 占四分之一。成包 {f['outputs']['candidate']['bytes']:,} 字节。搜索、动作规则和时限代码均沿用父版。原版对局资格、计时与正式强度须另行实测。

## 投入与对照

从 46 份双方含原版 Master3s2 的历史比赛取原版座次，从 60 份前五回放取登记高手座次，涵盖 Master6、Master5s，以及同期另三名高手；整局回放逐步复算至终局现金。选取第 64 至 192 步，每隔八步取一个状态。同场逐个工人保留与父代先前单位指令一致且当前示范合法的例子。训练 55 张物理地图、{f['training_rows']:,} 次动作，其中原版 {doc['training_by_source']['Master3s2']:,}，前五 {doc['training_by_source']['Top5']:,}。此外 26 对相同公开根的实际两分支原版响应提供谁优谁劣的辅助监督。原生核验图、另行预留的 326、339、343–349 号开发图均排除于拟合。图级留出对本次权重更新有效，历代实验已有部分打过这些历史地图。

候选把已走动作的未来 24 步投影到账户。动作落点两格范围内，己方实际土壤租金扣投资、减对手对应项，与过去八步该区的速率比较，再计入按己方在场工人数分摊的双方回库差额的四分之一。租金剔除敌工碳税，种树及夺树以实际扣款计入。公开网络输入只取行动前己席可见历史；完整地貌只用于重建比赛与赛后结算。两臂使用同样的动作样本、配对分支、训练步数和优化器，H4 的标签窗口缩为四步。回放监督表示某动作在其历史续弈中的经济结果，包含其他工人的贡献；在单轨迹上改走别处的收益未知。到达过的高收益地盘也可能原已占优。

## 冻结后二十份回放的实际公开入口

下表是按地图等权、图内按登记 H24 权重汇总的已执行专家动作负对数概率，越低越好。15 张物理地形共 {len(rows):,} 次决策。每个最终包独立推进原始公开特征历史并重新联合解码；{q['plain_entry_crosschecks']} 次还用未经插桩入口逐项比对真实输出。

| 留出样本 | 首胜父代 | H4 对照 | H24 候选 |
| --- | ---: | ---: | ---: |
| 原版 Master3s2 | {fmt('Master3s2')} |
| 天梯前五 | {fmt('Top5')} |
| 合计 | {fmt('all')} |

H24 实际改了 {q['changed_rows']} 个工人指令，涉及 {q['changed_maps']} 张图。采集工 {role['COLLECTOR']['candidate_changed']}/{role['COLLECTOR']['rows']}，种树工 {role['PLANTER']['candidate_changed']}/{role['PLANTER']['rows']}。种树工的示范匹配从错到对 {role['PLANTER']['candidate_became_expert']} 次、从对到错 {role['PLANTER']['candidate_left_expert']} 次。保留图上曾以原版独立响应测过的两条行动分支共 {q['historical_held_pairs']} 对；新候选和父代都完整匹配曾执行的联合指令且选出不同分支的仅 {q['outcome_supported_changes']} 对。完整指令不匹配时无法将历史终局胜负归于新策略。H24 改善主要出现在前五回放。对原版历史命令，H24 的损失与父代几乎持平，H4 更低；两项总体配对差值的按图重抽 95% 区间都跨零。采集及种树两类中，原本和高手相同而被改错的实发动作均多于新改对的动作。分布概率略有改善尚未转化成更多正确指令。原版独立分叉 H24 局部经济差与官方终局差在训练 12/26、留出 11/14 同号。源别差值与不确定度详见 analysis.json。

## 复现与证据

顺序执行本目录 data.py、train.py、check.py、summarize.py。cash_audit.py 独立复算每行的两席实际地租、植物购置支出、真实回库及区域持树变化，写入 seat_cash_windows.npz，字段和精度见 cash_audit.json。CPU 使用单线程。sources.json 绑定每份轨迹、地图和游戏身份；sequence.json.gz 和 sequence.npz 可按行相连；pairs 两件列对应真实分叉两臂与收益。protocol.json 在采集及训练前固定，fit.json 记录包与权重全 SHA，qualification.json 记录门槛判读，held_actions.json.gz 包含全体留出动作。完整归档包含候选及对照的两件训练权重与两个单文件包。
"""
    (P/'README.md').write_text(readme)

if __name__=='__main__':report()
