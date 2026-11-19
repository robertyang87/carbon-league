"""Publish the full map and opponent readout from the frozen evaluation."""
import json
from pathlib import Path
P=Path(__file__).resolve().parent
src=json.loads((P/'scores.json').read_text());bad=json.loads((P/'failure_ledgers.json').read_text());frozen=json.loads((P/'frozen.json').read_text())
labels={'old_val':'旧 Top5 留出','new_diag':'新增诊断'}
txt=['# 从当前局面预测 30 步后的现金分差变化','',
     '同一套 105 权重的小模型，在旧 21 张留出图和新增 26 张诊断图上，逐图均优于零变化、最近 10 步趋势和最近 30 步趋势的外推。其输出为双方沿当时策略再打 30 步的现金分差变化，以碳计。例：从席一看现在落后 500，预测 +200，预计 30 步后约落后 300。','',
     '| 评估集 | 图数／座位根 | 头 RMSE | 零变化 | 10 步趋势 | 30 步趋势 | 头 MAE | 非零变化方向命中 |','|---|---:|---:|---:|---:|---:|---:|---:|']
def val(r,key='head',metric='rmse'):return r['estimates'][key][metric]
for k,title in labels.items():
    m=src[k]['overall']
    txt.append(f"| {title} | {m['maps']}／{m['rows']:,} | {val(m):.2f} | {val(m,'zero'):.2f} | {val(m,'trend10'):.2f} | {val(m,'trend30'):.2f} | {val(m,'head','mae'):.2f} | {val(m,'head','sign_nonzero'):.1%} |")
txt+=['','每图等权。先在图内平均两席和各步的平方误差，再跨图平均、开平方。对照使用同一时间戳前的最近现金趋势；开局不足回看长度时用已有前史，第一帧输出零。','']
for k,title in labels.items():
    d=src[k]['head_minus_baseline_cluster_ci']['trend30']
    txt+=[f"{title}相对 30 步趋势的 RMSE 差为 {d['rmse_diff']:+.2f}，以地图为单位配对重抽的 95% 区间为 [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]。",'']
txt+=['## 输入、训练与使用','',
      '对双方分别计算相同定义的 35 个量，涵盖现金、招工能力、工人结构、运力、货量与到基地的距离、敌方轻工对返程格的威胁、载货工附近的敌树、树的余寿和上一帧树收入、周围可见碳，以及过去 3／10／30 步现金趋势。地面碳仅从自己工人当步视野更新，实见记录的权重在 12 步内线性降为零。','',
      f"以己方减对方的 35 维差量，加上早期与晚期的两组时间交互，共 105 个权重。按训练图等权 RMS 缩放，固定岭系数 0.01，无截距。训练只消费原清单 79 图、{frozen['rows']:,} 个座位根；每三步采一根，只纳入此后确有 30 个联合推进的根。权重包 {(P/'cash_margin_30.npz').stat().st_size:,} 字节，SHA-256 为 {frozen['checkpoint_sha256']}。",'',
      '旧 21 图和新 26 图的 ID、种子和内容均与训练图互斥。新图此前已做战术研究，作为迁移诊断。模型输入使用该行动席的公开观察，地图身份、选手名、未来动作和全图碳真值未进入模型。在训练图上逐步污染不可见格、加入虚假全图和身份，180 次预测最大变化为零。同步交换双方玩家列表和观察席编号、保持同一棋手视角时，预测最大变化也为零。','',
      '接口要求逐步给同一席观察。比较某个后继前先复制该根的观察历史，分别更新各分支的合法观测。这个头估计后继出发的 30 步现金变化；从根到后继的现金变化须另计。两臂应在同一时间点比较。策略更换会改变兑现方式，替代动作的收益与 Sapling 自身续局质量仍需对照检验。','',
      chr(96)*3+'python','from value import CashMargin30, History','','head = CashMargin30()','history = History()','# 每一帧，从第零步开始','history.update(observation)','expected_change = head.predict(history, observation, configuration)',chr(96)*3,'',
      '## 两席和失败局面','','| 集合／席 | 图数 | RMSE | MAE | 平均有符号误差 |','|---|---:|---:|---:|---:|']
for k,title in labels.items():
    for s,m in src[k]['by_seat'].items():
        txt.append(f"| {title}／{s} | {m['maps']} | {val(m):.2f} | {val(m,'head','mae'):.2f} | {val(m,'head','bias'):+.2f} |")
txt+=['','旧样本席零平均高估、席一平均低估，合并两席会掩盖这一点。以下从各集合中按绝对误差筛出方向也判错的不同地图。随后 30 步各项现金差取自实际游戏入账。','']
trans={'own_delivery':'双方正常存入差','tree_absorption':'双方树收入差','hire_collector':'招采集工支出差','hire_planter':'招种植工支出差','plant':'种树支出差','seize':'夺树支出差','opponent_worker_delivery_to_this_base':'敌基地交付差','base_collision_cash':'基地碰撞收入差','tree_collision_cash':'树上碰撞收入差','rounding':'舍入差'}
for k,title in labels.items():
    for e in bad[k][:2]:
        terms='、'.join(f'{trans.get(n,n)} {v:+,.2f}' for n,v in list(e['observed_cash_components'].items())[:3])
        txt.append(f"- {title} {e['id'][:8]}，席 {e['seat']}，{e['own']} 对 {e['opponent']}。第 {e['t']} 步，预测 {e['forecast']:+,.2f}，实际 {e['actual']:+,.2f}。主要现金项为{terms}。")
txt+=['','## 按对手','','| 集合 | 对手 | 图数 | 座位根 | 头 RMSE | 30 步趋势 RMSE |','|---|---|---:|---:|---:|---:|']
for k,title in labels.items():
    for opp,m in src[k]['by_opponent'].items():
        txt.append(f"| {title} | {opp} | {m['maps']} | {m['rows']} | {val(m):.2f} | {val(m,'trend30'):.2f} |")
txt+=['','## 逐图误差','','| 集合 | 图 ID | 席零／席一 | 头 RMSE | 席零／席一 RMSE | 零变化 | 10 步趋势 | 30 步趋势 |','|---|---|---|---:|---:|---:|---:|---:|']
for k,title in labels.items():
    for r in src[k]['by_map']:
        l,rhs=r['seat_head_rmse'];a,b=r['names']
        txt.append(f"| {title} | {r['id'][:8]} | {a}／{b} | {val(r):.1f} | {l:.1f}／{rhs:.1f} | {val(r,'zero'):.1f} | {val(r,'trend10'):.1f} | {val(r,'trend30'):.1f} |")
txt+=['','[冻结权重](cash_margin_30.npz) · [训练清单](frozen.json) · [预测接口](value.py) · [完整误差](scores.json) · [旧图逐根预测](old_val_predictions.tsv.gz) · [新图逐根预测](new_diag_predictions.tsv.gz) · [反例现金明细](failure_ledgers.json)','',
      '从本目录运行。OLD_REPLAYS 为原始 Top5 回放目录，NEW_REPLAYS 为新清单的缓存根目录。','',
      chr(96)*3+'bash','OPENBLAS_NUM_THREADS=1 python train.py --old "$OLD_REPLAYS"','OPENBLAS_NUM_THREADS=1 python evaluate.py --old "$OLD_REPLAYS" --new "$NEW_REPLAYS"','OPENBLAS_NUM_THREADS=1 python verify.py --old "$OLD_REPLAYS"','OPENBLAS_NUM_THREADS=1 python failures.py --old "$OLD_REPLAYS" --new "$NEW_REPLAYS"','python report.py',chr(96)*3,'']
(P/'README.md').write_text('\n'.join(txt))
