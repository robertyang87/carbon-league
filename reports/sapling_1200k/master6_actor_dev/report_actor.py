"""Build the actor-only training and serving report from frozen receipts."""
import json
from pathlib import Path

D=Path(__file__).resolve().parent

def read(n):return json.loads((D/n).read_bytes())

def main():
    metrics=read('metrics.json')['results'];w=read('weight_receipt.json');recipe=w['recipe'];h=metrics['holdout']['int8']; val=metrics['val']['int8']['all']
    lines=['# Master6 多对手示范的 Sapling V1','',
           '独立留出 18 张图上，新单网对真实 Master6 操作的符合率从 51.67% 升到 54.80%。18 张图逐图均有改善。留出完整对局都是 Master6 的胜局，这个结果只度量这批地图上复现示范的能力。', '',
           '## 学了哪些对局','',
           '在冻结的 1,373 局天梯全集中按注册身份找到 Master6 的 302 张不同地图。单独使用其实际席位，把旧策略数据、开发图及前五互打从这轮主数据中剔除。余下都是对外局。训练、验证和留出之间同时核对了比赛 ID、地图种子、初始碳图散列。', '',
           '| 分组 | 地图 | 对手名字 | 示范胜/平/负 | 物理席 0 / 1 | 实际根数 |',
           '|---|---:|---:|---:|---:|---:|',
           '| 训练 | 218 | 51 | 213/1/4 | 128/90 | 6,896 |','| 验证 | 10 | 10 | 10/0/0 | 5/5 | 320 |','| 留出 | 18 | 17 | 18/0/0 | 8/10 | 576 |','| 排除 | 56 | — | 38/0/18 | — | — |','',
           '每张图分别在第 0–74、75–149、150–224、225–298 步中随机抽至多 8 个有效决策点。短局的后段自然缺席。逐步给定本席可见的碳量，重建其八步历史和每个已执行的联合动作前缀。原始命令、合法性和恢复的命令须一致。训练 6,896 根中，共有采集工 49,216、种树工 13,033、基地 2,477 个可选动作。3 张留出图与 10 张验证图曾用于另一个运输预测器的诊断，均未用于现金 V3 的训练和调参。[地图与出处](data_split.json) · [逐根与特征收据](data_receipts.json.gz) · [独立特征交叉核对](data_audit.json)', '',
           '## 权重更新', '',
           f'从现金 V3 的 1,206,476 参数热启。单核实际运行 {recipe["total_updates"]} 次 AdamW 更新，将最后一个棋盘残差块、棋子注意力、局面上下文、GRU 及工人和基地的动作头一起训练。开放 {recipe["optimizable"]:,} 个参数，{len(w["changed"])} 个张量共有 {sum(x["elements_changed"] for x in w["changed"].values()):,} 个元素发生浮点数值变化。固定最后一步导出，验证与留出在导出后统一评估。从加载数据到冻结实测 {recipe["elapsed_to_freeze_seconds"]:.1f} 秒。', '',
           '损失的 80% 为 Master6 真实动作的交叉熵，12% 约束与 V3 在相同前缀上的输出分布，8% 来自另一批 18 张训练图上已封存的 it50 软动作分布。公开示范损失按采集工、种树工、基地分配 52%、27%、21%，避免大量采集动作淹没基地。四场负局保留真实选择，没有把该局所有动作反标成错误。[预先冻结的训练日程](training_protocol.json) · [权重逐张量变化](weight_receipt.json)', '',
           '## 单网离线结果', '',
           '以下按已执行的专家前缀，比较父代及候选 int8 在至少有两个合法选项时的最高分动作。', '',
           '| 分组 | 可选动作 | 父代符合率 | 新符合率 | 改动作 | 由错变对 | 由对变错 |','|---|---:|---:|---:|---:|---:|---:|']
    for lab,r in [('验证全集',val),('留出全集',h['all']),('留出采集工',h['role']['collector']),('留出种树工',h['role']['planter']),('留出基地',h['role']['base'])]:
        lines.append(f'| {lab} | {r["slots"]:,} | {r["parent_agreement"]:.2%} | {r["candidate_agreement"]:.2%} | {r["changed"]:,} | {r["corrected"]:,} | {r["damaged"]:,} |')
    lines+=['',f'留出交叉熵从 {h["all"]["parent_ce"]:.4f} 降至 {h["all"]["candidate_ce"]:.4f}。两席分列为 46.93%→50.02% 与 55.49%→58.66%。906 个动作改变中，有 290 个动作在改前改后都与示范不符。种树工虽有提升，其绝对符合率仍最低。每局独立统计、FP32 对照与两席分母均有留档。[逐图指标](metrics.json)', '']
    if (D/'serving.json').exists():
        s=read('serving.json');v=s['sums'];lines += [f'让两个出厂包在 18 张留出图的 {s["roots"]} 个真实局面自行解码整个联合动作，每次均经封装入口各调用一次本体网络，所有可选守卫关闭。在全部 {v["active"]} 个活动单位与基地槽位（含强制动作）中，有 {v["changed"]} 个决定改变，分布在 {s["changed_roots"]} 个局面。符合示范的槽位数为 {v["expert_correct_parent"]}→{v["expert_correct_candidate"]}。张量、内嵌源码与每次推理调用均已核对。[出厂包实际执行核验](serving.json)', '']
    if (D/'exposed_loss_metrics.json').exists():
        d=read('exposed_loss_metrics.json');x=d['scores']['all'];lines += [f'另把本轮排除的 {x["maps"]} 张历史负局单列为曝光诊断。共 {d["roots"]} 个根、{x["slots"]:,} 个可选槽位，符合率 {x["parent_agreement"]:.2%}→{x["candidate_agreement"]:.2%}。历史模型或其他公开研究已接触这些地图，不能据此声称新失败局上的泛化。[历史负局诊断](exposed_loss_metrics.json)', '']
    lines+=['## 复算与制品', '',
           '恢复原回放并按数据脚本抽根。it50 辅助概率可以用仓内既有生成器和固定种子重建。训练器核验每份实际读取的分片 SHA-256，只从软教师的 18 张训练图读取概率。复跑训练时给 `--out` 和 `--pack` 提供新路径，以保留原冻结工件。[回放恢复](data_restore.py) · [真实前缀建集](data_build.py) · [辅助软目标生成器](../../../code/build_sapling_soft_it50.py) · [训练和评估](train.py)', '',
           f'- FP32 权重 SHA-256 `{w["npz_sha256"]}`。[下载权重](../../../data/weights/sapling_1200k/master6_actor_v1.npz)',
           f'- int8 单网包 SHA-256 `{w["agent_sha256"]}`。[下载单网](../../../data/weights/sapling_1200k/master6_actor_v1.py)', '']
    if (D/'candidate/official_replay.json').exists() and (D/'parent/official_replay.json').exists():
        par=read('parent/results.json')['games']; cur=read('candidate/results.json')['games']; base={g['candidate_seat']:g['margin'] for g in par};delta=[g['margin']-base[g['candidate_seat']] for g in cur]
        assert len(cur)==2 and {g['candidate_seat'] for g in cur}=={0,1} and all(g['valid'] and not g['errors'] for g in cur)
        lines[2:2]=[f"完整原版 Master3s2 的开发地图 621458920，纯单网双席 {sum(g['result']=='win' for g in cur)}/2 胜，席 0 分差 {cur[0]['margin']:+,.2f}、席 1 {cur[1]['margin']:+,.2f}。相对同图父代，第一席 {delta[0]:+,.2f}、第二席 {delta[1]:+,.2f}，两席平均变动 {sum(delta)/2:+,.2f}。四局都完成 299 次联合推进并经官方回算。当前证据不足以把纯网替换为这版权重，也不足以用这一张图估计总体胜率。[候选逐局结果](candidate/results.json) · [候选官方回算](candidate/official_replay.json) · [父代官方回算](parent/official_replay.json)", '']
    (D/'ACTOR_TRAINING.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
