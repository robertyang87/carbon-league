"""Rewrite the compact human-readable training report from signed official outputs."""
from collections import Counter
import gzip
from pathlib import Path
import json,statistics
here=Path(__file__).resolve().parent
def j(name):return json.loads((here/name).read_bytes())
def main():
    fit=j('selection.json');score=j('scores.json');eco=j('economic_comparison.json');binding=j('binding.json')
    rows=sorted(score,key=lambda r:(r['seed'],r['seat'],r['sign']=='minus'))
    status={'unchanged_center':'保留原中心包','tested_endpoint':'保留实际参赛端点','unplayed_shrunk_training_candidate':'保留半步训练候选，尚未实战'}[fit['retained']['classification']]
    lines=['# 用完整 Master3s2 对局训练 Sapling 的神经决策方向','',
        f"**{status}。** 对两张已曝光的原版开发图跑满八场，原版每步使用自己的可见历史、固定 0.58 秒搜索预算与温度 1 的自然采样。训练前已锁定胜场优先、同图分差一致性与收缩规则。这批只比较两个扰动方向，当前中心没有在这两张图同期复赛。[选择和包身份](selection.json) · [训练登记](protocol.json)",'',
        '| 图 | 席 | 扰动 | 本方得分 | Master3s2 得分 | 官方分差 | 结果 | 原版调用 | 原版入口墙钟秒数 | CPU |','|---|---:|---|---:|---:|---:|---|---:|---:|---:|']
    zh={'win':'胜','draw':'平','loss':'负'}
    for r in rows:
        path=here/'games'/r['driver_label']/f"wall_{r['seed']}_{r['seat']}.json.gz"
        wall=json.loads(gzip.decompress(path.read_bytes()))
        clock=sum(turn['public_wall_seconds'] for turn in wall['calls'])
        lines.append(f"| {r['seed']} | {r['seat']} | {'正' if r['sign']=='plus' else '负'} | {r['own']:,.2f} | {r['opponent']:,.2f} | {r['margin']:+,.2f} | {zh[r['result']]} | {r['actual_opponent_calls']} | {clock:.2f} | {r['cpu']} |")
    lines+=['','正负两包都以 343 上冻结的 09b41c24 为中心，固定 it50 伙伴、75/25 混合、完整联合解码与返航处理。新方向在此前两个工人神经读出分量张成的平面内，与上一轮实际 FP32 变化的余弦约为 −4.1×10⁻¹⁰。用 343 已记录的两席可见历史定半径，这些动作没有进入训练回报。±0.30 均使 7/200 个采样局面实际改令，正负之间三个步段分别有 1、5、8 根动作不同。因此选用了第一档。每个端点只有共享工人头的 480 个矩阵系数和 5 个偏置在 int8 还原后改变。[方向、半径和联合命令](binding.json) · [地图来源](maps.json)','',
        '| 图 | 正端平均分差 | 负端平均分差 | 正减负 |','|---|---:|---:|---:|']
    for s in j('protocol.json')['seeds']:
        a,b=(fit['by_sign'][k]['maps'][str(s)]['margin'] for k in ('plus','minus'));lines.append(f'| {s} | {a:+,.2f} | {b:+,.2f} | {a-b:+,.2f} |')
    lines+=['',f"两张图上正减负的平均官方分差为 {fit['signed_plus_minus_mean_margin']:+,.2f}；保留的有符号半径为 {fit['retained_signed_radius']:+.2f}。包路径与完整 SHA-256 见选模收据。对手的随机流独立，八场只有两张物理地图。[分席结果](scores.json) · [原始动作带、官方终局、真实逐步墙钟](games/)",'','## 已冻结以外的经济解释','',
        '按真实逐步现金写入对账，100–199 步看地面树收入，200 步起看自有基地交仓收入，均按本方减原版计算。表内数字比较正负两份在同图区分两席后得到的实际经济差。双方真实行动使轨迹不同；另在每条原轨迹上读了另一端点的公开入口以核对实际改令。这两项没有参与选模。','',
        '| 图 | 中局地面树净收入的正减负 | 末局交仓净收入的正减负 |','|---|---:|---:|']
    for s in j('protocol.json')['seeds']:
        rr=[z for z in eco['pairs'] if z['seed']==s]
        d=[statistics.mean(z[k]['relative_delta_plus_minus'] for z in rr) for k in ('mid_tree_ground_own_original','late_delivery_own_original')]
        lines.append(f'| {s} | {d[0]:+,.2f} | {d[1]:+,.2f} |')
    totals={100:Counter(),200:Counter()}
    for f in eco['case_files']:
        for k,d in j(f)['period_actions'].items():totals[int(k)].update(d)
    lines+=['',f"两端在相同已记录观察下，中局共 {totals[100]['joint_different']}/{totals[100]['roots']} 根发出的联合命令不同，末局为 {totals[200]['joint_different']}/{totals[200]['roots']}。每条轨迹的实际参赛方使用签名动作带，另一个方向只作同态命令复算。[现金和改令明细](economic_comparison.json)",'']
    (here/'README.md').write_text('\n'.join(lines))
if __name__=='__main__':main()
