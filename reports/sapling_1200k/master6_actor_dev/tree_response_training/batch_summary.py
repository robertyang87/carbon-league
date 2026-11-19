"""Render separately the source full games and the paired training interventions after both arms settle."""
import argparse
from pathlib import Path
import statistics
import json


def report(area):
    def read(name):
        return json.loads((area/name).read_text())
    labels=read('training_labels.json')['pairs']
    source=read('source_receipt.json')['games']
    cumulative=read('aggregate.json')
    filled=[r for r in labels if r['pay_intent_filled']]
    wins=sum(r['outcome']=='win' for r in source)
    draws=sum(r['outcome']=='draw' for r in source)
    n=len(source);mean=statistics.mean(r['official_margin'] for r in source)
    lines=[f"# Sapling 真实付费训练数据 · {area.name}",'',f"本批在 {len({r['seed'] for r in source})} 张新训练图上双席打完 {n} 场原版 Master3s2。赛前冻结的 v1 在这四场取得 {wins} 胜、{draws} 平、{n-wins-draws} 负，平均官方分差 {mean:+,.2f}。",'',
           '| 训练图 | Sapling 席 | 终局步 | Sapling 官方分 | 考官官方分 | 分差 |','|---:|---:|---:|---:|---:|---:|']
    for r in source:
        s=r['seat'];x=r['scores']
        lines.append(f"| {r['seed']} | {s} | {r['steps']} | {x[s]:,.2f} | {x[1-s]:,.2f} | {r['official_margin']:+,.2f} |")
    lines.extend(['',f"事前分层选中 {len(labels)} 个条件根，两臂都让原版考官实际响应到官方终局。其中 {len(filled)} 根发生真实付款，分差改善 {sum(r['target_delta_official_margin']>0 for r in filled)} 根，恶化 {sum(r['target_delta_official_margin']<0 for r in filled)} 根。两臂在根步共享同一个对手命令，随后各自在其可见局面出手。每个根只有一次配对续弈，效应也包含其后的对手反应和原版限时搜索波动。",'',
                  '| 训练图 | 席 | 根步 | 实付 | 本方终分变化 | 对手终分变化 | 付款减暂缓的分差变化 |','|---:|---:|---:|---:|---:|---:|---:|'])
    for r in labels:
        lines.append(f"| {r['seed']} | {r['seat']} | {r['t']} | {r['pay']['root_debit']:,.2f} | {r['delta_own_official_score']:+,.2f} | {r['delta_rival_official_score']:+,.2f} | {r['target_delta_official_margin']:+,.2f} |")
    lines.extend(['',f"累计有 {cumulative['distinct_maps']} 张不同训练图、{cumulative['complete_pairs']} 个同根对照，其中 {cumulative['filled_pairs']} 个实际付款。正效果涉及 {len(cumulative['positive_maps'])} 张图，负效果涉及 {len(cumulative['negative_maps'])} 张图。预定的状态干预头训练支持条件{'已经达到' if cumulative['fit_support_gate'] else '尚未达到'}。条件根上的胜负和分差不用于推算完整对局的胜率。保护开发图尚未参赛。",'',
                  '[完整的源局与考官搜索](source_receipt.json) · [官方复算](source/official_replay.json) · [全部公开候选](opportunities.jsonl.gz) · [逐根配对结果](training_labels.json) · [跨图累计与单行证据](aggregate.json)',''])
    out=area/'RESULTS.md';data='\n'.join(lines).encode()
    if out.exists() and out.read_bytes()!=data:
        raise ValueError('existing summary differs from signed training data')
    if not out.exists():
        scratch=out.with_suffix('.tmp');scratch.write_bytes(data);scratch.replace(out)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('area',type=Path)
    print(report(parser.parse_args().area))
