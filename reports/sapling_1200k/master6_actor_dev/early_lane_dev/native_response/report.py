"""Render the four prespecified live branch pairs from the signed evidence."""
from pathlib import Path
import gzip
import hashlib
import json

D=Path(__file__).resolve().parent


def load(p):return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    summary=load(D/'result.json'); plan=load(D/'protocol.json')
    if summary['protocol_sha256']!=digest(D/'protocol.json') or summary['completed_arms']!=8 or summary['completed_pairs']!=4:
        raise ValueError('incomplete or unregistered final')
    for filename,sha in summary['files_sha256'].items():
        if digest(D/filename)!=sha:raise ValueError(f'changed output: {filename}')
    ok=all(summary['conditions'].values()); maps=summary['by_map']
    lines=['# 双方重新出招后的封路检验','',
           ('四组根完成，候选通过本轮扩测条件。' if ok else '四组根完成，候选未通过本轮扩测条件。')+
           '实际干预仅限一名采集工在共同根的一条指令。下步起双方每拍重新决策，各自走到正式终局。每根一个新随机流，两臂共用。双席来自相同的两张开发图，样本只能说明这四个局面的后果。',
           '', '| 地图 / 席 / 步 | 原令 → 改令 | 原令终分差 | 改令终分差 | Δ分差 | ΔU | 10步敌目标交付变化 | 10步我方全队交付变化 |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    pairs=[]
    for row in plan['roots']:
        p=D/f"case_{row['seed']}_{row['seat']}/verified.json.gz"; ev=load(p);a,b=(ev['arms'][k] for k in ('parent','alternative'))
        z=ev['metrics'];pairs.append((row,ev,p))
        lines.append(f"| {row['seed']} / {row['seat']} / t{row['t']} | {row['parent_order']} → {row['alternative_order']} | {a['margin']:+,.2f} | {b['margin']:+,.2f} | {z['delta_margin']:+,.2f} | {z['delta_u']:+.6f} | {z['target_banked10_delta']:+,.2f} | {z['own_team_banked10_delta']:+,.2f} |")
    lines+=['','## 事先锁定的判据','',
            '每图等权，两席在图内平均。两图分别需要正的终局 ΔU。十步内目标敌工的实收须下降，同时己方全队正常交付须维持。四个选根的留出或返回时间均未随结果更改。','']
    for m,rec in maps.items():lines.append(f"- {m} 的双席平均 ΔU 为 {rec['delta_u']:+.6f}，平均分差变化 {rec['delta_margin']:+,.2f}。")
    gg=summary['map_equal']
    lines.append(f"- 两图等权，目标货工十步内平均交付变化 {gg['target_banked10_delta']:+,.2f}，己方全队同期变化 {gg['own_team_banked10_delta']:+,.2f}。")
    lines+=['','## 对手和货物实际去了哪里','']
    for row,ev,p in pairs:
        t=row['t'];s=row['seat'];arms=ev['arms']; bif=ev['opponent_orders_different_at']; a,b=arms['parent'],arms['alternative']
        z=ev['metrics']; order='、'.join(map(str,bif[:3])) if bif else '无'
        lines.append(f"**{row['seed']} 席 {s}，t{t}**。共同根敌工 {row['target_uid']}。两臂中，对手全队首次几次发出不同联合令的决策步为 {order}。")
        for n in ('3','5','10'):
            base,alt=a['horizons'][n],b['horizons'][n]
            def worker(z):return f"格 {z['cell']}，货 {z['cargo']:.1f}" if z else '已退出'
            lines.append(f"- {n} 步后，敌工由原令的{worker(base['target_at_end'])}变为改令的{worker(alt['target_at_end'])}；该工累计本库交付由 {base['target_banked']:.2f} 变为 {alt['target_banked']:.2f}。")
        x,y=a['horizons']['10'],b['horizons']['10']
        lines.append(f"- 十步内己方全队实际新增采集由 {x['own_team_actual_harvest']:.2f} 变为 {y['own_team_actual_harvest']:.2f}，焦点采集工被敌方获得的碰撞货物由 {x['own_focal_collision_to_enemy']:.2f} 变为 {y['own_focal_collision_to_enemy']:.2f}。原版搜索入口两臂各调用 {a['search_calls']} / {b['search_calls']} 次，两臂均已 DONE。[本根完整双账]({p.relative_to(D).as_posix()})")
        lines.append('')
    lines+=['[预注册](protocol.json) · [四根八臂总表](result.json) · [复算源](verify.py) · [工位与恢复方式](README.md)']
    (D/'RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
