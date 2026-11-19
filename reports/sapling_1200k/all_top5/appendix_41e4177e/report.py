"""Compose the fixed score, replay, and per-map lookback summary from actual verified results."""
from collections import Counter
import gzip,json
from pathlib import Path
from intake import HERE,TOP

def read(p):
    raw=p.read_bytes();return json.loads(gzip.decompress(raw) if raw[:2]==b'\x1f\x8b' else raw)

def main():
    d=read(HERE/'manifest.json');iv=read(TOP/'intake_41e4177e.json');audit=read(HERE/'validation.json');cash=read(HERE/'turn_cash.json.gz');cases={r['id']:r for r in read(HERE/'cases.json.gz')['rows']};ar=d['new_archives'][0];tr=audit['original_board_transitions'];assert tr==sum(len(x['turns']) for x in cash['games'])==audit['fastsim_transitions'];assert audit['union_distinct']['id']==iv['unique_by_frozen_indexes']==1402
    wl={};details=[]
    for g in d['games']:
        ix=next(i for i,n in enumerate(g['seat_strategy_ids']) if n is not None);names=g['seat_names'];name=names[ix].removeprefix('claude-AlphaCarbon-');p,q=[g['score_by_seat'][i] for i in (ix,1-ix)];result='胜' if p>q else '负' if p<q else '平';wl.setdefault(name,Counter())[result]+=1
        sw=cases[g['id']]['economy']['swing'];phase=cases[g['id']]['economy']['stages'];factor=1 if ix==0 else -1;changes=[r['margin0_delta']*factor for r in phase];window=sw['b']-sw['a'];pos=sw['positions'];root=next(z for z in pos if z['t']==sw['a']);old=[(z['t'],z['seats'][ix]['cargo'],z['seats'][1-ix]['cargo']) for z in pos if z['t']<sw['a']];pr=sw['seats'];details.append(dict(id=g['id'],teacher=name,seat=ix,opponent=names[1-ix],teacher_score=p,opponent_score=q,outcome=result,steps=g['transitions'],seed=g['seed'],phase_changes=changes,window=[sw['a'],sw['b']],window_change=sw['margin0_delta']*factor,worker_cash=[pr[i]['bank'] for i in (ix,1-ix)],tree_cash=[pr[i]['tree'] for i in (ix,1-ix)],root_laden=[root['seats'][i]['cargo'] for i in (ix,1-ix)],earlier_laden=old))
    domains='、'.join(f'{n} {c["胜"]} 胜 {c["平"]} 平 {c["负"]} 负' for n,c in sorted(wl.items()));trans=audit['union_transitions'];lines=['# 天梯前五第十批回放', '',
            f'成绩卡冻结在 `{d["revision"][:9]}`。上一截点已有 1,386 局、115 局互打、1,271 局对外。本次 27 个新增成绩引用中，4 个比赛来自上批已归档的 Master5s 对局，另有 7 个同局跨索引重复引用。新增 16 张独立地图，全部为对外；{domains}。合计 **1,402 局，其中互打 115 局、对外 1,287 局**。这一批只进入观察与考试隔离，先前冻结的 Master6 训练/留出名单仍保留原有口径。[所有变更引用与物理席身份](../intake_41e4177e.json) · [归档清单](manifest.json)', '',
            '| 逐局回溯 | 前五选手/物理席 | 对手 | 最终现金，前五／对手 | 最大十步现金差变动，前五视角 | 窗口内正常存入，前五／对手 |','|---|---|---|---:|---:|---:|']
    for r in details:
        a,b=r['window'];x,y=r['worker_cash'];lines.append(f"| [{r['id'][:8]}]({r['id']}.md) | {r['teacher']} / {r['seat']} | {r['opponent']} | {r['teacher_score']:,.2f}／{r['opponent_score']:,.2f} | t{a}–{b}，{r['window_change']:+,.2f} | {x:,.2f}／{y:,.2f} |")
    lines+=['','## 回溯与账本','',f'每张地图都按同一规则取绝对变动最大的十步，逐局页面同时列出窗口起点、前三步、前五步和前十步的现金、工人、树与载货。所有 {tr:,} 次联合推进另有两席逐步现金八项分解、工号以及实际收款，不会把碰撞所得的在途货提前计入现金。[逐步钱账](turn_cash.json.gz) · [四个阶段的原始经济与真实截获](cases.json.gz) · [前缀和逐局回溯字段](turning_points.json)', '',
            f'用原版 Board 与 FastCarbon 分别从每张真实地图复算，整批 {audit["frames"]:,} 帧均有官方状态，比对和地图历史入账见验证件。截至本批累计 {trans:,} 次联合推进、{2*trans:,} 次行动前席决策。当前固化的 302 张 Master6 研究训练清单可继续独立复现。[逐局核验和地图排除](validation.md) · [全部行动前背景](../context_appendix_41e4177e_receipt.json) · [四阶段未来窗口的全量覆盖](future_economy_check.json) · [无损原始回放](../../../../'+ar['archive']+')',''];(HERE/'README.md').write_text('\n'.join(lines));(HERE/'turning_points.json').write_text(json.dumps(dict(covered=details,ordering='each pair in teacher/opponent order, earlier_laden is (pre-action t, teacher load, opponent load)'),ensure_ascii=False,indent=1)+'\n')
    strong=sum((r['worker_cash'][0]-r['worker_cash'][1])*r['window_change']>0 and abs(r['worker_cash'][0]-r['worker_cash'][1])>=.5*abs(r['window_change']) for r in details)
    rollout=read(HERE/'master6_pressure.json.gz')['groups']['all'];q=next(r for r in details if r['id'].startswith('408dd454'));other=next(r for r in details if r['id'].startswith('a496c8d5'))
    bullet=f"这批 {len(details)} 场按现金差绝对值选出的最大十步中，{strong} 场的普通存入差与分差变动同向、绝对量至少占一半。对安生N21，Master6 在 t104 已带货 {q['earlier_laden'][0][1]:,.2f}，t114 增至 {q['root_laden'][0]:,.2f}，随后十步交回 {q['worker_cash'][0]:,.2f}，对手零存入；其树息却只得 {q['tree_cash'][0]:.2f}，对手为 {q['tree_cash'][1]:.2f}。对同一批的 stephen-K4TacSearch-v4，最大十步开始前十步的存货还只是 {other['earlier_laden'][0][1]:,.2f} 对 {other['earlier_laden'][0][2]:,.2f}。这说明仅以当时总货领先为触发，会错过部分随后形成的回款窗口。该窗口经赛后挑选，此比例不能外推到所有决策时刻。"
    census=f"累计全样本 Master6 出场 {rollout['games']} 次，对 {rollout['opponents']} 名对手，{rollout['outcome']['win']} 胜、{rollout['outcome']['draw']} 平、{rollout['outcome']['loss']} 负。这批新增对手均输给 Master6，旧负局依然在累计分母内。[累计逐局现金、人数与持续领先](master6_pressure.json.gz)"
    body=(HERE/'README.md').read_text().replace('## 回溯与账本\n','## 回溯与账本\n\n'+bullet+'\n\n'+census+'\n');(HERE/'README.md').write_text(body)
    md=['# 原版复算与正式考试地图隔离','',f'独立扫描改动成绩表，扣除 1,386 个已归档 ID 和场次，与全仓重扫一致，均识别出 16 张新图。双方物理席、官方终分与归档重放一一对应。官方 Board 与 FastCarbon 分别核过 {tr:,} 次联合推进。', '',f'十批各自互不重复的 ID、环境 UUID、规范回放 SHA、种子和真实开局地形均为 1,402 个。独立考试同时排除线上种子及官方初盘 225 格真实碳图指纹。本批仅用相同种子在本地生成而恰好与原图相符的有 {audit["seed_alone_recreates_map"]}/16 张。保留真实地图能使重放脱离本机种子生成路径。', '',f'新归档 {ar["bytes"]:,} 字节，含 {ar["members"]} 张原始压缩回放，SHA-256 `{ar["sha256"]}`。各回放另有单独的原字节 SHA-256。[逐局引擎报告](validation_games.jsonl.gz) · [种子和地图指纹](validation_maps.jsonl.gz) · [合计核验](validation.json) · [正式隔离入口](../../map_identity.py)',''];(HERE/'validation.md').write_text('\n'.join(md))
if __name__=='__main__':main()
