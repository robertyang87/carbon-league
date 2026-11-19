"""Join turn-level cash with registered competitors and archived outcomes on the same physical seat."""
import gzip
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
TOP=HERE.parent
sys.path.insert(0,str(TOP))
import fetch
from continue_appendix import previous_manifests
from defense_summary import ranking_lookup

def read(p):
    data=p.read_bytes()
    return json.loads(gzip.decompress(data) if p.suffix=='.gz' else data)
def build():
    m=read(HERE/'manifest.json');prev=previous_manifests(TOP/'appendix_628e28d0')
    past=[g for p in prev for g in read(p)['games']]
    top={g['id'] for g in m['top5']}
    rows={r['id']:r for r in read(HERE/'cases.json.gz')['rows']}
    games=m['games']
    val=read(HERE/'validation.json'); ctx=read(HERE/'context_receipt.json')
    assert val['original_board_transitions']==val['fastsim_transitions']==ctx['joint_transitions']==sum(g['transitions'] for g in games)
    assert set(rows)=={g['id'] for g in games}
    rank,date=ranking_lookup()
    lines=['# 前五追加两场逐局复核','',f"截至 `{m['revision'][:8]}`，两场均为前五对外赛。累计有 1,638 场获双方官方终分，其中互打 121 场、对外 1,517 场。另有 1 场历史执行故障，共 1,639 个实测初盘。[逐索引差分](score_delta.json) · [地图指纹](exposure.json)",'',
           '| 前五选手 / 席位 | 对手 | 官方分（前五 / 对手） | 最大十步现金摆动 | 十步分差变化 | 正常交仓（前五 / 对手） | 树收入（前五 / 对手） |',
           '|---|---|---:|---:|---:|---:|---:|']
    results=[]; comparisons=[]
    for g in games:
        seat=next(i for i,pid in enumerate(g['seat_strategy_ids']) if pid in top); other=1-seat
        gid=g['id']; seatorder=(seat,other); names=[g['seat_names'][i] for i in seatorder]
        scores=[g['score_by_seat'][i] for i in seatorder]; stats=rows[gid]['economy']; swing=stats['swing']
        bank=[swing['seats'][i]['bank'] for i in seatorder];tree=[swing['seats'][i]['tree'] for i in seatorder]
        change=(1-2*seat)*swing['margin0_delta']
        lines.append(f"| [{names[0]} / {seat}]({gid}.md) | {names[1]} | {scores[0]:,.2f} / {scores[1]:,.2f} | {swing['a']}–{swing['b']} | {change:+,.2f} | {bank[0]:,.2f} / {bank[1]:,.2f} | {tree[0]:,.2f} / {tree[1]:,.2f} |")
        horizon={o:next(x for x in swing['positions'] if x['t']==max(0,swing['a']-o)) for o in (3,5,10)}
        results.append(dict(id=gid,seed=g['seed'],top_seat=seat,top_name=names[0],opponent=names[1],score_by_seat=g['score_by_seat'],
                            swing=swing,lookback=horizon,terminal_cash_and_assets=stats['terminal_assets']))
        opp_rank=rank(g['seat_strategy_ids'][other],names[1])
        comparable=[dict(game=h['id'],seed=h['seed'],physical_seat_scores=h['score_by_seat'],
                         top_margin=round(h['score_by_seat'][seat]-h['score_by_seat'][other],2)) for h in past if h['seat_names']==g['seat_names']]
        comparisons.append(dict(current=gid,opponent=names[1],opponent_tier=opp_rank,frozen_rank_date=date,top_name=names[0],top_seat=seat,
                                current_margin=round(scores[0]-scores[1],2),same_two_seats_and_registrants=comparable))
    fetch.dump(HERE/'turning_points.json',results);fetch.dump(HERE/'same_opponent.json',comparisons)
    lines+=['','两局均保留现金转折前 3、5、10 步双方具体工人、基地、树、货量及实际命令。逐拍现金包含正常交仓、树吸收、碰撞入账、双方支出。双方同时选择动作，后续结果仅作标签。 [回溯快照](turning_points.json) · [逐步动作](turn_actions.json.gz) · [收入账](turn_cash.json.gz) · [攻防与树权](cases.json.gz)','']
    for item in comparisons:
        comp=item['same_two_seats_and_registrants']; counts=[sum(r['top_margin']>0 for r in comp),sum(r['top_margin']==0 for r in comp),sum(r['top_margin']<0 for r in comp)]
        tier=item['opponent_tier']
        classification='冻结榜未查到对手档位' if tier=='unknown' else '冻结榜档位为 '+tier
        lines.append(f"{item['top_name']} 在物理席 {item['top_seat']} 对 {item['opponent']}，本场终局分差 {item['current_margin']:+,.2f}。此前同一对阵、双边物理席位也相同的比赛共有 {len(comp)} 场，胜／平／负为 {counts[0]}／{counts[1]}／{counts[2]}。{classification}。 [同对手逐场证据](same_opponent.json)")
        lines.append('')
    lines.append('双引擎分别复算 598 个联合回合，两席共 1,196 个行动前状态可从公开历史重建。另独立抽核 64 个席位前瞻窗口，区分完整 10／30 步和被终局截断的未来收入。 [规则复演](validation.json) · [无泄漏索引](context_receipt.json) · [前瞻收入](future_economy_check.json) · [原始回放](replays.tar)')
    (HERE/'README.md').write_text('\n'.join(lines)+'\n')
    games_all=[];sources=[];faults=[]
    for p in [*prev,HERE/'manifest.json']:
        for f in (p.parent/'validation_maps.jsonl.gz',p.parent/'unscored/validation_maps.jsonl.gz'):
            if not f.exists():continue
            data=gzip.decompress(f.read_bytes());entries=[json.loads(z) for z in data.splitlines()];games_all+=entries
            sources.append([f.relative_to(TOP.parent).as_posix(),len(entries),fetch.sha(data),fetch.sha('\n'.join(sorted(r['id'] for r in entries)).encode())])
            if f.parent.name=='unscored':faults.extend(r['id'] for r in entries)
    assert len(games_all)==1639==len({g['id'] for g in games_all})==len({g['seed'] for g in games_all})==len({g['initial_terrain_u8_sha256'] for g in games_all}) and len(faults)==1
    fetch.dump(HERE/'exposure.json',dict(revision=m['revision'],complete_two_scores=1638,official_fault_prefix=1,distinct_ids_seeds_and_initial_terrains=1639,
               distinct_archived_ids_including_fault=1639,distinct_archived_seeds_including_fault=1639,distinct_archived_terrains_including_fault=1639,
               fault_ids=sorted(faults),source_ledger_sha256=sources,all_source_sha256=fetch.sha(json.dumps(sources,separators=(',',':')).encode()),
               new_ids=[g['id'] for g in games],new_observed_openings=games_all[-2:]))
if __name__=='__main__':build()
