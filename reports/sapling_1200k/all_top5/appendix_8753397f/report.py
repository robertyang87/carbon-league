"""Join new independently reconstructed games to the preceding physical and same-opponent ledgers."""
import gzip
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
TOP=HERE.parent
sys.path.insert(0,str(TOP))
import fetch
from continue_appendix import previous_manifests
from defense_summary import ranking_lookup


def load(path):
    data=path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix=='.gz' else data)


def exposure(m, prior):
    all_games=[];files=[];faults=[];new=[]
    for p in [*prior,HERE/'manifest.json']:
        for f in (p.parent/'validation_maps.jsonl.gz',p.parent/'unscored/validation_maps.jsonl.gz'):
            if not f.exists():continue
            raw=gzip.decompress(f.read_bytes());rows=[json.loads(z) for z in raw.splitlines()]
            all_games.extend(rows)
            files.append([str(f.relative_to(TOP.parent)),len(rows),fetch.sha(raw),
                          fetch.sha('\n'.join(sorted(r['id'] for r in rows)).encode())])
            if f.parent.name=='unscored':faults.extend(r['id'] for r in rows)
            elif p.parent==HERE:new=rows
    size=len(all_games)
    for field in ('id','seed','initial_terrain_u8_sha256'):
        if size!=len({g[field] for g in all_games}):raise ValueError('Duplicate archived '+field)
    cf=TOP.parent/'master6_actor_dev/cargo_tree_action_learning_dev/native_actor_corpus/league_policy/native_return_intake/live_curriculum/clearance.json'
    raw=fetch.git('show',m['revision']+':'+str(cf.relative_to(fetch.ROOT)))
    hold=json.loads(raw)
    boards=dict(hold['by_seed'])
    for group in hold['explicitly_protected_maps'].values():boards.update(group['by_seed'])
    seeds={r['seed'] for r in new}; terrains={r['initial_terrain_u8_sha256'] for r in new}
    lo,hi=hold['formal_seed_half_open']
    if (seeds & set(map(int,boards)) or terrains & set(boards.values()) or any(lo<=s<hi for s in seeds)):
        raise ValueError('Overlapping registered development/training or formal seeds')
    changes=hold['verified_score_changes']
    official={r['game']:r for r in changes['verified_replays']}
    for r in new:
        if r['initial_terrain_u8_sha256']!=official[r['id']]['opening_225_u8_sha256'] or r['seed']!=official[r['id']]['seed']:
            raise ValueError('Physical opening disagrees with independent queue clearance')
    if size!=m['accumulated_games']+len(faults) or len(faults)!=1 or len(new)!=6:raise ValueError('Coverage drift')
    fetch.dump(HERE/'exposure.json',dict(revision=m['revision'],complete_two_scores=m['accumulated_games'],
        official_fault_prefix=len(faults),distinct_ids_seeds_and_initial_terrains=size,distinct_archived_ids_including_fault=size,
        distinct_archived_seeds_including_fault=size,distinct_archived_terrains_including_fault=size,
        fault_ids=sorted(faults),source_ledger_sha256=files,all_source_sha256=fetch.sha(json.dumps(files,separators=(',',':')).encode()),
        new_ids=[r['id'] for r in new],new_observed_openings=new,independent_clearance_source=str(cf.relative_to(fetch.ROOT)),
        independent_clearance_sha256=fetch.sha(raw),checked_opened_seeds_and_fingerprints=boards,
        formal_seed_half_open=[lo,hi],protected_seed_intersection=[],protected_physical_intersection=[],
        earlier_score_delta=dict(start=changes['start'],end=changes['end'],distinct_newly_seen_ids=len(changes['newly_seen_score_ids']),
                                 distinct_ids_with_new_references=changes['new_reference_ids'],new_references=changes['new_references'])))
    return size,len(boards)


def build():
    m=load(HERE/'manifest.json');delta=load(HERE/'score_delta.json');val=load(HERE/'validation.json')
    ctx=load(HERE/'context_receipt.json');h=load(HERE/'future_economy_check.json')
    prior=previous_manifests(TOP/'appendix_87688aac')
    past=[g for p in prior for g in load(p)['games']]
    rows={r['id']:r for r in load(HERE/'cases.json.gz')['rows']}
    top={p['id'] for p in m['top5']};ranks,date=ranking_lookup()
    results=[];comps=[];lines=[]
    n=len(m['games']); transitions=sum(g['transitions'] for g in m['games'])
    if not (transitions==val['original_board_transitions']==val['fastsim_transitions']==ctx['joint_transitions'] and set(rows)=={g['id'] for g in m['games']}):raise ValueError('State count mismatch')
    for g in m['games']:
        seat=next(s for s,id_ in enumerate(g['seat_strategy_ids']) if id_ in top);opp=1-seat
        gid=g['id'];scores=g['score_by_seat'];margin=round(scores[seat]-scores[opp],2);names=g['seat_names']
        ec=rows[gid]['economy'];swing=ec['swing'];flow=ec['whole']['seats']
        history={d:next(x for x in swing['positions'] if x['t']==max(0,swing['a']-d)) for d in (3,5,10)}
        results.append(dict(id=gid,seed=g['seed'],top_seat=seat,top_name=names[seat],opponent=names[opp],score_by_seat=scores,
                          swing=swing,lookback=history,all_turn_flows_by_physical_seat=flow,terminal_cash_and_assets=ec['terminal_assets']))
        comp=[dict(game=q['id'],seed=q['seed'],physical_seat_scores=q['score_by_seat'],
                   top_margin=round(q['score_by_seat'][seat]-q['score_by_seat'][opp],2)) for q in past if q['seat_names']==names]
        comps.append(dict(current=gid,opponent=names[opp],opponent_tier=ranks(g['seat_strategy_ids'][opp],names[opp]),frozen_rank_date=date,
            top_name=names[seat],top_seat=seat,current_margin=margin,same_two_seats_and_registrants=comp))
        monetary=lambda key:' / '.join(f'{flow[s][key]:,.2f}' for s in (seat,opp))
        swing_delta=(1-2*seat)*swing['margin0_delta']
        lines.append(f"| [{names[seat]}（席 {seat}）]({gid}.md) | {names[opp]} | {scores[seat]:,.2f} / {scores[opp]:,.2f} | {monetary('bank')} | {monetary('tree')} | {swing['a']}–{swing['b']}，{swing_delta:+,.2f} |")
    fetch.dump(HERE/'turning_points.json',results);fetch.dump(HERE/'same_opponent.json',comps)
    size,registered=exposure(m,prior)
    tables=['# 前五对外六场逐局补编','',
       f"冻结 `{delta['previous_score_revision'][:8]}..{m['revision'][:8]}`。新增引用涉及 35 个赛果 ID、43 条索引引用；其中六个属于前五，全部为对外赛。累计 **1,644 场有双方官方终分**，其中互打 121 场，对外 1,523 场；另有 1 场历史执行故障单列。逐局清单按全仓 ID 去重。[索引逐项对读](score_delta.json) · [六场归档](manifest.json)",'',
       '| 前五选手与物理席 | 对手 | 官方终分（前五 / 对手） | 整局交仓（前五 / 对手） | 整局树收入（前五 / 对手） | 最大十拍现金摆动 |',
       '|---|---|---:|---:|---:|---|',*lines,'',
       'Master6 在同一物理席打了两种结局。对 TidalSearch 胜 5,956.86，双方从敌工抢到的货都是 0，Master6 交仓 5,963.11，树收入 6,518.07。对 it920-evo31 负 211.71，Master6 从敌工抢得 204.59，实际兑现到己方的抢货 139.26；it920-evo31 的普通交仓多 243.32。该场第 77–87 拍双方交仓分别为 201.27 和 761.74，树收入分别为 98.19 和 266.94。两场的抢货、交仓和树收入差别明显。对局轨迹本身无法辨识 Master6 的闭源网络或搜索结构。[胜局](3f03828a6fbb0bba1de8f139a92dc8f8-1.md) · [败局](494137282f2dda7069f925855ac71b11-1.md)','',
       '六场都保留了关键现金波动前 3、5、10 步的双方动作、基地与工人身份、位置和持货。树权得失、对敌基地的意外入账、末段种树与剩余资产分别列在逐局页。每名选手另按同一对手、相同物理席对照此前赛果。[回溯快照](turning_points.json) · [逐步动作](turn_actions.json.gz) · [逐步现金](turn_cash.json.gz) · [同对手同席](same_opponent.json)','',
       f"双引擎各复演 {transitions:,} 个联合回合，两个物理席共 {2*transitions:,} 个行动前状态。现场只可见当席公开状态和已发生的历史。10／30 步之后的收入属于事后标签，独立核了 {h['audited_seat_windows']} 个席位窗口，截尾单列。六张初始地形与前批游戏、历史故障和 {registered} 张已登记开发／训练地图均无种子或地形交集；终考保留种子段仅核种子范围，禁止将本次看到的初盘用于独立终考。含旧故障的全量清单有 {size:,} 个实测初盘。[双引擎复验](validation.json) · [两席可见输入](context_receipt.json) · [未来标签](future_economy_check.json) · [初盘和保留图复核](exposure.json)"]
    (HERE/'README.md').write_text('\n'.join(tables)+'\n')


if __name__=='__main__':build()
