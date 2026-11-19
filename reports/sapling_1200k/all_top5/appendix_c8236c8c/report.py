"""Summarize the additional it720-preview game and verify it against the protected physical layouts."""
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

PRIOR_PUBLICATION='2ed572054748fc443baa6e596c776ce54ffc736d'

def main():
    m=fetch.load(HERE/'manifest.json'); g=m['games'][0]
    gid=g['id']; names=g['seat_names']; scores=g['score_by_seat']
    v=fetch.load(HERE/'validation.json')
    q=json.loads(gzip.decompress((HERE/'cases.json.gz').read_bytes()))['rows'][0]
    e=q['economy'];s=e['swing'];terminal=e['terminal_assets']
    assert v['fastsim_transitions']==v['original_board_transitions']==g['transitions']==299
    prior=previous_manifests(TOP/'appendix_8753397f')
    six=prior[-1].parent/'validation_maps.jsonl.gz'
    assert fetch.git('show',PRIOR_PUBLICATION+':'+str(six.relative_to(fetch.ROOT)))==six.read_bytes()
    comps=[]
    for p in prior:
        for old in fetch.load(p)['games']:
            if old['seat_names']==names:
                comps.append(dict(game=old['id'],seed=old['seed'],physical_seat_scores=old['score_by_seat'],
                                  top_margin=round(old['score_by_seat'][0]-old['score_by_seat'][1],2)))
    rank,date=ranking_lookup()
    fetch.dump(HERE/'same_opponent.json',[dict(current=gid,opponent=names[1],opponent_tier=rank(g['seat_strategy_ids'][1],names[1]),
             frozen_rank_date=date,top_name=names[0],top_seat=0,current_margin=round(scores[0]-scores[1],2),same_two_seats_and_registrants=comps)])
    history={n:next(x for x in s['positions'] if x['t']==max(0,s['a']-n)) for n in (3,5,10)}
    fetch.dump(HERE/'turning_points.json',[dict(id=gid,seed=g['seed'],top_seat=0,top_name=names[0],opponent=names[1],score_by_seat=scores,
              swing=s,lookback=history,all_turn_flows_by_physical_seat=e['whole']['seats'],terminal_cash_and_assets=terminal)])
    expo=fetch.load(HERE/'exposure.json'); old=fetch.load(prior[-1].parent/'exposure.json')
    boards=old['checked_opened_seeds_and_fingerprints']
    gm=json.loads(gzip.decompress((HERE/'validation_maps.jsonl.gz').read_bytes()).strip())
    lo,hi=old['formal_seed_half_open']
    assert str(g['seed']) not in boards and gm['initial_terrain_u8_sha256'] not in boards.values() and not lo<=g['seed']<hi
    expo.update(previous_ledger_publication=PRIOR_PUBLICATION,checked_opened_seeds_and_fingerprints=boards,
                protected_seed_intersection=[],protected_physical_intersection=[],formal_seed_half_open=[lo,hi],
                new_observed_openings=[gm],distinct_archived_ids_including_fault=1646,
                distinct_archived_seeds_including_fault=1646,distinct_archived_terrains_including_fault=1646)
    assert expo['distinct_ids_seeds_and_initial_terrains']==1646 and expo['complete_two_scores']==1645 and expo['official_fault_prefix']==1
    fetch.dump(HERE/'exposure.json',expo)
    win=sum(r['top_margin']>0 for r in comps);tie=sum(r['top_margin']==0 for r in comps)
    summary=f"""# it720-preview 对 TidalSearch 的新增一局

仅覆盖 `8753397f..c8236c8c` 的六个新增赛果引用，其中一个涉及冻结前五。此前六场补编的赛果截止为 `8753397f`，证据发布提交为 `{PRIOR_PUBLICATION[:8]}`。两段累计 **1,645 场有双方官方终分**，互打 121 场、对外 1,524 场；另有 1 场历史执行故障。[索引差分](score_delta.json) · [本场原始回放](replays.tar)

席 0 的 it720-preview 对席 1 的 TidalSearch 以 **7,174.29 比 6,051.07** 获胜。it720-preview 全局普通交仓 7,213.43，比对手少 192.59；树收入 3,146.79，比对手多 210.80。更大的账面差异是种植费用，it720-preview 支出 885.86，对手支出 2,330.96，相差 1,445.10。这一局中，各项净收支合计给出 +1,123.22 的终局分差。不能据此把少种树作为普适的增收动作。[全局收入、攻防与末段树资产]({gid}.md)

最大十拍波动在第 178–188 步，it720-preview 的现金分差缩小 535.89。两边正常交仓分别为 66.97 和 694.36，树收入分别为 106.25 和 150.71。附录另保留该窗口前 3、5、10 步双方工人载货、位置与实际命令；此前双方在相同物理席还有 {len(comps)} 场，it720-preview 胜／平／负为 {win}／{tie}／{len(comps)-win-tie}。[位置与实令](turning_points.json) · [逐拍回款](turn_cash.json.gz) · [同席对照](same_opponent.json)

官方与第二引擎分别复演 299 个联合回合；两席 598 个行动前状态与后验收入标签分开，另核 32 个席位窗口。开图种子 {g['seed']} 及实际 225 格地形与已有 1,645 个初盘、37 个在用或退休开发／训练图均无交集，另核对正式种子保留范围。本图列入独立终考排除名单。[逐回合核验](validation.json) · [两席公开状态](context_receipt.json) · [前瞻标签](future_economy_check.json) · [开图排除](exposure.json)
"""
    (HERE/'README.md').write_text(summary)
if __name__=='__main__':main()
