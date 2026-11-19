"""Write the external game's observational account and check all previously reserved opening identities."""
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

def main():
    m=fetch.load(HERE/'manifest.json');g=m['games'][0];gid=g['id'];scores=g['score_by_seat'];names=g['seat_names']
    cohort={q['id'] for q in m['top5']};s=next(i for i,pid in enumerate(g['seat_strategy_ids']) if pid in cohort);o=1-s
    prior=previous_manifests(TOP/'appendix_c8236c8c'); old=fetch.load(prior[-1].parent/'exposure.json')
    v=fetch.load(HERE/'validation.json');h=fetch.load(HERE/'future_economy_check.json');audit=fetch.load(HERE/'reference_audit.json')
    assert v['fastsim_transitions']==v['original_board_transitions']==g['transitions']==299 and h['decision_seats']==598
    exposure=fetch.load(HERE/'exposure.json')
    gm=json.loads(gzip.decompress((HERE/'validation_maps.jsonl.gz').read_bytes()))
    boards=old['checked_opened_seeds_and_fingerprints'];lo,hi=old['formal_seed_half_open']
    assert str(g['seed']) not in boards and gm['initial_terrain_u8_sha256'] not in boards.values() and not lo<=g['seed']<hi
    assert exposure['distinct_ids_seeds_and_initial_terrains']==1647 and exposure['official_fault_prefix']==1
    exposure.update(checked_opened_seeds_and_fingerprints=boards,protected_seed_intersection=[],protected_physical_intersection=[],
        formal_seed_half_open=[lo,hi],new_observed_openings=[gm],distinct_archived_ids_including_fault=1647,
        distinct_archived_seeds_including_fault=1647,distinct_archived_terrains_including_fault=1647)
    fetch.dump(HERE/'exposure.json',exposure)
    ec=json.loads(gzip.decompress((HERE/'cases.json.gz').read_bytes()))['rows'][0]['economy'];w=ec['whole']['seats'];sw=ec['swing']
    comp=[]
    for p in prior:
        for q in fetch.load(p)['games']:
            if q['seat_names']==names:comp.append(dict(game=q['id'],seed=q['seed'],physical_seat_scores=q['score_by_seat'],top_margin=round(q['score_by_seat'][s]-q['score_by_seat'][o],2)))
    rank,date=ranking_lookup()
    fetch.dump(HERE/'same_opponent.json',[dict(current=gid,opponent=names[o],opponent_tier=rank(g['seat_strategy_ids'][o],names[o]),frozen_rank_date=date,
        top_name=names[s],top_seat=s,current_margin=round(scores[s]-scores[o],2),same_two_seats_and_registrants=comp)])
    history={n:next(x for x in sw['positions'] if x['t']==max(0,sw['a']-n)) for n in (3,5,10)}
    fetch.dump(HERE/'turning_points.json',[dict(id=gid,seed=g['seed'],top_seat=s,top_name=names[s],opponent=names[o],score_by_seat=scores,swing=sw,
              lookback=history,all_turn_flows_by_physical_seat=w,terminal_cash_and_assets=ec['terminal_assets'])])
    counts=[sum(r['top_margin']>0 for r in comp),sum(r['top_margin']==0 for r in comp),sum(r['top_margin']<0 for r in comp)]
    text=f"""# Fable5.2-v7 对 it920-evo31，一场新增对外赛

冻结 `c8236c8c..ea224932`。成绩索引新增引用涉及 47 个不同赛果 ID。与旧版全仓索引核对，30 个是已有赛果新增镜像入口（其中 5 个为此前已归档的前五比赛），17 个此前没有任何索引引用。首次出现的赛果中，16 个属于其他队伍，1 个涉及冻结前五。另有 4 个新增回放文件：Echo9 的两份对应旧赛果，Echo10 的两份对应新的非前五赛果。累计 **1,646 场有双方官方终分**（互打 121、对外 1,525），历史执行故障 1 场另列。[逐索引差分](score_delta.json) · [逐 ID 分类与原始回放文件身份](reference_audit.json)

ClaudeFable5.2-v7 在物理席 1 以 **6,090.28 比 8,069.37** 负于 it920-evo31。双方全局普通交仓分别为 7,702.31、8,824.55，树收入为 2,770.77、3,855.27。Fable 较少的种植支出收回部分差距（1,257.03 对 1,680.50）。前五选手取得对手所携货物 4.09，对手取得 146.53；抢得的货可能在后续转移或损失，只在到账时计入现金。[全场真实收支、攻防与终局树资产]({gid}.md)

第 84–94 步 Fable 的现金分差减少 1,009.36。Fable 在窗口内交仓 287.98，对手交仓 1,305.82；同期 Fable 的树收入反而多 26.49。窗口前 3、5、10 步双方的工人位置、载货、树状态和全部基地及工人实际命令均单独归档。这一窗口描写已发生的收入转折，单次对局不足以归因于某一工人的决定。此前同一对手、物理席也相同的对局有 {len(comp)} 场，Fable 胜／平／负 {counts[0]}／{counts[1]}／{counts[2]}。[窗口与回溯](turning_points.json) · [整局公开动作](turn_actions.json.gz) · [逐拍现金](turn_cash.json.gz) · [同席对照](same_opponent.json)

本场双引擎各复演 {v['fastsim_transitions']} 个联合回合，重建 {h['decision_seats']} 个行动前单席状态。事后收益与当时可观察状态分别保存，另核 {h['audited_seat_windows']} 个 10／30 步前瞻样本窗口。种子 {g['seed']} 和 225 格实际开局地形与此前 1,646 个实测初盘、{len(boards)} 张已登记开发／训练图均无重合；正式考卷保留种子段另行核对，本场排除在独立终考之外。[官方与第二引擎核验](validation.json) · [无泄漏观测](context_receipt.json) · [前瞻标签](future_economy_check.json) · [开图排除凭据](exposure.json)
"""
    (HERE/'README.md').write_text(text)
if __name__=='__main__':main()
