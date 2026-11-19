"""Run the locked seven-game freight diagnostic on the complete, separately frozen 26-game inventory."""
import numpy as np
from pathlib import Path
import argparse
import hashlib
import json
from latest5_freight import main as run_freight, money
from latest5_freight_mutual import KEY


def main(inventory,cache,mutual,out,md):
    frozen=json.loads(inventory.read_bytes())
    regs={g["id"]:g for g in frozen["games"]}
    run_freight(inventory,cache,out,md)
    result=json.loads(out.read_bytes())
    result["inventory_sha256"]=hashlib.sha256(inventory.read_bytes()).hexdigest()
    result["source_cohorts"]=frozen["counts_by_cohort"]
    for game in result["games"]:
        game["cohort"]=regs[game["id"]]["cohort"]
        game["github_index_paths"]=regs[game["id"]].get("github_index_paths",[])
    # Carry the first seven's cutoffs unchanged into the next 19 games. A
    # bootstrap draw keeps every worker and both seats of each selected map.
    strata={}
    initial=json.loads((Path(__file__).parent/"latest5_inventory_github.json").read_bytes())
    initial_ids={g["id"] for g in initial["games"] if g["eligible"]}
    if len(initial_ids)!=7 or len(regs)!=26 or not initial_ids<=regs.keys():
        raise ValueError("different replay cohort from the frozen 7+19 comparison")
    for tag in ("github_7","subsequent_19"):
        sample=[]
        for g in result["games"]:
            if (g["id"] in initial_ids) != (tag=="github_7"):continue
            row=[]
            for k in ("approaching","other"):
                vals=[v["distant_load_followup"].get(k,{}) for v in g["game"]["seats"]]
                row.extend([sum(v.get("first_own_deposit_within_10",0) for v in vals),sum(v.get("seen",0) for v in vals)])
            sample.append(row)
        sample=np.array(sample)
        total=sample.sum(axis=0)
        boot=np.random.default_rng(20261119).integers(len(sample),size=(10000,len(sample)))
        b=sample[boot].sum(axis=1); b=b[(b[:,1]>0)&(b[:,3]>0)]
        ci=np.percentile(100*(b[:,0]/b[:,1]-b[:,2]/b[:,3]),[2.5,97.5])
        strata[tag]=dict(maps=len(sample),approach_hits=int(total[0]),approach_n=int(total[1]),
                       other_hits=int(total[2]),other_n=int(total[3]),
                       delta_pp=float(100*(total[0]/total[1]-total[2]/total[3])),
                       map_cluster_bootstrap_95ci_pp=ci.tolist())
    result["fixed_threshold_strata"]=strata
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    report=md.read_text()
    cc=frozen["counts_by_cohort"]
    indexed=cc.get("github_top5_external",0)+cc.get("github_top5_mutual",0)
    mutual_count=cc.get("live_top5_mutual",0)+cc.get("github_top5_mutual",0)
    report=report.replace("26 份 Top5 对榜外队伍的新回放，零场 Top5 互打。",
           f"26 份新回放。按 GitHub 的五个 Top5 家族索引（{frozen['github_after'][:8]}），其中 {indexed} 场在库，另 {26-indexed} 场由天梯补齐；查看全部 {frozen.get('all_github_families_index_count',0)} 个家族索引（{frozen.get('all_github_families_as_of','')[:8]}），有 {frozen.get('all_families_present_in_github',indexed)} 场可在库中找到。26 场包含 {mutual_count} 场 Top5 互打和 {26-mutual_count} 场对榜外队伍。首次分析的 7 场与后取的 19 场按对局 ID 锁定。")
    names={"github_top5_external":"Top5 索引，对榜外队伍","live_top5_external":"天梯补齐，对榜外队伍","live_top5_mutual":"天梯补齐，Top5 互打"}
    lines=report.splitlines()
    for i,line in enumerate(lines):
        for g in result["games"]:
            if line.startswith(f"| `{g['id'][:8]}`"):
                lines[i]=line.replace("新图，诊断",names[g["cohort"]]+"，新图诊断")
    report="\n".join(lines)+"\n"
    m=json.loads(mutual.read_bytes())["ledger"]
    focus=next(g["game"] for g in result["games"] if g["id"]==KEY)
    if m["verified_transitions"]!=focus["transitions"] or m["official_rewards"]!=focus["reported_rewards"]:
        raise ValueError("mutual ledger mismatch")
    detailed=["","## 新出现的 Top5 互打 · Master6 对 it720-preview","",
        "回放 `5bd3791f228a5dda51411c62a10b6bdf-1` 的席 0 是 it720-preview，席 1 是 Master6。按归档联合动作从开局重放 299 步，逐步核对两边现金、工人身份、位置与载货、树的所有权和年龄，再将引擎的每笔现金变化入账。下文差额均以 Master6 减 it720 计算。","",
        "**开局到中盘。** 第一步双方都在两个基地各招一名采集工和一名种树工，第 8 步都首次种树。it720 第 7 步先回款 10.48，Master6 第 13 步才回款 18.37。第 30 步 Master6 现金 1.94、比对手少 8.72；第 60 步虽只领先 15.61，已有 7 棵树、9 个工人，对手只有 2 棵树、6 个工人。第 30–60 步间 Master6 的正常回款反而少 313.03，树吸收已多 548.35。这个阶段树的产出开始支撑扩张。","",
        "**比分何时脱离。** 第 94 步 Master6 开始持续领先，现金只有 94.96 对 75.37，但工人已是 10 对 4、树是 6 对 2。第 108 步树 10 对 0、采集工 8 对 2，领先 572.43；第 118 步树 13 对 0、现金 1,031.57 对 16.36，领先首次持续越过 1,000。","",
        "**钱从哪里来。** 整局正常回款 7,110.62 对 3,421.44，净多 3,689.18；树吸收 7,491.24 对 1,593.93，净多 5,897.31。为此 Master6 在种树与夺树上净多花 1,676.26，招工多花 90.00。树吸收差额中只有 169.23 来自敌方载货，其余约 5,728.08 来自地面碳。第 113 步，Master6 的空载工 player-1-worker-52 向左进 it720 的基地 160，当格截下 player-0-worker-43 的 116.70。游戏当步先判碰撞、再向基地所有者入库，所以这 116.70 最终进了 it720 的账；碰撞移交给工人的货不能直接当成现金收益。全局现金差为 7,703.62。","",
        "**后半局最大的十步。** 第 198 至 208 步 Master6 一口气拉开 1,413.83，其中正常回库多 1,224.83、树吸收多 169.01、树投资少 20.00。这个瞬间表现为交付，能长期维持的中盘优势则在百步之前已经开始形成。","",
        "| 区间 | Master6 现金差变化 | 正常回款差 | 树吸收差 | 种树与夺树支出差（带符号） | 招工支出差（带符号） |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for p in m["strategic_phases"]:
        d=p["diff"]
        detailed.append(f'| {p["start"]}–{p["end"]} | {p["margin_change"]:+.2f} | {d["own_delivery"]:+.2f} | {d["absorption"]:+.2f} | {d["tree_spend"]:+.2f} | {d["recruit"]:+.2f} |')
    detailed.extend(["","逐阶段完整账本另列对手基地误送及舍入。单场比较仍受双方布阵和地图影响。"])
    subgroup=["","## 在另外 19 场检验预警信号","",
              "用先前 7 场确定的载货、距离、前三步朝向和 10 步回款定义，保持阈值重算后取的 19 场。不确定度按地图重抽 10,000 次，每次保留该图两席和全部工人；差值单位为百分点。","",
              "| 样本 | 距库缩短后回款 | 持平或变远后回款 | 差值（95% 地图聚类区间） |","| --- | ---: | ---: | ---: |"]
    for label,tag in (("先前 GitHub 七局，探索","github_7"),("后取十九局，固定口径","subsequent_19")):
        q=strata[tag];lo,hi=q["map_cluster_bootstrap_95ci_pp"]
        subgroup.append(f"| {label} | {q['approach_hits']}/{q['approach_n']} | {q['other_hits']}/{q['other_n']} | {q['delta_pp']:+.1f} [{lo:+.1f}, {hi:+.1f}] |")
    subgroup.extend(["","这个信号可作为对手返程概率的一个输入。它没有定位敌工具体下一格，也未验证埋伏后我方能留住并兑付货物。"])
    report+="\n"+"\n".join(subgroup)+"\n"
    insert_at=report.index("## 拉开差距前的布局")
    report=report[:insert_at]+"\n".join(detailed)+"\n\n"+report[insert_at:]
    md.write_text(report)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("inventory","cache","mutual","out","md"):p.add_argument(n,type=Path)
    a=p.parse_args();main(a.inventory,a.cache,a.mutual,a.out,a.md)
