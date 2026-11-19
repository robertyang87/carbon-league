"""Recheck the portable survival veto on held-out-opponent source tapes, without starting a match."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
DEV=HERE.parent
TRIAL=DEV.parent.parent
for path in (TRIAL,DEV):
    sys.path.insert(0,str(path))
import parity
import opportunities
from extract import load, write, sha, ROOT
from model import MOVES

SIGNATURE = "8931f82a283890b7fb157237dc9be2e14eb5236f523160bbe7ffefd308da4d2c"


def enroll(games):
    """Only data already available to the player, and our submitted orders, determine enrollment."""
    dest=HERE/"opportunities.jsonl.gz"
    temp=dest.with_suffix(".tmp")
    with temp.open("wb") as sink, gzip.GzipFile(filename="",fileobj=sink,mode="wb",mtime=0) as handle:
        for g in games:
            for row in opportunities.iter_game(ROOT/g["raw_path"]):
                args={k:row[k] for k in ("seed","seat","t","uid","parent_command")}
                args["actions"]={}
                for name in MOVES:
                    fields=("entry_legal","friendly_conflict","price","can_pay_now") if name=="NONE" else ("entry_legal","friendly_conflict")
                    args["actions"][name]={k:row["actions"][name][k] for k in fields}
                handle.write((json.dumps(args,separators=(",",":"))+"\n").encode())
    temp.replace(dest)


def main():
    p=parity.document(HERE/"protocol.json",SIGNATURE)
    model=parity.document(DEV.parent/"weights.json",p["weights"])
    parity.check(sha(DEV/"candidate.py")==p["candidate"] and sha(DEV/"parity.py")==p["portable_parity_script"] and
                 sha(TRIAL/"opportunities.py")==p["opportunities_code"], "frozen checker or actor changed")
    parity.check(sha(ROOT/p["opponent"])==p["opponent_sha256"],"opponent source mismatch")
    parity.check(sha(DEV/"protocol.json")==p["native_dev_protocol"],"portable selection changed")
    benchmark=TRIAL.parent/"terminal_return_fit/opponent_transfer"
    parity.check(sha(benchmark/"protocol.json")==p["source_protocol"] and sha(benchmark/"results.json")==p["source_scores"],"source matches changed")
    parity.check(set(p["map_ids"]).isdisjoint(model["fit_seeds"]+p["risk_validation_maps"]),"risk training/validation overlap")
    games=p["games"]
    for g in games:
        r=parity.document(ROOT/g["receipt_path"],g["receipt_sha256"])
        z=parity.document(ROOT/g["raw_path"],g["raw_sha256"])
        o=r["official_replay"]
        parity.check(r["opponent_sha256"]==z["opponent_sha256"]==p["opponent_sha256"] and
                     r["pack_sha256"]==z["pack_sha256"]==p["parent"] and
                     o["statuses"]==["DONE","DONE"] and o["transcript_sha256"]==g["raw_sha256"] and
                     o["official_reward_by_seat"]==g["scores"] and o["map_sha256"]==g["map_sha256"],"signature or reward mismatch")
    enroll(games)
    index,source=parity.read_index(HERE,p["map_ids"])
    identity=dict(benchmark_sha256=SIGNATURE, pack_sha256=p["candidate"],protocol_sha256=p["native_dev_protocol"],
                  weights_sha256=p["weights"],opponent_sha256=p["opponent_sha256"],reference_chooser_sha256=p["portable_parity_script"],
                  opportunity_sha256=sha(HERE/"opportunities.jsonl.gz"),source_results_sha256=p["source_scores"])
    per_map={s:Counter() for s in p["map_ids"]}
    total=Counter()
    files={}
    per_seat=[]
    for g in games:
        path=HERE/f"{g['seed']}_{g['seat']}.json.gz"
        wanted={**identity,"tape_sha256":g["raw_sha256"],"map_sha256":g["map_sha256"]}
        if not path.exists():
            parity.run_game(DEV/"candidate.py",g,index,model,path,identity)
        doc=load(path)
        parity.check(doc["signed"]==wanted,"stale shard")
        files[path.name]=sha(path)
        stats=doc["stats"]
        per_map[g["seed"]].update(stats)
        total.update(stats)
        per_seat.append(dict(seed=g["seed"],seat=g["seat"],source_roots=source[g["seed"],g["seat"]],source_scores=g["scores"],stats=stats))
    out=dict(identity=identity,source_planter_roots=sum(source.values()),source_maps=p["map_ids"],per_map=per_map,
             per_map_seat=per_seat,total=total,shards=files,changed_trajectory_games=0,frozen_source_opponent_wdl=[0,0,4])
    write(HERE/"result.json",out)
    lines=["# 种植工风险头能否应对另一套对手策略","",
        f"从两张图交换席位的四条历史 v1 对局，逐步重建 {total['action_turns']:,} 次出手。四局中，手写对手 policy_v0_8_replace 全负于原 v1。风险头的两张训练图、先前两张验证图和 03／04 地图均与本次不同。候选包、移动抽样、门槛和每步最多干预一人均沿用原冻结版本。[资格与来源](protocol.json) · [原对局与官方复验](../../../../terminal_return_fit/opponent_transfer/README.md)","",
        "|地图、v1 物理席|原终分 v1 / 对方|活种植工根|可付停令|真正改令|救回 / 反伤|可救未救|放弃原实付笔数 / 金额|",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in per_seat:
        s=r["seat"]; z=r["stats"];a,b=r["source_scores"][s],r["source_scores"][1-s]
        lines.append(f"|{r['seed']} / {s}|{a:,.2f} / {b:,.2f}|{r['source_roots']}|{z['opportunities']}|{z['portable_selected']}|{z['saved']} / {z['harmed']}|{z['missed_rescues']}|{z['foregone_payment']} / {z['foregone_debit']:,.2f}|")
    lines+=["",f"{sum(source.values()):,} 个活种植工根中有 {total['opportunities']} 个本来会停留、有支付能力且存在合法无友方冲突移动的机会。独立实现与实际包在全部 {total['action_turns']:,} 步都输出同一份命令，并重现原 v1 记录；所有 {total['leaf_agreements']} 个动作概率一致。对冻结采样到的移动，{total['savable']} 个原停留死亡的工人可获救，实救 {total['saved']}、误伤 {total['harmed']}。实际变化后以相同对手该拍命令、相同友军其余命令，用官方 Board 和 FastCarbon 独立推进，逐步核对工人、树、现金与碳图。[逐席文件摘要与哈希](result.json)","",
        "这些是初始 v1 四条轨迹上的即时结果。轨迹间累计救工不能相加成一整局中实际增加的工人数，放弃的付款是成本及潜在树权，不是已经损失的终局利润。新策略对该对手的整局胜率未测，亦未取得 Master3s2 新胜场。"]
    highlights = (f"**在 {total['opportunities']} 个真实可付停令中改令 {total['portable_selected']} 次，真正救回 {total['saved']} 名原本会当拍阵亡的种植工，误伤 {total['harmed']} 次。** "
                  f"另有 {total['foregone_payment']} 次原可安全支付，改为移动放弃合计 {total['foregone_debit']:.2f} 的购买机会。"
                  "此前对完整 Master3s2 的独立两图有 22 次改令、22 次救工，未放弃付款；不同对手下，原本会撞上的频率和风险提示的命中数发生明显变化。")
    lines[2:2] = [highlights,""]
    (HERE/"README.md").write_text("\n".join(lines)+"\n")
    print("COMPLETE",dict(total),flush=True)


if __name__=="__main__":
    main()
