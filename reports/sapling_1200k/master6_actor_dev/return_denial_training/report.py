"""Audit the frozen four-map split. List signed wins, failures and a label-blind terminal follow-up sample."""
from collections import Counter
from pathlib import Path
from statistics import mean
import math
import numpy as np
from forecast import HERE, extract, bind, MOVES


def source_summary(games):
    counts=Counter()
    per_method={k:Counter() for k in ("forecast","geometry")}
    per_map={}
    root_sets=[]
    for g in games:
        local={k:Counter() for k in per_method}
        counts.update(g["counts"])
        local_rows=[]
        for frame in g["rows"]:
            pair=frame["arms"]
            for method,arm in pair.items():
                z=local[method];z["opportunities"]+=1
                if not arm["issued"]:
                    z["no_positive_estimate"]+=1
                    continue
                z["issued"]+=1
                y=arm["delta_transfer"]
                z["positive" if y>1e-7 else "negative" if y< -1e-7 else "zero"]+=1
                z["gross_positive"]+=max(0,y);z["gross_negative"]+=max(0,-y);z["total"]+=y
                s=g["seat"]
                for k in ("cross_to_worker","cross_to_tree_cash","cross_to_base_cash","ground_collision_admitted",
                          "ground_collision_overflow","real_new_harvest","gross_cargo_unloaded_by_worker_owner","actual_spend","tree_tax_from_enemy_cargo"):
                    for who,physical in (("us",s),("them",1-s)):
                        z[f"{k}_{who}"]+=arm["alternative"][k][physical]-arm["v1"][k][physical]
                z["own_cash_margin_delta"]+=arm["delta_cash_margin"]
                z["own_deposit_delta"]+=arm["delta_own_deposit"]
                z["fewer_public_shortest_exits"]+=arm["delta_open_shortest"]<0
                z["target_arrived_at_chosen_exit"]+=arm["observed_enemy_cell"]==arm["choice"]["cell"]
                z["selected_worker_new_deaths"]+=arm["alternative"]["own_died"] and not arm["v1"]["own_died"]
            local_rows.append(pair["forecast"])
        for k,z in local.items():
            per_method[k].update(z)
        if g["map"] not in per_map:
            per_map[g["map"]]={"source_turns":0,**{k:Counter() for k in per_method},"forecast_scores":Counter()}
        group=per_map[g["map"]]
        group["source_turns"]+=g["counts"]["source_turns"]
        for k,z in local.items():
            group[k].update(z)
        for f in g["forecasts"]:
            p=np.asarray(f["predicted"])
            y=MOVES.index(f["physical_label"])
            dest=np.asarray(f["shortest"],dtype=float);dest/=dest.sum()
            z=group["forecast_scores"];z["n"]+=1;z["ce"]+=-math.log(max(float(p[y]),1e-12))
            z["correct"]+=int(np.argmax(p))==y
            true=np.eye(5)[y]
            z["brier_fit"]+=float(((p-true)**2).sum())
            z["brier_geometry"]+=float(((dest-true)**2).sum())
            z["observed_shortest"]+=bool(f["shortest"][y])
        root_sets.append((g,local_rows,local))
    for z in per_method.values():
        for k in ("issued","positive","negative","zero","total","gross_positive","gross_negative","no_positive_estimate"):
            z[k]+=0
    map_metrics={m:{"source_turns":z["source_turns"],
                    **{method:dict(z[method]) for method in per_method},
                    "forecast_scores":dict(z["forecast_scores"])}
                 for m,z in per_map.items()}
    avg={k:mean(z[k]["total"]/z["source_turns"] for z in per_map.values()) for k in per_method}
    forecast={key:mean(z["forecast_scores"][key]/z["forecast_scores"]["n"] for z in per_map.values()) for key in
              ("ce","correct","brier_fit","brier_geometry","observed_shortest") if all(z["forecast_scores"]["n"] for z in per_map.values())}
    return {"counts":dict(counts),"methods":{k:dict(v) for k,v in per_method.items()},"per_map":map_metrics,
            "map_equal_gain_per_source_turn":avg,"forecast_map_equal":forecast},root_sets


def main():
    summary={};roster=[];examples=[];shards=[]
    for split in ("train","validation"):
        specs=bind(split)["tapes"];games=[]
        for g in specs:
            p=HERE/f"{split}_{g['seed']}_{g['seat']}.json.gz";d=extract.load(p);games.append(d)
            if d["weights_sha256"]!=extract.sha(HERE/"weights.json"):
                raise ValueError("trained identity changed across splits")
            shards.append({"filename":p.name,"sha256":extract.sha(p),"source":g["raw_path"]})
        summary[split],sets=source_summary(games)
        for g,rows,local in sets:
            if split=="validation":
                for issued in (True,False):
                    same=[r for r in rows if r["issued"]==issued]
                    # No result, opposing simultaneous action, or cash after this action is accessed.
                    if same:
                        r=min(same,key=lambda r:r["signature"])
                        roster.append(dict(map=g["map"],seat=g["seat"],source=g["source_tape"],source_sha256=g["source_sha256"],
                                           kind="issued" if issued else "unissued",t=r["t"],signature=r["signature"],
                                           target=r["target"],choice=r["choice"],prior=r["history"]))
                    else:
                        roster.append(dict(map=g["map"],seat=g["seat"],kind="issued" if issued else "unissued",empty=True))
                for label,func in (("gain",lambda y:y>1e-7),("loss",lambda y:y< -1e-7),("no_change",lambda y:abs(y)<=1e-7)):
                    arm=next((r for r in rows if r["issued"] and func(r["delta_transfer"])),None)
                    if arm:
                        examples.append(dict(kind=label,seed=g["map"],seat=g["seat"],t=arm["t"],signature=arm["signature"],y=arm["delta_transfer"]))
            byseat=summary[split].setdefault("per_map_seat",[])
            byseat.append(dict(seed=g["map"],seat=g["seat"],original_scores=g["factual_scores"],counts=g["counts"],methods={k:dict(v) for k,v in local.items()}))
    extract.write(HERE/"future_roots.json",{"procedure":"Minimum SHA per pre-registered public stratum; no labels or reward; full physical prefix lives in source tape.","roots":roster})
    summary["examples"]=examples;summary["shards"]=shards;summary["weights_sha256"]=extract.sha(HERE/"weights.json")
    extract.write(HERE/"results.json",summary)
    tr,va=summary["train"],summary["validation"]
    lines=["# 提前占住敌方重货的返库出口","",
        "只看当时已向基地靠近、仍距基地至少三步、身上有至少 100 货的敌采集工。用前两张图学习它下一拍在五个动作中会选哪一个，再在另外两张图换席核验。每拍最多改一名本方采集工，改令必须使它能在这一拍以更轻的载荷抵达对手的一个最短返库出口。所有其他工人的实令固定。","",
        "|来源与方法|联合回合|符合返航的敌工次|有合法替代的回合|实发改令|货权改善 / 无变化 / 恶化|净货权增量|每源回合图等权增量|",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for n,x in (("训练",tr),("留图",va)):
        for method,alias in (("forecast","学习预测"),("geometry","最短出口均分")):
            z=x["methods"][method];c=x["counts"];per=x["map_equal_gain_per_source_turn"][method]
            lines.append(f"|{n}／{alias}|{c['source_turns']}|{c.get('immediate_returns',0)}|{c.get('actionable_turns',0)}|{z['issued']}|{z['positive']} / {z['zero']} / {z['negative']}|{z['total']:+,.2f}|{per:+.4f}|")
    z=va["methods"]["forecast"]; c=va["counts"]; f=va["forecast_map_equal"]
    lines+=["",f"验证集有 {c.get('no_eligible_enemy_turns',0)} 拍没有符合事前条件的重货；另有 {c.get('return_but_no_interceptor_turns',0)} 拍观察到返航，但没有任何本方采集工可以合法抢到可比较出口。已有替代、却无正预期而维持 v1 的有 {z['no_positive_estimate']} 拍。真正改令后，对手实际进入指定格 {z.get('target_arrived_at_chosen_exit',0)} 次；即时货权比 v1 好的总额 {z['gross_positive']:,.2f}，恶化总额 {z['gross_negative']:,.2f}。","",
        f"把实际下一步五选一作为单独标签，验证图等权的预测命中率 {f.get('correct',0):.1%}，实际走最短出口的比例 {f.get('observed_shortest',0):.1%}。五分类 Brier：学习预测 {f.get('brier_fit',0):.4f}、最短路均分 {f.get('brier_geometry',0):.4f}，越低越好；预测可靠与抢占动作是否值得，分别看两个结果。","",
        "|留图／席|真正改令|改善 / 无变化 / 恶化|即时净货权|本方正常回库差|本方实际新采集差|当步现金分差改变|",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in va["per_map_seat"]:
        x=Counter(row["methods"]["forecast"])
        lines.append(f"|{row['seed']} / {row['seat']}|{x['issued']}|{x['positive']} / {x['zero']} / {x['negative']}|{x['total']:+,.2f}|{x['own_deposit_delta']:+,.2f}|{x['real_new_harvest_us']:+,.2f}|{x['own_cash_margin_delta']:+,.2f}|")
    lines+=["","货权只算碰撞时真正由一方转给另一方的货。赢者背包与直接给树主／基地主的现金分别核账；货物掉到空地或因地面容量上限溢出另列，税款、存入和采集也有独立账。仅以另外一方的实令观察这一步，封堵出口不会自动产生现金。这里只检验下一步，后续对手改走其他路线、弃货、自家树收入与终局 ΔU 需要真实响应。两张留图尚不足以推断总体收益。","",
        "## 留图例子与后续复验","",
        "|方向|地图／席／回合|当步比原 v1 的货权差|",
        "|---|---|---:|"]
    for x in examples:
        lines.append(f"|{dict(gain='改善',loss='恶化',no_change='无变化')[x['kind']]}|{x['seed']} / {x['seat']} / {x['t']}|{x['y']:+,.2f}|")
    lines+=["","终局抽样名单按每张图、每席及已出手／可出手但未出手两类，取事前固定公开身份哈希最小者；没有候选的分层保留空缺。回放分片给出 t−3／5／10 的工号、位置、背包、敌方实际动作以及两臂收支，候选决策不得读取当拍敌令。[预注册](protocol.json) · [下一步预测权重](weights.json) · [指标和分片哈希](results.json) · [稀疏终局抽样名单](future_roots.json)","",
        "过去十步入库预测与三步可达多路覆盖已有各自证据。此次专测离基地三步以上、一个动作能直接占其下一出口的情形。四局新开发考试的最大回撤只提供研究问题，未用于挑选任何根。[十步入库预测](../freight_horizon/README.md) · [先前三步覆盖试验](../freight_response/README.md)"]
    lines[2:2] = [f"**留图 {z['issued']} 次改令，实际打劫改善 {z['positive']} 次，恶化 {z['negative']} 次，合计货权改变 {z['total']:+,.2f}。** 在可达的本方替代动作中，预测敌工下个最短出口仍然不足以可靠截货。这批保留 v1，当前拦截头不进入正式对局。[逐一排查其他可达落点](ROUTING_AUDIT.md)", ""]
    lines += ["", "训练图中有一个真截获：621473121 席 1 的 t95，敌工 5 身带 257.22、距基地 3 格。我方工 15 从停留改向左走，敌工同拍向上，碰撞后 257.22 货进入我工背包。它在留图中未出现。"]
    (HERE/"README.md").write_text("\n".join(lines)+"\n")
    print("DONE",va["methods"],va["map_equal_gain_per_source_turn"],flush=True)


if __name__=="__main__":
    main()
