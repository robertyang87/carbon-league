"""Apply frozen weights. Retain both positive and adverse action examples."""
from collections import Counter
from extract import HERE, load, sha, write
from fit import shards
from metrics import evaluate
from model import NAMES, score


def run(split, model):
    rows, docs, paths = shards(split)
    maps = Counter(r["seed"] for r in rows)
    measures = Counter()
    base_only = Counter()
    contrasts = {name:Counter() for name in ("fitted","geometric")}
    contrast_map = {str(seed):{name:Counter() for name in contrasts} for seed in maps}
    examples = {}
    first_v1_change = {}
    for row in rows:
        w = 1/(len(maps) * maps[row["seed"]])
        for a in row["actions"]:
            a["p"] = score(model, a["x"], a["geometric"])
            fields = dict(zip(NAMES,a["x"]))
            enemy_reach = any(fields[role+"_"+region] for role in ("collector","planter") for region in ("on","adjacent"))
            if fields["enemy_base"] and not fields["other_friend_reaching"] and not enemy_reach:
                base_only["arms"] += 1
                base_only["deaths"] += a["died"]
                base_only["stay_arms" if a["name"]=="NONE" else "move_arms"] += 1
            err = (model["contact_constant"] * a["geometric"] - a["died"])**2
            measures["brier_constant"] += w * err / 2
            if a["geometric"]:
                measures["exposed_mass"] += w / 2
                measures["contact_error_constant"] += w * err / 2
                measures["contact_error_fit"] += w * (a["p"] - a["died"])**2 / 2
        s,m = row["actions"]
        switch = s["p"] >= .2 and s["p"] >= m["p"]+.05
        for name, choose_move in (("fitted",switch),("geometric",s["geometric"] and not m["geometric"])):
            action = m if choose_move else s
            actual = s if row["actual"]=="NONE" else m
            for stats in (contrasts[name], contrast_map[str(row["seed"])][name]):
                stats["all_roots"] += 1
                if name == "fitted" and actual["died"] != action["died"]:
                    category = str(row["seed"])+("_saved" if actual["died"] else "_lost")
                    if category not in first_v1_change:
                        first_v1_change[category] = dict(seed=row["seed"],seat=row["seat"],t=row["t"],uid=row["uid"],origin=row["origin"],
                           actual=actual["name"],chosen=action["name"],p_actual=actual["p"],p_chosen=action["p"],
                           actual_paid=actual["paid"],chosen_paid=action["paid"])
                if actual["name"] != action["name"]:
                    stats["changed"] += 1
                    stats["saved"] += actual["died"] and not action["died"]
                    stats["lost"] += not actual["died"] and action["died"]
                    stats["unaffected"] += actual["died"]==action["died"]
                    stats["net_saved"] += int(actual["died"])-int(action["died"])
                    stats["foregone_payment"] += actual["paid"]>0
                    stats["foregone_debit"] += actual["paid"]
                    stats["new_payment"] += action["paid"]>0
                    stats["new_debit"] += action["paid"]
        category = ("move" if switch else "hold")+f"_dead{int(s['died'])}{int(m['died'])}_paid{int(s['paid']>0)}"
        if category not in examples:
            examples[category] = {k:v for k,v in row.items() if k!="actions"}
            examples[category]["actions"] = [{k:v for k,v in a.items() if k!="x"} for a in row["actions"]]
    if split == "validation":
        encoded = []
        for row in rows:
            s,m = row["actions"]
            encoded.append([row["seed"],row["seat"],row["t"],row["uid"],row["actual"],m["name"],
                            s["p"],m["p"],s["died"],m["died"],s["paid"],s["geometric"],m["geometric"]])
        write(HERE/"validation_predictions.json.gz", {
            "schema":["seed","seat","t","uid","actual_v1","registered_move","p_stay","p_move","died_stay","died_move","held_debit","geometric_stay","geometric_move"],
            "weights_sha256":sha(HERE/"weights.json"),"rows":encoded})
    results = evaluate(rows)
    results["extra_contact_probability_check"] = dict(measures)
    results["base_only_alerts"] = {k:base_only[k] for k in ("arms","deaths","stay_arms","move_arms")}
    results["first_example_per_stratum"] = examples
    results["physical_actual_v1_contrast"] = {"pooled":contrasts,"by_map":contrast_map,"first_changed_alive_label":first_v1_change}
    results["source"] = [{k:v for k,v in d.items() if k!="rows"} for d in docs]
    results["input_sha256"] = {p.name:sha(p) for p in paths}
    return results


def main():
    model = load(HERE/"weights.json")
    tr, va = (run(s,model) for s in ("train","validation"))
    write(HERE/"evaluation.json.gz",dict(weights_sha256=sha(HERE/"weights.json"),train=tr,validation=va))
    cash = va["counts"]
    metrics = va["equal_map"]["all"]["metrics"]
    brier = metrics["brier"]
    extra = va["extra_contact_probability_check"]
    lines = [
        "# 种植工下一拍会不会阵亡","",
        f"两张验证图交换双方席位，共有 **{cash['pairs']:,} 个可比停／移局面**，来自 {cash['worker_instances']} 个实际种植工实例。每根固定对手同时行动和其余友军实令，从同一张棋盘分别算停留和移动。两张训练图的标签与权重落盘后才提取第二批标签。团队事前已知精选分支有七次付款失败时阵亡，四张图均属已曝光开发域。[冻结协议](protocol.json) · [轻量权重](weights.json) · [全量指标](evaluation.json.gz) · [逐根验证预测与实付](validation_predictions.json.gz)","",
        "|分组|活种植工根|可比根|预留后可付|停留实际付款|停死、移活|停活、移死|两臂都死|",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for data,name in ((tr,"训练"),(va,"验证")):
        for seed,g in data["maps"].items():
            b=g["all"]; e=b["counts"]; p=b["metrics"]["paired_deaths"]
            n=sum(s["source_counts"]["source_roots"] for s in data["source"] if s["seed"]==int(seed))
            lines.append(f"|{name} {seed}|{n:,}|{e['pairs']:,}|{e['can_pay_now']:,}|{e['actually_paid_on_stay']:,}|{p['stay_dies_move_survives']['roots']}|{p['stay_survives_move_dies']['roots']}|{p['both_die']['roots']}|")
    lines += ["",f"验证集在预留招工及其余种植工意向后仍够支付的 {cash['can_pay_now']:,} 根中，停留时 {cash['can_pay_now_and_stay_paid']:,} 次真实付款，{cash['can_pay_now_and_stay_died']} 次阵亡，{cash['can_pay_now_and_stay_alive_unpaid']} 次工人存活却未付。另外 {cash['without_can_pay_now_and_stay_paid']} 根位于保守预算机会集外但确实付了款。","",
        "## 风险预测与行动取舍","",
        "|方法|检出死亡|漏报死亡|误报存活|正确放行|告警精确率|死亡召回率|",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for k,label in (("geometric","一格可达或敌基地"),("fitted_p20","逻辑头 p≥0.20"),("fitted_p50","逻辑头 p≥0.50")):
        a=metrics["alerts"][k]["both_arms"]
        c=a["counts"]
        prec="—" if a["precision"] is None else f"{a['precision']:.1%}"
        lines.append(f"|{label}|{c['tp']}|{c['fn']}|{c['fp']}|{c['tn']}|{prec}|{a['recall']:.1%}|")
    base = va["base_only_alerts"]
    lines += ["", f"登记的几何对照会对敌基地单独发出告警。这是保守动作选择：种植工站到敌基地本身可以存活，敌人若能招工或同一拍另有工人抵达，才可能发生碰撞。在验证集，单凭敌基地、周围没有可达敌工且没有已知友方冲突的动作有 {base['arms']} 个，实死 {base['deaths']} 个；其中停留 {base['stay_arms']} 个、移动 {base['move_arms']} 个。", "", f"按图等权，每根内停、移各占一半。Brier 分数为逻辑头 {brier['fitted']:.5f}、二值几何告警 {brier['binary_geometric']:.5f}。若在训练集估算可达范围内的一个统一阵亡率（{model['contact_constant']:.2%}），验证集得 {extra['brier_constant']:.5f}。单看可达范围，逻辑头为 {extra['contact_error_fit']/extra['exposed_mass']:.5f}，统一概率为 {extra['contact_error_constant']/extra['exposed_mass']:.5f}，均以越低越好。[概率对照预先说明](probability_baseline.json)","",
        "|从默认停留出发的选择|选择移动|移动救活|移动致死|净少阵亡|放弃的实际付款机会|这些购买的总价|",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for k,label in (("fitted_p20_delta05","停留风险≥0.20 且移动至少降0.05"),("geometric","仅停留格可达、移动格不可达")):
        a=metrics["policy"][k]
        lines.append(f"|{label}|{a['selected_move']}|{a['saved_by_moving']}|{a['deaths_caused_by_moving']}|{a['net_saved_vs_always_stay']}|{a['foregone_actual_stay_purchases']}|{a['foregone_purchase_debit']:,.2f}|")
    lines += ["", "原策略 v1 有时已经选择移动。与真正记录的 v1 逐工命令比较，两种冻结决策与全部真实 v1 出手逐根比较。","",
              "|相对 v1|全部根|改令|从死到活|从活到死|改令后生死相同|放弃原实付笔数|新增实付笔数|",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k,label in (("fitted","固定逻辑头"),("geometric","固定几何规则")):
        entry=va["physical_actual_v1_contrast"]["pooled"][k]
        lines.append("|"+label+"|"+"|".join(str(entry[f]) for f in ("all_roots","changed","saved","lost","unaffected","foregone_payment","new_payment"))+"|")
    lines += ["","逐图按原始席位及帧序，取第一例救活和第一例误判，两种方向都保留。","",
        "|地图、物理席、根步|工号、原格|v1 → 逻辑头|各命令预计死亡率|双引擎实际一步结果|",
        "|---|---|---|---|---|"]
    for ident,c in sorted(va["physical_actual_v1_contrast"]["first_changed_alive_label"].items()):
        outcome = "从阵亡到存活" if ident.endswith("saved") else "从存活到阵亡"
        lines.append(f"|{c['seed']} / {c['seat']} / {c['t']}|{c['uid']} / {c['origin']}|{c['actual']} → {c['chosen']}|{c['p_actual']:.2f} → {c['p_chosen']:.2f}|{outcome}|")
    lines += ["","新增实付均为各自独立单步里花出去的钱。在真实策略下反复增付会改变余额、树价、未来位置及对手出招，不能按笔数累加收入。","",
              "购买总价是原地时能支付的价格。延期少收的树款、原地牵制对手的收益，以及后续能否夺回，均需要双方重新应对到终局。两张验证地图上的工人、回合和席位相关，逐图结果支持诊断。","",
        "## 标签与复现","",
        "每拍从真实前缀恢复完整资源和按创建者存储的树与工人编号，事实与每个替代联合动作均分别由官方 Board.next() 和 FastCarbon.step() 推进。逐拍核对两席全部资产、树账和碳图。本人停留、存活、形成新树或夺得敌树且出现同工号同格的扣款，才记实付。已到寿命的树可枯死后重种。敌方当拍命令和完整碳图仅用于标签。预测输入为本席可见局面、己方拟定命令和此前最多三步的可见运动。","",
        "选根沿用真实前缀，占位冲突的停留或替代移动排除。指定种植工以外的队友保留原命令。每局排除分母、逐根风险、实付、反向例由分片、权重和指标文件重建。复现顺序为 extract.py train、fit.py、extract.py validation、report.py。model.death_risk(weights, public_obs, history, own_orders, uid, cmd) 可直接调用，无引擎或科学计算库依赖。实际策略调用耗时尚未测量。", "", "规则与标签：[官方碰撞、移动及种／夺树](../../../../../code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/helpers.py#L1019) · [按工号扣款](../../../turning_point_ledger.py) · [官方与独立引擎逐步对照](extract.py)。"]
    (HERE/"README.md").write_text("\n".join(lines)+"\n")
    print("validation",cash, brier,flush=True)


if __name__ == "__main__":
    main()
