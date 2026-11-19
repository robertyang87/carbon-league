"""Check ordered submissions along both realized paths and reconcile target, fleet, and final cash."""
from collections import Counter
import importlib.util
from pathlib import Path
import sys
from run import HERE, DEPLOY, BASE, ROOT, PLAN, PLAN_SHA, read, sha, retain, live, clean


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HELPER = BASE/"planter_routes_dev/endgame_responses"
sys.path.insert(0, str(HELPER))
ledger = load("native_response_cash_ledger", HELPER/"verify.py").ledger
HORIZON = DEPLOY.parent/"freight_horizon"
sys.path.insert(0, str(HORIZON))
freight = load("first_response_settlement", HORIZON/"data.py")


def native_recheck(item, arm, wd, plan):
    s = item["seat"]
    opp = 1-s
    r = read(wd/"live_protocol.json")["root"]
    raw = read(ROOT/item["source"])
    out = read(wd/f"live_{arm}.json")
    records = wd/out["action_tape"]
    tail = read(records)
    prefix = raw["actions"][:r["t"]]
    whole = prefix + tail
    assert sha(records) == out["action_sha256"]
    assert out["statuses"] == ["DONE","DONE"] and out["search_calls"] == len(tail)-1
    assert out["source_sha256"] == plan["opponent_sha256"] and out["pack_sha256"] == plan["parent_sha256"]
    assert out["original_trajectory_sha256"] == item["source_sha256"] == sha(ROOT/item["source"])
    assert out["shared_init_sha256"] == sha(wd/"live_shared_init.json")
    assert out["protocol_sha256"] == sha(wd/"live_protocol.json") and len(whole) == out["terminal_t"]
    root_input = read(wd/"root_input.json.gz")
    initial = list(map(tuple, root_input["ordered_parent" if arm=="parent" else "ordered_intervention"]))
    assert list(tail[0][s].items()) == initial
    assert list(tail[0][opp].items()) == list(root_input["fixed_simultaneous_opponent_commands"].items())
    receipts = ledger(raw,r,tail)
    assert receipts["score_by_physical_seat"] == out["score"] and abs(receipts["margin"]-out["margin"]) < .011
    sim = freight.FastCarbon(raw["game_configuration"]).reset(seed=raw["seed"])
    freight.check_cfg(raw["game_configuration"])
    actual = live.module(DEPLOY/"agent.py" if arm=="alternative" else ROOT/plan["parent_path"],
                         f"actual_submission_{s}_{arm}")
    events, commits, reasons = [], [], Counter()
    for t, joint in enumerate(whole):
        obs = sim.obs(s)
        orders = actual.agent(obs,raw["game_configuration"])
        assert list(orders.items()) == list(joint[s].items()), ("submission order mismatch",s,arm,t)
        if arm == "alternative":
            memo = actual._FREIGHT.last
            reasons[memo["reason"]]+=1
            if memo["changed"]: commits.append(t)
        if t == r["t"]:
            current = sim.workers[r["target_worker"]].carbon
            assert abs(current-r["enemy_load"]) < .0006
            sim.cash = freight.FreightCash(sim.cash)
        if t < r["t"]:
            sim.step(*joint)
        else:
            events.append(freight.add_step(sim,joint))
            assert all(abs(a-b)<.011 for a,b in zip(sim.cash,receipts["path"][t-r["t"]]["cash"]))
    assert commits == ([r["t"]] if arm=="alternative" else [])
    f,paid,gross,principal,first = freight.target(r["target_worker"],current,opp,0,events)
    relevant = [z for z in receipts["deposits"] if z["seat"] == opp and z["worker"] == r["target_worker"]]
    assert all(z["kind"] == "own_delivery" for z in relevant)
    actual10 = sum(z["amount"] for z in relevant if z["t"]<=r["t"]+10)
    assert abs(actual10-gross)<.004 and abs(sum(z["amount"] for z in relevant)-receipts["focal_enemy_deposits"])<.001
    m = dict(official_by_physical_seat=out["score"],margin=receipts["margin"],u=receipts["u"],wdl=receipts["wdl"],
             root_t=r["t"],terminal_t=out["terminal_t"],root_working_cargo=r["enemy_load"],
             target_10_gross=round(gross,5),target_10_principal=round(principal,5),
             target_10_principal_fraction=round(f,6),target_10_first_payment=first if paid else None,
             target_remaining_gross=round(sum(z["amount"] for z in relevant),4),target_receipts=relevant,
             enemy_fleet_10_actual=round(sum(z["amount"] for z in receipts["deposits"] if z["seat"]==opp
                                     and z["kind"]=="own_delivery" and z["t"]<=r["t"]+10),4),
             enemy_fleet_remaining=receipts["gross"][opp]["bank"],own_fleet_remaining=receipts["gross"][s]["bank"],
             gross_by_physical_seat=receipts["gross"],master_search_calls=out["search_calls"],
             master_search_completed=out["search_completed"],master_search_aborted=out["search_aborted"],
             master_search_single=out["search_single"],ordered_pack_replay_turns=len(whole),intervention_steps=commits,
             package_reason_counts=dict(reasons),opponent_root_observation_sha256=out["root_opponent_observation_sha256"])
    retain(wd/f"full_{arm}_actions.json.gz", whole)
    m["first_ten_target_path"] = [z["enemy_target"]["cell"] if z["enemy_target"] else None for z in receipts["path"][:10]]
    m["first_ten_interceptor_path"] = [z["our_target"]["cell"] if z["our_target"] else None for z in receipts["path"][:10]]
    m["interceptor_remaining_own_bank"] = receipts["focal_own_deposits"]
    m["full_actions_sha256"]=sha(wd/f"full_{arm}_actions.json.gz")
    retain(wd/f"cash_{arm}.json.gz",receipts)
    m["cash_ledger_sha256"]=sha(wd/f"cash_{arm}.json.gz")
    return m,tail


def seat_summary(item,plan):
    s=item["seat"]
    wd=HERE/f"seat{s}"
    pa,pt=native_recheck(item,"parent",wd,plan)
    al,at=native_recheck(item,"alternative",wd,plan)
    assert pa["opponent_root_observation_sha256"]==al["opponent_root_observation_sha256"]
    def diffs(who):
        return [item["step"]+j for j,(a,b) in enumerate(zip(pt,at)) if clean(a[who])!=clean(b[who])]
    cash_sources = {k:round(al["gross_by_physical_seat"][s][k]-al["gross_by_physical_seat"][1-s][k]
                           -pa["gross_by_physical_seat"][s][k]+pa["gross_by_physical_seat"][1-s][k],4)
                    for k in pa["gross_by_physical_seat"][0]}
    delta = {k:round(al[k]-pa[k],6) for k in ("margin","u","target_10_gross","target_10_principal",
            "target_remaining_gross","enemy_fleet_10_actual","enemy_fleet_remaining","own_fleet_remaining")}
    assert abs(delta["margin"]-sum(cash_sources.values()))<.022
    result = dict(seed=plan["map_seed"],seat=s,at=item["step"],proposal=item["selection"],
                  parent=pa,alternative=al,delta_alternative_minus_parent=delta,
                  delta_margin_sources=cash_sources,observed_master_order_difference_at=diffs(1-s),
                  observed_own_order_difference_at=diffs(s),
                  ledger_implementation_sha256=sha(HELPER/"verify.py"),
                  target_implementation_sha256=sha(HORIZON/"data.py"))
    retain(wd/"verification.json",result)
    return result


def aggregate(plan):
    games=[read(HERE/f"seat{i['seat']}"/"verification.json") for i in plan["roots"]]
    doc=["# 部署包首次拦路的真实后续", "",
         "地图 621458920 是已曝光的开发图。冻结实际单次触发包 acb32fd1，在双网亲本 416cc31b 两条历史轨迹逐步运行，取席 0 第 225 步和席 1 第 246 步的第一次实发改令。每席一条新的考官随机流，配对比较原令与改令，共四条分支。根步保留考官当时已同时提交的命令，从下一步开始，双网和原版 Master3s2 按新局面重新决策。考官使用原搜索、温度 1 和原墙钟预算。","","| 席 / 决策 | 方案 | 目标敌工十步入库 | 其中根时已有货兑现 | 目标敌工余局总入库 | 敌全队余局回库 | 我全队余局回库 | 官方分差 |","|---|---|---:|---:|---:|---:|---:|---:|"]
    for g in games:
        p=g["proposal"]
        for arm,desc in (("parent","原令"),("alternative","改令")):
            a=g[arm]
            doc.append(f'| {g["seat"]} / t{g["at"]} | {desc} | {a["target_10_gross"]:.2f} | {a["target_10_principal"]:.2f} | {a["target_remaining_gross"]:.2f} | {a["enemy_fleet_remaining"]:.2f} | {a["own_fleet_remaining"]:.2f} | {a["margin"]:+.2f} |')
    d0,d1=(g["delta_alternative_minus_parent"] for g in games)
    if all(g["parent"]["target_10_gross"]>0 and g["alternative"]["target_10_gross"]==0 and
           g["delta_alternative_minus_parent"]["target_remaining_gross"]>0 for g in games):
        dates=[(g["parent"]["target_receipts"][0]["t"],g["alternative"]["target_receipts"][0]["t"]) for g in games]
        doc+=["",f'两次拦路都推迟了首笔交付。席 0 从 t{dates[0][0]} 推到 t{dates[0][1]}，席 1 从 t{dates[1][0]} 推到 t{dates[1][1]}。两名敌工都活着交了货，到终局各自累计交付反而增加 {d0["target_remaining_gross"]:.2f}、{d1["target_remaining_gross"]:.2f}。从这两根学到的动作标签应分别记录交付延迟、余局入款和双方总分。把“未来十步零回库”标为“抢到整笔货”会训错。']
    for g in games:
        s=g["seat"];t=g["at"];p=g["proposal"];d=g["delta_alternative_minus_parent"]
        opp=g["observed_master_order_difference_at"];src=g["delta_margin_sources"]
        doc+=["",f'## 席 {s}，t{t}',"",
              f'己方 {p["worker"]} 将 {p["baseline_command"]} 改成 {p["alternative"]}，落点 {p["landing"]}。目标 {p["target_worker"]} 当时载 {p["target_forecast"]["cargo"]:.2f}，距己方基地 {p["target_forecast"]["home_distance"]} 格，模型给出十步回库概率 {p["target_forecast"]["p_deposit_10"]:.3f}。改令相对原令，目标十步实收入库改变 {d["target_10_gross"]:+.2f}，起始已有货的兑现改变 {d["target_10_principal"]:+.2f}，余局该工总交付改变 {d["target_remaining_gross"]:+.2f}。同期敌全队十步回库改变 {d["enemy_fleet_10_actual"]:+.2f}。',""]
        for arm,key in (("parent","原令"),("alternative","改令")):
            a=g[arm]
            history="、".join(f't{e["t"]} {e["amount"]:.2f}' for e in a["target_receipts"]) or "零笔"
            cells=" → ".join("退场" if pos is None else str(pos) for pos in a["first_ten_target_path"])
            doc.append(f'- {key}分支目标敌工的后续十个落点为 {cells}，其余局实际回库为 {history}。官方按物理席记分 {a["official_by_physical_seat"][0]:.2f} / {a["official_by_physical_seat"][1]:.2f}，推进至 t{a["terminal_t"]}，原版搜索入口调用 {a["master_search_calls"]} 次。')
        doc+=["",f'两臂对手首个不同联合命令在 t{opp[0]}，共 {len(opp)} 个回合出现差异。官方分差改变 {d["margin"]:+.2f}，两队常规回库净差的变化贡献 {src["bank"]:+.2f}，树息贡献 {src["tree"]:+.2f}，招工贡献 {src["recruit"]:+.2f}。', "",
              f'[逐项核验和目标收据](seat{s}/verification.json) · [原令全局联合命令](seat{s}/full_parent_actions.json.gz) · [改令全局联合命令](seat{s}/full_alternative_actions.json.gz) · [原令现金及两工去向](seat{s}/cash_parent.json.gz) · [改令现金及两工去向](seat{s}/cash_alternative.json.gz)']
    doc+=["","每臂从开局用对应的完整提交包重算己方动作到终局，逐步核对实体和命令的顺序。两个改令分支均只在记录的首次触发点改过一次。逐步现金另用官方解释器验证，十步本金沿同一工人追踪并扣实际树税。起始本金、该工后采的新货、对手其他工人的入款分别入账。","",
          "两席共用一张已曝光历史地图，每席仅一个随机实现。根后双方时间预算真实运行，搜索耗时也会影响分支表现。分支终局分差可用于诊断这两次局部决策，独立正式胜率仍需按预注册地图整体考试。","",
          "[选根、随机种子及指标冻结](protocol.json) · [续弈程序](run.py) · [核验程序](verify.py) · [部署包首次触发源台账](../real_roots.json)"]
    (HERE/"README.md").write_text("\n".join(doc)+"\n")
    retain(HERE/"summary.json",dict(protocol_sha256=PLAN_SHA,distinct_maps=1,paired_roots=2,continuations=4,
        results=[{k:r[k] for k in ("seat","at","delta_alternative_minus_parent","delta_margin_sources")} for r in games]))


if __name__=="__main__":
    plan=read(PLAN)
    assert sha(PLAN)==PLAN_SHA
    if len(sys.argv)>1:
        seat_summary(plan["roots"][int(sys.argv[1])],plan)
    else:
        for item in plan["roots"]:
            seat_summary(item,plan)
        aggregate(plan)
        print("Verified two roots, four completed continuations and both seats on one exposed map.")
