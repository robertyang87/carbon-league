"""Keep both paired observations grouped by the original map, seat and public decision."""
import json
from itertools import groupby
import math
from pathlib import Path
from run import HERE, ORIGINAL, PLAN, PLAN_SHA, read, sha, require


def display(value):return f"{value:+,.2f}"

def sign(value):return 0 if not value else 1 if value>0 else -1


def summarize():
    streams={item["family"]:item for item in read(HERE/"streams.json")["streams"]}
    roots=[]
    for spec in read(PLAN)["families"]:
        f=spec["family"]
        obs=[]
        for origin,wd in (("original",ORIGINAL/f),("post_result_replication",HERE/f)):
            o=read(wd/"verified.json.gz");a,b=o["arms"]["parent"],o["arms"]["alternative"]
            for arm in ("parent","alternative"):
                v=o["arms"][arm]
                official=read(wd/f"live_{arm}.json")
                require(official["score"]==v["score_by_physical_seat"] and official["statuses"]==["DONE","DONE"],"unverified settlement")
            require(abs(sum(o["cash_delta_by_channel"].values())-o["delta_margin"])<.001,"cash channel mismatch")
            tree=[];tenures=[]
            for arm,v in (("parent",a),("alternative",b)):
                by_side=[sum(row[2] for row in v["focal_tree_timeline"] if row[1]==s) for s in (0,1)]
                require(all(math.isclose(x,y,abs_tol=1e-8) for x,y in zip(by_side,v["focal_tree_income"])),"target tree receipt mismatch")
                tree.append(by_side)
                spans=[]
                for owner,segment in groupby(v["focal_tree_timeline"],key=lambda z:z[1]):
                    timeline=list(segment)
                    if owner is not None:
                        spans.append({"owner":owner,"start":timeline[0][0],"last":timeline[-1][0]})
                tenures.append(spans)
            obs.append({"sampling_stage":origin,"verified_path":str((wd/"verified.json.gz").relative_to(ORIGINAL)),
                        "verified_sha256":sha(wd/"verified.json.gz"),"joint_stream_init_sha256":o["shared_init_sha256"],
                        "terminal_scores_by_seat":[a["score_by_physical_seat"],b["score_by_physical_seat"]],
                        "terminal_margin":[a["margin"],b["margin"]],"utility":[a["u"],b["u"]],
                        "delta_margin":o["delta_margin"],"delta_u":o["delta_u"],"focal_tree_income_by_seat":tree,
                        "focal_worker_state_first_settlement":[a["path"][0]["our_target"],b["path"][0]["our_target"]],
                        "focal_tree_owner_first_settlement":[a["focal_tree_timeline"][0][1],b["focal_tree_timeline"][0][1]],
                        "focal_tree_settled_tenures_by_arm":tenures,
                        "net_cash_delta_by_channel":o["cash_delta_by_channel"],
                        "original_examiner_calls":[a["official_search_calls"],b["official_search_calls"]],
                        "completed_full_search":[a["actual_searched"],b["actual_searched"]]})
        old,new=obs
        require(old["joint_stream_init_sha256"]!=new["joint_stream_init_sha256"],"repeated stream")
        case=read(HERE/f/"live_protocol.json");r=case["root"]
        folder=ORIGINAL/f
        roots.append({"seed":r["seed"],"decision_t":r["t"],"perspective":r["seat"],"family":f,
                      "input_file":str((folder/"root_input.json.gz").relative_to(ORIGINAL)),
                      "input_sha256":sha(folder/"root_input.json.gz"),"worker":r["worker"],"target_tree":r["tree"],
                      "arms_in_order":[r["old"],r["alternative"]],"paired_observations":obs,
                      "two_stream_mean_delta_margin":(old["delta_margin"]+new["delta_margin"])/2,
                      "two_stream_mean_delta_u":(old["delta_u"]+new["delta_u"])/2,
                      "margin_sign_matches":sign(old["delta_margin"])==sign(new["delta_margin"]),
                      "utility_sign_matches":sign(old["delta_u"])==sign(new["delta_u"])})
    return {"protocol_sha256":PLAN_SHA,"distinct_maps":len({r["seed"] for r in roots}),"distinct_decision_roots":len(roots),
            "staging":"Each root has one previously observed trial and one after-result newly frozen replication. Arms and streams share the same root. Tree-state gains and realized cash channels describe the executed course of each complete rollout.",
            "roots":roots}


def render(doc):
    lines=["# 中盘树决策的第二条随机流","",
           "在已读到首轮两个负结果后，冻结两处原决策全部复测。每处额外抽取一条随机流，每条流内原令与替代令共享初始状态和搜索成本。两张不同开发图各取一个决策根，每根两对实战。所有改令结果均列出。","",
           "| 图、席和决策 | 首轮改令 Δ分差 | 新流原令分差 | 新流改令分差 | 新流 Δ分差 | 两流均值 Δ分差 | 方向相同 |",
           "|---|---:|---:|---:|---:|---:|---|"]
    for r in doc["roots"]:
        old,new=r["paired_observations"];a,b=new["terminal_margin"]
        lines.append(f"| {r['seed']} 席{r['perspective']} t{r['decision_t']} {r['arms_in_order'][0]}→{r['arms_in_order'][1]} | {display(old['delta_margin'])} | {display(a)} | {display(b)} | {display(new['delta_margin'])} | {display(r['two_stream_mean_delta_margin'])} | {'是' if r['margin_sign_matches'] else '否'} |")
    for r in doc["roots"]:
        old,new=r["paired_observations"];s=r["perspective"];o=1-s
        lines+=["","## "+("夺树，遇敌时停留还是撤离" if r["family"]=="seize_enemy_tree" else "守树，采集工停留还是离开"),"",
                f"按原行动席计，首轮 ΔU {old['delta_u']:+.6f}，追加 ΔU {new['delta_u']:+.6f}，两条流均值 {r['two_stream_mean_delta_u']:+.6f}。"]
        invariant=all(old[k]==new[k] for k in ("focal_worker_state_first_settlement","focal_tree_owner_first_settlement"))
        require(invariant,"identical frozen joint root did not reproduce the initial collision and tree owner")
        local=new["focal_worker_state_first_settlement"]
        lines+=["",("两条流的第一步都一样：原令保住工人；改令停留时它立刻阵亡。" if local[0] is not None and local[1] is None else
                     "两条流两种动作下，焦点工人均活过第一步，差别从站位和此后响应产生。")]
        for label,x in (("首轮",old),("追加",new)):
            tr=x["focal_tree_income_by_seat"];cash=x["net_cash_delta_by_channel"]
            lines+=["",f"{label}的目标树实收（己方／敌方）：原令 {tr[0][s]:.2f}／{tr[0][o]:.2f}，改令 {tr[1][s]:.2f}／{tr[1][o]:.2f}。两案终局分差的差额里，正常交付变化 {display(cash['bank'])}，树息变化 {display(cash['tree'])}；支出、对敌交付及其余资金项见逐笔账。"]
    reversed_count=sum(not r["margin_sign_matches"] for r in doc["roots"])
    lines+=["",f"两个根中有 {reversed_count} 个的终局分差方向随新流翻转。夺树根两条流里都能复现停留当步损失种树工，当步结果适合作为局部辅助标签。守树根首轮两案皆守至树末，新流中原令在 t258 丢树，敌方随后实收 32.91；停守的树保有至 t280。学习终局收益时应一并保留这些分歧。"]
    lines+=["","训练时保留每根的两对标签和采样阶段，同一根在训练／验证划分中只出现一次。每条流描述一次真实响应，对某一动作的平均优势或所有中盘局面的强弱，只靠这两图估不准。考官为原版温度 1、全搜索，墙钟预算可能使实际搜索轨迹发生变化。","",
            "[冻结协议和种子](protocol.json) · [四个随机状态的身份与相同热启动成本](streams.json) · [按决策根分组的训练标签](paired_targets.json) · [首轮实战](../results.md)",""]
    return "\n".join(lines)


if __name__=="__main__":
    result=summarize()
    (HERE/"paired_targets.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    (HERE/"README.md").write_text(render(result))
