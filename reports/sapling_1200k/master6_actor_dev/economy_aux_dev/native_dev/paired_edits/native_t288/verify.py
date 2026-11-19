"""Independent official replay and two-seat cash and worker settlement for both registered t288 roots."""
from collections import Counter
import json
from pathlib import Path
import run

H=run.HERE
def confirm(arm,tape,root,registration):
    d=H/arm;proof=run.read(d/"outcome.json");first=run.read(d/"root.json")
    run.check(proof["root_sha256"]==run.sha(d/"root.json") and proof["protocol_sha256"]==run.sha(H/"protocol.json")
              and proof["shared_sha256"]==first["shared_sha256"]==run.sha(H/"shared.json"),"changed root or protocol")
    runs=sorted(d.glob("turn_*.json.gz"))
    attempts=sorted(d.glob("search_*.json"))
    run.check([run.sha(p) for p in runs]==proof["completed"] and [run.sha(p) for p in attempts]==proof["issued"]
              and len(runs)==len(attempts)==proof["search_calls"]<=10,"search budget mismatch")
    rows=[run.read(p) for p in runs]
    tail=[first["joint"],*[r["joint"] for r in rows]]
    s,t=run.SEAT,run.T
    run.check(tail[0][0]==tape["actions"][t][0] and tail[0][1]==registration["arms"][arm],"invalid root")
    engine=run.live.begin(tape)
    for joint in [*tape["actions"][:t],*tail]:
        run.check(not engine.done,"orders after official terminal")
        engine.step(joint)
    scores=[x.reward for x in engine.states]
    run.check(engine.done and [x.status for x in engine.states]==proof["status"]==["DONE","DONE"]
              and scores==proof["official_scores"] and engine.states[0].observation.step==proof["final_t"],"different official reward")
    cfg=tape["game_configuration"];o=run.analysis.official;state=o.initial(tape)
    sim=run.analysis.FastCarbon(cfg).reset(seed=run.SEED)
    policy=run.live.module(run.DEV/"candidate.py","checked_friendly_continuation")
    for step,joint in enumerate(tape["actions"][:t]):
        obs=sim.obs(s)
        if step==0:run.check(policy.agent(obs,cfg)==joint[s],"opening action differs")
        else:policy._FEAT.observe(obs)
        state=o.helpers.Board(state,cfg,joint).next().observation;sim.step(*joint)
    c=[Counter(),Counter()];focal={};timeline=[]
    initial=[p[0] for p in state["players"]];focal_ids={e["unit"] for e in registration["edited"]}
    for i,joint in enumerate(tail):
        step=t+i; obs=sim.obs(s); moves=policy.agent(obs,cfg)
        if i==0:run.check(moves==tape["actions"][t][s],"candidate root history differs")
        else:run.check(joint[s]==moves,"common future policy differs")
        pre=[p[0] for p in state["players"]]
        with o.ledger() as ev,run.analysis.casualties() as kills:
            state=o.helpers.Board(state,cfg,joint).next().observation
            events=[list(e) for e in ev]
            deaths=list(kills)
        sim.step(*joint)
        cash=[p[0] for p in state["players"]]
        run.check(state["players"]==sim.obs(s)["players"],"independent simulator roster, trees or cash differs")
        seat_accounts=[Counter(),Counter()]
        for k,p,v,w,pos,tid in events:
            c[p][k]+=v;seat_accounts[p][k]+=v
            if w in focal_ids and abs(v)>1e-9 and k in ("own_delivery","opponent_worker_delivery_to_this_base","tree_collision_cash","base_collision_cash"):
                focal.setdefault(w,[]).append({"turn":step,"paid_to":p,"amount":v,"kind":k})
        run.check(all(abs(sum(seat_accounts[p].values())-cash[p]+pre[p])<1e-6 for p in (0,1)),"cash write mismatch")
        if i:
            rr=rows[i-1];x=rr["pre"]
            run.check(x==run.read(attempts[i-1]) and x["turn"]==step and rr["original_search_calls"]==1
                      and rr["post_cash"]==cash and x["arm"]==arm,"per-turn native ticket or cash differs")
        timeline.append({"pre":step,"post_cash":cash,"joint":joint,"cash_channels":[dict(z) for z in seat_accounts],
                         "positive_focal_cash_events":[e for e in events if e[3] in focal_ids and abs(e[2])>1e-8],
                         "focal_assets_after":{wid:state["players"][s][2].get(wid) for wid in sorted(focal_ids)},
                         "casualties":deaths})
    return {"arm":arm,"source_sha256":run.sha(d/"outcome.json"),"official_scores":scores,"step":proof["final_t"],
            "official_margin":scores[1]-scores[0],"physical_cash_before_root":initial,
            "terminal_cash":cash,"official_settlement_adjustment":[scores[p]-cash[p] for p in (0,1)],
            "U":run.live.utility(scores,s),"search_calls":len(runs),"search_wall":sum(r["wall_seconds"] for r in rows),
            "physical_cash_channels":[dict(z) for z in c],"focal_deposits":focal,"tail":timeline}


def main():
    registration=run.read(H/"protocol.json");tape,root,_,_=run.inputs()
    data={a:confirm(a,tape,root,registration) for a in run.ARMS}
    n=sum(z["search_calls"] for z in data.values())
    run.check(n<=20 and all(z["step"]-289==z["search_calls"] for z in data.values()),"exceeded allowance")
    a,c=(data[n] for n in run.ARMS)
    md=a["official_margin"]-c["official_margin"]
    by_channel={}
    for key in sorted(set.union(*[set(side) for arm in data.values() for side in arm["physical_cash_channels"]])):
        by_channel[key]=(a["physical_cash_channels"][1].get(key,0)-a["physical_cash_channels"][0].get(key,0)
                        -c["physical_cash_channels"][1].get(key,0)+c["physical_cash_channels"][0].get(key,0))
    settlement=a["official_settlement_adjustment"];other=c["official_settlement_adjustment"]
    run.check(abs(sum(by_channel.values())+settlement[1]-settlement[0]-other[1]+other[0]-md)<1e-6,"terminal delta mismatch")
    result={"schema":"registered-t288-live-two-root-final-v1","protocol_sha256":run.sha(H/"protocol.json"),
            "grant_sha256":run.sha(H/"grant.json"),"shared_sha256":run.sha(H/"shared.json"),"verifier_sha256":run.sha(Path(__file__)),
            "official_engine_sources":run.read(run.DEV/"candidate/official_replay.json")["engine_source_sha256"],
            "total_native_search_calls":n,"auxiliary_minus_control_margin":md,"auxiliary_minus_control_U":a["U"]-c["U"],
            "margin_components":by_channel,"results":data,"formal_games":0}
    run.freeze(H/"verified.json.gz",result)
    lines=["# 第 288 步两名采集工改道的原版续局","",
           "开发图 621497120 席 1，根步考官使用已经同步发出的真实命令。辅助根让 54 号向下、41 号向上；控制根让二者原地采集，当步多持货 17.15，两方现金相同。从 t289 起，两路本方均执行同一冻结辅助策略，原版 Master3s2 各自按当前实际局面重新采样、完整搜索至官方终局。"
           "考官的公开历史、信念状态和根时共同新随机流按登记重建，后续计时预算自由演进。它没有复用原开发考卷的未来指令。","",
           "| 根动作 | 官方终分，我方 / 考官 | 官方分差 | U | 后继原版搜索 | 两名焦点采集工实际交入己库 |",
           "|---|---:|---:|---:|---:|---|"]
    for arm,title in zip(run.ARMS,("辅助，两工改道","控制，两工停采")):
        z=data[arm];v=z["official_scores"];settled=[]
        for wid,events in sorted(z["focal_deposits"].items()):
            for e in events:
                if e["kind"]=="own_delivery" and e["paid_to"]==run.SEAT:
                    settled.append(f"{wid.rsplit('-',1)[-1]}号 t{e['turn']} +{e['amount']:.2f}")
        lines.append(f"| {title} | {v[1]:.2f} / {v[0]:.2f} | {z['official_margin']:+.2f} | {z['U']:.5f} | {z['search_calls']} | "+("、".join(settled) if settled else "0")+" |")
    lines+=["",f"这一对以辅助根减控制根，官方分差变化 **{md:+.2f}**，U 变化 **{a['U']-c['U']:+.5f}**。详细账按现金实际写入区分回库、树息、种植、夺树和招工，并区分两名焦点工人和全队。"
            f"共执行 {n} 次原版搜索。所有最终分数用官方解释器独立重放；每次联合推进的现金、工人和树再与第二模拟器核对。","",
            "这里只比较一张已见图、一个根和一份新的共同初始随机流；行动改变后双方都能随局面调整。正式新图样本数为零。","",
            "[两臂逐步实发、到账、伤亡和独立终分](verified.json.gz) · [研究登记及包身份](protocol.json) · [工位签收](grant.json) · [根步动作差复核](../README.md)"]
    p=H/"README.md";raw=("\n".join(lines)+"\n").encode()
    if p.exists():run.check(p.read_bytes()==raw,"changed report")
    else:p.write_bytes(raw)


if __name__=="__main__":
    main()
