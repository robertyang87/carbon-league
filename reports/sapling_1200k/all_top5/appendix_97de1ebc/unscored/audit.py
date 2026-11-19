"""Keep the public fault-game prefix, payout and map exposure independently of the numeric cohort."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
TOP=HERE.parents[1]
sys.path.insert(0,str(TOP))
import validation
import fetch
sys.path.insert(0,str(TOP.parent))
from turning_point_ledger import CashEvents


def main():
    cards=json.loads((HERE/"cards.json").read_bytes())
    if len(cards)!=1:
        raise ValueError("unexpected number of unscored replays")
    key,rec=next(iter(cards.items()))
    path=HERE/(key+".json.gz")
    if not path.exists():
        public=fetch.SERVER+"/replay/"+key.rsplit("-",1)[0]+"/"+key.rsplit("-",1)[1]
        env,_,_=fetch.extract(fetch.Retrieval(defaultdict(list),.4).get(public))
        path.write_bytes(gzip.compress(fetch.canonical(env),mtime=0))
    env=json.loads(gzip.decompress(path.read_bytes()))
    frames=env["steps"]; cfg=env["configuration"]; opening=frames[0][0]["observation"]
    order=rec["card"]["order"]
    if env["rewards"] != [rec["card"]["scores"][s] for s in order] or env["statuses"] != ["ERROR","DONE"]:
        raise ValueError("scorecard and original status differ")
    sim=validation.FastCarbon(cfg).reset(seed=cfg["randomSeed"])
    sim.load_observation(opening)
    sim.cash=CashEvents(sim.cash)
    books=[Counter(),Counter()]; leads=[]; partial_steps=0
    for i,pair in enumerate(frames[1:],1):
        if any(r["status"]!="ACTIVE" for r in pair):
            break
        before=list(sim.cash); sim.cash.events.clear()
        sim.step(*(r["action"] for r in pair))
        obs=pair[0]["observation"]
        if sim._players_obs()!=obs["players"] or sim.carbon!=obs["full_carbon"]:
            raise ValueError(f"public opening or engine divergence at frame {i}")
        for category,owner,amount,*_ in sim.cash.events:
            books[owner][category]+=amount
        for s in (0,1):
            if sim.obs(s)["carbon"] != pair[s]["observation"]["carbon"]:
                raise ValueError("visible carbon differs")
        leads.append(sim.cash[1]-sim.cash[0]);partial_steps=i
    if partial_steps != len(frames)-2:
        raise ValueError("unverified intermediate stop")
    official=validation.official({"path":path.name},HERE)
    if official["matched"]<partial_steps:
        raise ValueError("official Board disagrees in the live prefix")
    roles=[rec["card"]["name_"+s] for s in order]
    terrain=validation.terrain_sha256(opening["full_carbon"])
    tick=max(range(10,len(leads)),key=lambda t:abs(leads[t]-leads[t-10]))
    windows={}
    for offset in (10,5,3,0):
        t=max(0,tick+1-10-offset)
        obs=frames[t][0]["observation"]; executed=frames[t+1]
        windows[str(t)]={"players":obs["players"],"trees":obs["trees"],"visible_carbon":[r["observation"]["carbon"] for r in frames[t]],
                         "next_joint_action":[r["action"] for r in executed]}
    out={"id":key,"environment_uuid":env["id"],"seed":cfg["randomSeed"],"initial_terrain_u8_sha256":terrain,
         "archive_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"sources":rec["sources"],
         "roles":roles,"rewards":env["rewards"],"statuses":env["statuses"],"frames":len(frames),
         "verified_live_steps":partial_steps,"original_board":official,"last_active_cash":list(sim.cash),
         "cash_channels_before_fault":books,"largest_ten_turn_change":{"from_post":tick+1-10,"to_post":tick+1,
             "lead_delta_seat_1":round(leads[tick]-leads[tick-10],2),"earlier_public_states_and_orders":windows}}
    (HERE/"receipt.json").write_text(json.dumps(out,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n")
    manifest={"schema":"official-fault-prefix-map-exclusion/1","revision":"97de1ebc1834785a6648f5876ce803311354a63b",
              "games":[{"id":key,"seed":out["seed"],"sha256":out["archive_sha256"],"path":path.name,
                        "source":"official_public_get","status_by_seat":env["statuses"],"score_by_seat":env["rewards"],
                        "canonical_sha256":hashlib.sha256(fetch.canonical(env)).hexdigest(),
                        "physical_names":roles,"verified_active_transitions":partial_steps,
                        "initial_terrain_u8_sha256":terrain}]}
    (HERE/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    registry={k:out[k] for k in ("id","seed","initial_terrain_u8_sha256")}
    (HERE/"validation_maps.jsonl.gz").write_bytes(gzip.compress(validation.encoded(registry)+b"\n",mtime=0))
    (HERE/"README.md").write_text(
        f"# 小肥鱼v5-late1 对 Master6 的执行故障局\n\n第 258 步记录为 ERROR / DONE。官网给小肥鱼空分，Master6 为 7,764.09。"
        f"退出前的 {partial_steps} 次双方正常联合推进已核对行动、双方视野、现金和真实棋盘。真实开发与正式考图皆须排除种子 {cfg['randomSeed']} 及此初盘。\n\n"
        f"最后正常帧余额为 {sim.cash[0]:,.2f} 与 {sim.cash[1]:,.2f}。本局单独保留，比分、胜率和因经营策略形成的完整终局收入使用双边可核场次。"
        f"[原始回放]({path.name}) · [逐项核验、现金渠道和现金转折前 3/5/10 步](receipt.json) · [原始成绩卡](cards.json)\n")
    print(key,partial_steps,official,terrain)


if __name__=="__main__":
    main()
