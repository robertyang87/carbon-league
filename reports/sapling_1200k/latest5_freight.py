"""Measure per-seat banking, mining and pre-swing positions on newly indexed verified replays."""
from collections import Counter, defaultdict
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from statistics import median
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/"code"))
sys.path.insert(0,str(ROOT/"reports/master_family/deep_review"))
import cnet2_feat as F
from carbon_deep_events import _actions,_reconstruct,_snapshot
from top5_turning_points import watch

ROUND=lambda x:round(float(x),3)


def game_stats(env):
    frames,cfg=env["steps"],env["configuration"]
    T=len(frames)-1
    views=[f[0]["observation"]["players"] for f in frames]
    cash=[tuple(v[s][0] for s in (0,1)) for v in views]
    size=cfg["size"]
    width=min(10,T)
    swings=[(cash[t+width][0]-cash[t+width][1])-(cash[t][0]-cash[t][1]) for t in range(T-width+1)]
    first=max(range(len(swings)),key=lambda t:abs(swings[t]))
    end=first+width
    beneficiary=int(swings[first]<0)
    deposits=[[],[]]
    exposed=[Counter(),Counter()]
    mining=[Counter(),Counter()]
    rich_rows=[[],[]]
    active_streak=[{},{}]
    streaks=[[],[]]
    flow=[defaultdict(Counter),defaultdict(Counter)]
    previous=_snapshot(frames[0],size)
    for t in range(T):
        nxt=_snapshot(frames[t+1],size)
        actions=_actions(frames[t+1])
        per_events=[_reconstruct(previous,nxt,actions,cfg,s) for s in (0,1)]
        for s,events in enumerate(per_events):
            # The actor sees its collector's current cell. Only offline
            # accounting sees logged commands, future transitions and their cash.
            own,foe=views[t][s],views[t][1-s]
            observed=frames[t][s]["observation"]["carbon"]
            homes=list(own[1].values())
            treecells={cell for p in (own,foe) for cell,age in p[3].values()}
            tolled={z for cell,age in foe[3].values() for z in (int(cell),*F.NB8[cell].tolist())}
            anyhomes=homes+list(foe[1].values())
            colliders=set()
            for e in events:
                k=e["kind"]
                n=e["step_to"]
                if k=="collision":
                    colliders.update(p["worker_id"] for p in e["participants"])
                    exposed[s]["own_cargo_to_opponent"]+=e["own_cargo_to_opponent"]
                    exposed[s]["own_cargo_to_ground"]+=e["own_cargo_to_ground"]
                    exposed[s]["enemy_cargo_captured"]+=e["enemy_cargo_captured"]
                if k in ("delivery","enemy_delivery_to_us"):
                    amount=e["net_cash"]
                    flow[s][n][k]+=amount
                    if k=="delivery" and amount>0:
                        deposits[s].append(dict(t=n,uid=e["worker_id"],cell=e["position"],amount=ROUND(amount)))
            flow[s][t+1]["tree_absorption"]=sum(frames[t+1][0]["observation"].get("trees",{}).get(tid,[None,0])[1]
                                                      for tid in views[t+1][s][3])
            workers=own[2]
            live_streak=set()
            for uid,(pos,load,role) in workers.items():
                if role!="COLLECTOR":continue
                cmd=actions[s].get(uid) or "NONE"
                aft=views[t+1][s][2].get(uid)
                d=int(F.DIST[pos,homes].min())
                expected=float(observed[pos])*max(0., cfg["initialCollectRate"]-sum(v[2]=="COLLECTOR" for v in workers.values())*cfg.get("collectDecreaseRate",0.))
                confirmed=(cmd=="NONE" and pos not in tolled and pos not in treecells and pos not in anyhomes
                           and uid not in colliders and aft and aft[0]==pos and expected>0 and abs(aft[1]-load-expected)<.025)
                if confirmed:
                    hist=active_streak[s].setdefault(uid,[t,pos,0,0.])
                    hist[2]+=1;hist[3]+=aft[1]-load;live_streak.add(uid)
                    mining[s]["clean_harvest_steps"]+=1
                    mining[s]["clean_harvest_carbon"]+=aft[1]-load
                if (T-t<d+6 or not 1<=d<=5 or pos in treecells or pos in tolled or pos in anyhomes or
                        expected<15. or not 5<=load<=100 or any(v[2]=="COLLECTOR" and v[1]<=load+35 and F.DIST[v[0],pos]<=2 for v in foe[2].values())):
                    continue
                dest=F.DEST[pos,F.MOVES.index(cmd)]
                typ="stay" if cmd=="NONE" else "toward_home" if F.DIST[dest,homes].min()<d else "other"
                rich_rows[s].append(dict(t=t,uid=uid,action=typ,load=ROUND(load),one_more_gross=ROUND(expected),
                                         harvest_confirmed=bool(confirmed)))
            for uid in set(active_streak[s])-live_streak:
                streaks[s].append((uid,*active_streak[s].pop(uid)))
        previous=nxt
    for s in (0,1):
        streaks[s].extend((u,*v) for u,v in active_streak[s].items())
    byseat=[]
    for s in (0,1):
        ds=deposits[s]
        windows=[e for e in ds if first<e["t"]<=end]
        lead=max(windows,key=lambda e:e["amount"]) if windows else None
        rich=Counter(r["action"] for r in rich_rows[s])
        for row in rich_rows[s]:
            if row["action"]=="toward_home" and any(e["uid"]==row["uid"] and row["t"]<e["t"]<=row["t"]+5 for e in ds):
                rich["toward_then_banked_within_five"]+=1
            if row["action"]=="stay" and row["harvest_confirmed"]:
                rich["stayed_and_harvested"]+=1
        # Each maximal stay sequence contributes once. Harvest after a collision
        # or tree siphon is deliberately outside this lower-bound account.
        chains=[v for v in streaks[s] if v[3]>=2]
        chain=max(chains,key=lambda v:v[4]) if chains else None
        bank_sorted=sorted(ds,key=lambda e:-e["amount"])
        counts=[(k,sum(w[k] for w in flow[s].values()),sum(flow[s][j][k] for j in range(first+1,end+1))) for k in ("delivery","enemy_delivery_to_us","tree_absorption")]
        fleet=[]
        for t in sorted({max(0,first-k) for k in (10,5,3,0)}|{end}):
            own=views[t][s];opp=views[t][1-s]
            homes=list(own[1].values())
            close=[w for w in own[2].values() if w[2]=="COLLECTOR" and w[1]>0 and F.DIST[w[0],homes].min()<=3]
            free=[w for w in close if not any(e[2]=="COLLECTOR" and e[1]<=w[1] and F.DIST[e[0],w[0]]<=1 for e in opp[2].values())]
            fleet.append(dict(t=t,close_count=len(close),close_cargo=ROUND(sum(w[1] for w in close)),
                              close_safe_cargo=ROUND(sum(w[1] for w in free))))
        byseat.append(dict(seat=s,cash=ROUND(cash[-1][s]),delivered=len(ds),
                    median_delivery=ROUND(median(e["amount"] for e in ds)) if ds else 0,
                    largest_deliveries=bank_sorted[:3],economy={k:dict(full=ROUND(a),swing=ROUND(b)) for k,a,b in counts},
                    collision={k:ROUND(v) for k,v in exposed[s].items()},verified_harvest={k:ROUND(v) for k,v in mining[s].items()},
                    multi_step_mining_runs=len(chains),longest_mining_run=max((v[3] for v in chains),default=0),
                    richest_mining_run=dict(worker=chain[0],begin=chain[1],cell=chain[2],turns=chain[3],carbon=ROUND(chain[4])) if chain else None,
                    safe_rich_tile_decisions=rich,window_home_pipeline=fleet,
                    largest_swing_deposit=lead,
                    largest_swing_depositor_path=[watch(frames,t,s,lead["uid"],size) for t in sorted({max(0,lead["t"]-k) for k in (10,5,3,1,0)})] if lead else []))
    # Exploratory early-warning check on each worker, with ten-step spacing
    # chosen before assigning the future deposit outcome.
    for seat in (0,1):
        last={}
        lead_test=defaultdict(Counter)
        for t in range(3,T-9):
            own=views[t][seat]
            homes=list(own[1].values())
            for uid,(pos,cargo,role) in own[2].items():
                d=int(F.DIST[pos,homes].min())
                old=views[t-3][seat][2].get(uid)
                if role!="COLLECTOR" or cargo<200 or not 4<=d<=8 or old is None or t-last.get(uid,-20)<10:
                    continue
                last[uid]=t
                cat="approaching" if d<F.DIST[old[0],homes].min() else "other"
                future=[e for e in deposits[seat] if e["uid"]==uid and t<e["t"]<=t+10]
                lead_test[cat]["seen"]+=1
                lead_test[cat]["first_own_deposit_within_10"]+=bool(future)
                if future:
                    lead_test[cat]["turns_to_credit_sum"]+=future[0]["t"]-t
        byseat[seat]["distant_load_followup"]=dict(lead_test)
    return dict(transitions=T,ending_cash=list(cash[-1]),reported_rewards=env["rewards"],swing=dict(start=first,end=end,beneficiary=beneficiary,
                cash_margin_change=ROUND(abs(swings[first]))),seats=byseat)


def previous_games():
    from materialize_top5_h2h import ARCHIVE,read_repo
    full=json.loads((HERE/"top5_h2h_data.json").read_text())
    labels={g["id"]:"earlier_top5_"+g["split"] for g in full["games"]}
    historic={}
    for g in full["mutual_matches"]:
        historic[g["id"]]=(g["seed"],labels.get(g["id"],"earlier_top5_excluded"))
    for g in json.loads(read_repo(ARCHIVE/"later7.json"))["games"]:
        historic[g["id"]]=(g["seed"],"earlier_top5_later7")
    for g in json.loads((HERE/"ladder_cash_rules.json").read_text())["live"]:
        env=json.loads(gzip.decompress(read_repo(Path("data/sapling_ladder_income_nov19")/(g["game"]+".json.gz"))))
        historic[g["game"]]=(env["configuration"]["randomSeed"],"earlier_two_ladder_income")
    if len(historic)!=110:raise ValueError("previous corpus identity changed")
    return historic,set(full["strategy_ids"])


def money(x):
    return f"{x:,.0f}"


def render(report):
    lines=["# 新增 Top5 对局的运货与现金拐点","",
           f"本次逐局验过 {len(report['games'])} 份 Top5 对榜外队伍的新回放，零场 Top5 互打。旧样本以 101 场互打（其中 79 场训练、21 场验证、1 场排除）、后来的 7 场及此前另外分析的 2 场为准。每个新增 ID 单独核对过旧局身份与地图种子。本次读取的比赛全部进入诊断集，未进入本轮训练，也不可再认作未见过的正式考图。","",
           "以下用整场现金差变化幅度最大的连续十步定位拐点。窗口由赛后选出。各席位按原比赛顺序排列，胜负以回放官方奖励为准。","",
           "| 对局与席位 0 / 1 | 训练或验证归属 | 官方终分 0 / 1 | 最大十步现金差变化 | 窗内正常回库 0 / 1 | 窗内树吸收 0 / 1 |",
           "| --- | --- | --- | --- | --- | --- |"]
    for g in report["games"]:
        v=g["game"];s=v["swing"];ss=v["seats"]
        names=" / ".join(g["seat_names"])
        earnings=lambda k:" / ".join(money(r["economy"][k]["swing"]) for r in ss)
        lines.append(f"| `{g['id'][:8]}` {names} | {g['split_cn']} | {' / '.join(map(money,v['reported_rewards']))} | {s['start']}–{s['end']}，席 {s['beneficiary']} 扩大 {money(s['cash_margin_change'])} | {earnings('delivery')} | {earnings('tree_absorption')} |")
    lines.extend(["","## 拉开差距前的布局","",
                  "“安全近库货”统计距己方任一基地三格内的载货采集工，剔除其当前格可被敌方等载或轻载工下一步抵达的工人；尚未预判对方再往后的走法。每行三项数值依次对应窗口起点前 10、5、3 步。",""])
    for g in report["games"]:
        v=g["game"];sw=v["swing"];s=sw["beneficiary"];me,opp=v["seats"][s],v["seats"][1-s]
        p1={p["t"]:p for p in me["window_home_pipeline"]};p2={p["t"]:p for p in opp["window_home_pipeline"]}
        times=[max(0,sw["start"]-k) for k in (10,5,3)]
        d=" → ".join(money(p1[t]["close_safe_cargo"]-p2[t]["close_safe_cargo"]) for t in times)
        a=" → ".join(money(p1[t]["close_safe_cargo"]) for t in times)
        ref=" → ".join(money(p2[t]["close_safe_cargo"]) for t in times)
        lead=me["largest_swing_deposit"]
        witness="这一席在窗口内没有正常回库。" if not lead else f"第 {lead['t']} 步 {lead['uid']} 正常交付 {money(lead['amount'])}；完整的前 10 步载货和距库轨迹已存逐局 JSON。"
        lines.append(f"- **{g['id'][:8]} · {g['seat_names'][s]}**。其安全近库货为 {a}，另席为 {ref}，差额 {d}。{witness}")
    counts=defaultdict(Counter)
    per_map=[]
    witnesses=[]
    for g in report["games"]:
        local=defaultdict(Counter)
        for seat in g["game"]["seats"]:
            for k,v in seat["distant_load_followup"].items():
                counts[k].update(v)
                local[k].update(v)
        if all(local[k]["seen"] for k in ("approaching","other")):
            per_map.append(local["approaching"]["first_own_deposit_within_10"]/local["approaching"]["seen"]-
                           local["other"]["first_own_deposit_within_10"]/local["other"]["seen"])
        winner=g["game"]["swing"]["beneficiary"]
        path=g["game"]["seats"][winner]["largest_swing_depositor_path"]
        if path and path[0].get("present",True) and path[-1]["t"]-path[0]["t"]==10:
            witnesses.append(path[0])
    witness_note=""
    if witnesses:
        loads=[w["cargo"] for w in witnesses]; distances=[w["home_distance"] for w in witnesses]
        witness_note=(f"可回溯满十步的 {len(witnesses)} 局中，受益席在窗口内最大一笔交付的运货工，当时载货为 {min(loads):.0f}–{max(loads):.0f}、距库 {min(distances)}–{max(distances)} 格。")
    all_games=len(report["games"])
    positive=sum(g["game"]["seats"][g["game"]["swing"]["beneficiary"]]["economy"]["delivery"]["swing"]>
                 g["game"]["seats"][1-g["game"]["swing"]["beneficiary"]]["economy"]["delivery"]["swing"] for g in report["games"])
    lines.extend(["",f"{positive}/{all_games} 个事后选出的最大现金差窗口，受益席的正常回库量都较多。{witness_note}只看己方三格内的货会漏掉正在远处成形的回运。","",
                  "将这个线索扩到全部合格工人逐次核对。工人当前至少载 200、距任一己方基地 4–8 格；同一工人每十步最多取一次，且须留足十步观察后续。按过去三步是否在缩短距库距离分组，跟踪这名工人自己在接下来十步是否有正额正常回款。规则取自本批案例，结果属于探索性估计，观察之间仍有同图相关。","",
                  "| 过去三步表现 | 工人观察次数 | 十步内同工人正常回库 | 条件比例 |",
                  "| --- | ---: | ---: | ---: |"])
    for label,k in (("距库缩短","approaching"),("持平或变远","other")):
        n=counts[k]["seen"];hit=counts[k]["first_own_deposit_within_10"]
        lines.append(f"| {label} | {n} | {hit} | {100*hit/n:.1f}% |" if n else f"| {label} | 0 | 0 | — |")
    lines.append(f"\n逐局同方向核对，有 {sum(d>0 for d in per_map)} 局缩短距库的回款比例更高、{sum(d==0 for d in per_map)} 局持平、{sum(d<0 for d in per_map)} 局更低。")
    lines.extend(["","## 采矿与回库的取舍","",
                  "下表只统计当前格再采预期至少 15 碳、当前载货 5–100、距库 1–5 格、还能多走至少六步、没有敌树抽成或已知轻载敌工两步威胁的决策。同一工人连续停留会贡献多个根。真实连续采量用本席可见碳量、实际采收率、工人前后载货逐步核对，剔除了同格碰撞和敌树抽成。","",
                  "| 对局 | 整场正常回库 0 / 1 | 合格时停采次数 0 / 1 | 朝家走且五步内真正回库 0 / 1 | 连采≥2 步的段数 0 / 1 | 对撞丢给对手的载货 0 / 1 |",
                  "| --- | --- | --- | --- | --- | --- |"])
    for g in report["games"]:
        ss=g["game"]["seats"]
        def pair(fn):return " / ".join(money(fn(v)) for v in ss)
        lines.append(f"| {g['id'][:8]} | {pair(lambda v:v['economy']['delivery']['full'])} | {pair(lambda v:v['safe_rich_tile_decisions'].get('stayed_and_harvested',0))} | {pair(lambda v:v['safe_rich_tile_decisions'].get('toward_then_banked_within_five',0))} | {pair(lambda v:v['multi_step_mining_runs'])} | {pair(lambda v:v['collision'].get('own_cargo_to_opponent',0))} |")
    if report["excluded"]:
        lines.extend(["","## 已有或缺失的记录",""])
        for item in report["excluded"]:
            lines.append(f"- `{item['id']}`，{item['reason']}。")
    lines.extend(["","现金变化还受招工、种树、抢树、碰撞、误送敌方基地等影响。正常回库、对手误送及树吸收在 JSON 分列。单一窗口内的近库货变化只能定位待检验的决策，不能给一次提前回家或延长开采赋予反事实胜率。"])
    return "\n".join(lines)+"\n"


def main(inventory,cache,out,md):
    hist,names=previous_games()
    seeds=defaultdict(list)
    for gid,(seed,cohort) in hist.items():seeds[seed].append((gid,cohort))
    manifest=json.loads(inventory.read_bytes())
    games=[];excluded=[dict(id=k,reason="先前已分析的两场天梯诊断之一，本次仅新增赛果索引") for k in manifest.get("already_exposed",[])]
    for r in manifest["games"]:
        gid=r["id"]
        if gid in hist:
            excluded.append(dict(id=gid,reason=f"旧样本 {hist[gid][1]}，此处去重"))
            continue
        if not r.get("eligible") or not r.get("path"):
            excluded.append(dict(id=gid,reason="采集清单未认证为可用新增回放"))
            continue
        packed=(cache/r["path"]).read_bytes()
        if hashlib.sha256(packed).hexdigest()!=r["sha256"]:raise ValueError(gid)
        env=json.loads(gzip.decompress(packed))
        if env["statuses"]!=["DONE","DONE"]:raise ValueError(gid+" unfinished")
        seed=env["configuration"]["randomSeed"]
        if seed!=r["seed"]:raise ValueError(gid+" seed mismatch")
        same=seeds[seed]
        games.append(dict(id=gid,seed=seed,sha256=r["sha256"],seat_names=r["seat_names"],
                          top5_seats=[s for s,v in enumerate(r["seat_names"]) if v in names],
                          earlier_map_seeds=same,split_cn="旧图种重合，诊断" if same else "新图，诊断",
                          game=game_stats(env)))
    result=dict(inventory_sha256=hashlib.sha256(inventory.read_bytes()).hexdigest(),
                historical_IDs=len(hist),scope="Verified incremental replays only. No candidate training or full-game competition. Explicit map seed overlap with known 110.",
                games=games,excluded=excluded)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    md.write_text(render(result))
    print(len(games),len(excluded))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("inventory","cache","out","md"):p.add_argument(n,type=Path)
    a=p.parse_args();main(a.inventory,a.cache,a.out,a.md)
