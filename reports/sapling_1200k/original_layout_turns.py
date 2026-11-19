"""Reconstruct lead turns of the strongest original 75:25 hybrid games with both seats' action tapes."""
from collections import Counter
from hashlib import sha256
import gzip
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/"code"))
import cnet2_feat as F
import numpy as np
from official_game_replay import make
from tree_win_gate import validate_game,verify_raw
from turning_point_ledger import CashEvents,FastCarbon,compact,sha

FOLDER=HERE/"delivery_parent_dev"
ACTOR="b54cc51f4109f19f4a7b508eafc9cf04a28b7e295a9f442c5208f0d33f10a9ca"
CENTS=lambda v:round(float(v),2)


def raw_game(row,header):
    validate_game(row);verify_raw(FOLDER,row,ACTOR,header["weight_sha256"])
    data=(FOLDER/row["raw_file"]).read_bytes()
    raw=json.loads(gzip.decompress(data))
    cfg=raw["game_configuration"];seed=row["seed"]
    side=row["candidate_seat"]
    np.random.seed(seed)
    env=make("carbon",configuration={**cfg,"randomSeed":seed})
    sim=FastCarbon(cfg).reset(seed=seed)
    sim.cash=CashEvents(sim.cash)
    states=[];credits=[];events=[];actions=raw["actions"]
    for j in range(len(actions)+1):
        for s in (0,1):
            if env._Environment__get_shared_state(s).observation!=sim.obs(s):
                raise ValueError((seed,side,j,s,"visible observation mismatch"))
        states.append(sim.full_observation()["players"])
        if j==len(actions):break
        if env.done:raise ValueError("early end")
        sim.cash.events.clear();sim.step(*actions[j]);env.step(actions[j])
        flows=[Counter(),Counter()]
        for e in sim.cash.events:flows[e[1]][e[0]]+=e[2]
        for s in (0,1):
            if abs(states[-1][s][0]+sum(flows[s].values())-sim.cash[s])>1.e-6:
                raise ValueError("cash did not reconcile")
        credits.append(flows)
        events.append(dict(positive=[e for e in sim.cash.events if e[2]>0 and e[0] in ("own_delivery","opponent_worker_delivery_to_this_base",
                                                 "tree_collision_cash","base_collision_cash")],
                           collisions=[list(e) for e in sim.last_collisions if e[2]>0]))
    if ([p.reward for p in env.states]!=row["official_score_by_seat"] or
            sha(compact(actions))!=row["actions_sha256"] or not env.done):
        raise ValueError("terminal or action identity mismatch")
    return states,credits,events,actions


def profile(states,t,side,cfg):
    own=states[t][side];foe=states[t][1-side];bases=list(own[1].values());enbase=list(foe[1].values())
    workers=own[2];their=foe[2]
    trees={p[0] for p in foe[3].values()}
    route_tax=np.array([sum(q in trees for q in (i,*F.NB8[i])) for i in range(F.N)])
    profile_workers=[]
    for uid,(pos,cargo,role) in sorted(workers.items()):
        entry=dict(id=uid,role=role,cell=pos,load=CENTS(cargo),distance=int(F.DIST[pos,bases].min()))
        if role=="COLLECTOR" and cargo>0:
            lethal=[v[0] for v in their.values() if v[2]=="COLLECTOR" and v[1]<=cargo]
            dangers=F.DIST[:,lethal].min(axis=1)<=1 if lethal else np.zeros(F.N,bool)
            # Dynamic program over strictly distance-decreasing routes to either
            # home. Projection freezes visible threats; future enemy actions are unknown.
            ds=F.DIST[:,bases].min(axis=1)
            paths=np.full(F.N,np.inf)
            paths[bases]=np.where(dangers[bases],np.inf,0)
            for radius in range(1,entry["distance"]+1):
                cells=np.where((ds==radius)&~dangers)[0]
                nb=F.DEST[cells,1:]
                vals=np.where(ds[nb]<radius,paths[nb],np.inf)
                paths[cells]=vals.min(axis=1)+route_tax[cells]
            nxt=F.DEST[pos,1:];nxt=nxt[ds[nxt]<ds[pos]]
            score=float(np.min(paths[nxt])) if len(nxt) else 0.
            prev=states[max(0,t-3)][side][2].get(uid)
            entry.update(current_threat=bool(dangers[pos]),closest_enemy_collector=min([int(F.DIST[pos,v[0]]) for v in their.values() if v[2]=="COLLECTOR"] or [None]),
                         moving_closer_last3=bool(prev and ds[pos]<ds[prev[0]]),
                         own_home_short_route_visible_safe=bool(np.isfinite(score)),
                         min_tree_exposures_on_safe_short_route=int(score) if np.isfinite(score) else None)
        profile_workers.append(entry)
    tree_profile=[]
    for tid,(pos,age) in sorted(own[3].items()):
        home=int(F.DIST[pos,bases].min());enemy=int(F.DIST[pos,enbase].min())
        enemy_p=[v[0] for v in their.values() if v[2]=="PLANTER"]
        mine_c=[v[0] for v in workers.values() if v[2]=="COLLECTOR"]
        tree_profile.append(dict(id=tid,cell=pos,age=age,remaining=int(cfg["treeLifespan"]-age),
                       dist_home=home,dist_enemy_home=enemy,foe_planter_reach=bool(foe[0]>=cfg["seizeCost"] and any(F.DIST[pos,p]<=1 for p in enemy_p)),
                       friendly_collector_reach=any(F.DIST[pos,p]<=1 for p in mine_c)))
    loaded=sorted((q for q in profile_workers if q["load"]>0 and q["role"]=="COLLECTOR"),key=lambda q:-q["load"])
    safe=sum(v["load"] for v in loaded if v["distance"]<=3 and not v["current_threat"])
    lifetime=sum(v["remaining"] for v in tree_profile)
    return dict(side=side,cash=own[0],workers=len(workers),collectors=sum(v[2]=="COLLECTOR" for v in workers.values()),tree_count=len(tree_profile),
                remaining_tree_turns=lifetime,trees_in_enemy_half=sum(v["dist_enemy_home"]<v["dist_home"] for v in tree_profile),
                mature_trees_age_40plus=sum(v["age"]>=40 for v in tree_profile),uncovered_seizure_exposures=sum(v["foe_planter_reach"] and not v["friendly_collector_reach"] for v in tree_profile),
                cargo=CENTS(sum(v["load"] for v in loaded)),near_safe_cargo=CENTS(safe),
                cargo_no_safe_short_route=CENTS(sum(v["load"] for v in loaded if not v["own_home_short_route_visible_safe"])),
                most_loaded=loaded[:3],all_units=profile_workers,all_trees=tree_profile)


def interception_bound(states,t,defender,target):
    # Hindsight upper bound: compare actual future waypoints with gross travel
    # time. It ignores blockers, rival responses and moves needed to get paid.
    enemy=1-defender
    own,foe=states[t][defender],states[t][enemy]
    uid=target["id"];origin=target["cell"]
    base=set(own[1].values())|set(foe[1].values())
    available=[(u,w) for u,w in own[2].items() if w[2]=="COLLECTOR" and w[1]<target["load"]]
    strongest=None
    first_home=None
    for j in range(t+1,min(t+11,len(states))):
        w=states[j][enemy][2].get(uid)
        if w is None:break
        old=states[j-1][enemy][2][uid]
        if w[0] in foe[1].values() and old[1]>0 and w[1]==0:
            first_home=j
            break
        if w[0] in base:continue
        possible=sorted((float(g[1]),int(F.DIST[g[0],w[0]]),u,g[0]) for u,g in available if g[1]<old[1] and F.DIST[g[0],w[0]]<=j-t)
        if possible and strongest is None:
            load,dist,runner,cell=possible[0]
            strongest=dict(t=j,enemy_cell=w[0],defender=runner,defender_start=cell,
                           defender_cargo=load,distance=dist,arrive_in=j-t)
    nearest=min([int(F.DIST[v[0],origin]) for u,v in available] or [99])
    return dict(enemy=uid,cargo=target["load"],current_cell=origin,target_home_distance=target["distance"],
                nearest_lighter_distance=nearest,first_nonbase_possible=strongest,actual_home_receipt_time=first_home)


def fixed_alert(states,events,observed):
    earliest={};stats={"approaching":Counter(),"other":Counter()}
    for t in range(3,len(states)-10):
        p=states[t][observed];bases=list(p[1].values())
        for uid,(pos,cargo,kind) in p[2].items():
            old=states[t-3][observed][2].get(uid);d=min(F.DIST[pos,bases])
            if kind!="COLLECTOR" or cargo<200 or not 4<=d<=8 or old is None or t-earliest.get(uid,-20)<10:continue
            earliest[uid]=t
            cat="approaching" if d<min(F.DIST[old[0],bases]) else "other"
            yes=any(e[0]=="own_delivery" and e[1]==observed and e[3]==uid for f in events[t:t+10] for e in f["positive"])
            stats[cat]["eligible"]+=1;stats[cat]["bank_within_10"]+=int(yes)
    return stats


def window(t,states,credits,events,orders,side,cfg):
    start=max(0,t-10);end=min(len(orders),t+10)
    times=sorted({max(0,t-k) for k in (10,5,3,0)}|{end})
    timeline=[]
    for j in times:
        pairs=[profile(states,j,s,cfg) for s in (side,1-side)]
        bounds=[interception_bound(states,j,side,target) for target in pairs[1]["most_loaded"]]
        timeline.append(dict(t=j,candidate=pairs[0],examiner=pairs[1],actual_enemy_route_interception_upper_bounds=bounds))
    def interval(a,b):
        c=[Counter(),Counter()]
        for flows in credits[a:b]:
            for s in (0,1):c[s].update(flows[s])
        changes=[]
        for j in range(a+1,b+1):
            old={k:(s,*v) for s in (0,1) for k,v in states[j-1][s][3].items()}
            new={k:(s,*v) for s in (0,1) for k,v in states[j][s][3].items()}
            for k in old.keys()|new.keys():
                u,v=old.get(k),new.get(k)
                if u is None or v is None or u[0]!=v[0]:
                    who,cell,age=v or u
                    changes.append(dict(t=j,id=k,cell=cell,before=u,after=v,cause="plant" if u is None else "expiry" if v is None else "seize"))
        banking=[];hit=[]
        for j in range(a,b):
            for e in events[j]["positive"]:banking.append([j+1,*e])
            for e in events[j]["collisions"]:hit.append([j+1,*e])
        return dict(start=a,end=b,component_by_game_seat=[{k:CENTS(v) for k,v in holds.items()} for holds in c],
                    signed_opponent_advance=CENTS(sum(c[1-side].values())-sum(c[side].values())),
                    tree_ownership_changes=changes,positive_cash_events=banking,collision_cargo_events=hit)
    return dict(crossing=t,states=timeline,before=interval(start,t),after=interval(t,end))


def delivery_branch(cfg,start,actions,side,case):
    """Favourable timing/target chosen using hindsight; live avoidance only while following it."""
    pair=[]
    shooter=case["defender"];victim=case["enemy"]
    for intervene in (False,True):
        sim=FastCarbon(cfg).set_state(start);sim.cash=CashEvents(sim.cash)
        initial=list(sim.cash);own_receipts=[];captures=[];tape=[];committed=False
        for tick in range(case["root"],min(case["root"]+18,len(actions))):
            joint=[a.copy() for a in actions[tick]]
            if intervene and shooter in sim.workers:
                mine=sim.workers[shooter];obs=sim.obs(side);us,foe=obs["players"][side],obs["players"][1-side]
                end=F.DEST[mine.pos];home=list(us[1].values())
                if tick<case["meet"]:
                    dests=[case["cell"]]
                elif committed and not any(t>=captures[0][0] for t,_ in own_receipts):
                    dests=home
                else:dests=[]
                if dests:
                    allied={int(F.DEST[w[0],F.MOVES.index(joint[side].get(uid) or "NONE")]) for uid,w in us[2].items() if uid!=shooter}
                    allied.update(p for bid,p in us[1].items() if joint[side].get(bid))
                    lighter=[w[0] for w in foe[2].values() if w[2]=="COLLECTOR" and w[1]<=mine.carbon]
                    d0=F.DIST[end][:,dests].min(axis=1)
                    root=F.MOVES.index(joint[side].get(shooter) or "NONE")
                    def key(a):
                        q=int(end[a]);bad=any(F.DIST[q,z]<=1 for z in lighter)
                        return (q in allied or q in foe[1].values(),bool(bad),int(d0[a]),a!=root,a)
                    chosen=min(range(5),key=key)
                    if chosen:joint[side][shooter]=F.MOVES[chosen]
                    else:joint[side].pop(shooter,None)
                    tape.append([tick,mine.pos,F.MOVES[chosen]])
            original_victim=sim.workers.get(victim)
            expected_victim_cell=None if original_victim is None else int(F.DEST[original_victim.pos,F.MOVES.index(joint[1-side].get(victim) or "NONE")])
            sim.cash.events.clear();sim.step(*joint)
            for u,s,m,recipient,winner in sim.last_collisions:
                if (u==victim and s==1-side and recipient==side and winner==side and shooter in sim.workers
                        and sim.workers[shooter].pos==expected_victim_cell):
                    committed=True;captures.append([tick+1,CENTS(m)])
            for e in sim.cash.events:
                if e[0]=="own_delivery" and e[1]==side and e[3]==shooter and e[2]>0:
                    own_receipts.append([tick+1,CENTS(e[2])])
        pair.append(dict(case="branch" if intervene else "recorded",captures=captures,all_source_shooter_deposits=own_receipts,
                         cash_delta_by_seat=[CENTS(sim.cash[s]-initial[s]) for s in (0,1)],forced_orders=tape))
    change=[pair[1]["cash_delta_by_seat"][s]-pair[0]["cash_delta_by_seat"][s] for s in (0,1)]
    return dict(**case,arms=pair,candidate_cash_margin_change=CENTS(change[side]-change[1-side]))


def analyze(row,header):
    states,credits,events,actions=raw_game(row,header)
    raw=json.loads(gzip.decompress((FOLDER/row["raw_file"]).read_bytes()))
    side=row["candidate_seat"];cfg=raw["game_configuration"]
    m=[CENTS(x[1-side][0]-x[side][0]) for x in states]
    lasting=1+max(t for t,v in enumerate(m) if v<=0)
    sudden=max(range(len(m)-10),key=lambda a:m[a+10]-m[a])
    persistent250=next((t for t in range(len(m)-14) if min(m[t:t+15])>=250),None)
    examples=[]
    for anchor in (lasting,sudden):
        for t in (max(0,anchor-10),max(0,anchor-5),max(0,anchor-3),anchor):
            richest=profile(states,t,1-side,cfg)["most_loaded"]
            if not richest:continue
            bound=interception_bound(states,t,side,richest[0])
            meet=bound["first_nonbase_possible"]
            if meet is not None and not any(e["root"]==t for e in examples):
                examples.append(dict(root=t,meet=meet["t"],cell=meet["enemy_cell"],defender=meet["defender"],enemy=bound["enemy"]))
    sim=FastCarbon(cfg).reset(seed=row["seed"]); branches=[]
    for t,joint in enumerate(actions):
        for e in examples:
            if t==e["root"]:branches.append(delivery_branch(cfg,sim.get_state(),actions,side,e))
        sim.step(*joint)
    return dict(offline_interception_upper_bound_branches=branches,seed=row["seed"],seat=side,official_score_by_seat=row["official_score_by_seat"],margin=CENTS(row["margin"]),
                raw_sha256=row["raw_sha256"],actions_sha256=row["actions_sha256"],both_seat_observations_verified=len(states)*2,
                fixed_alert_by_role=dict(candidate=fixed_alert(states,events,side),examiner=fixed_alert(states,events,1-side)),
                crossing_values=dict(permanent_loss=lasting,largest_opponent_ten_start=sudden,first_fifteen_deficit_250=persistent250),
                permanent=window(lasting,states,credits,events,actions,side,cfg),largest=window(sudden,states,credits,events,actions,side,cfg),
                first_sustained_250=window(persistent250,states,credits,events,actions,side,cfg) if persistent250 is not None else None)


def historical():
    hist=[]
    for s in (0,1):
        path=HERE/f"it50_v3_fresh_games/620098122_{s}.json";r=json.loads(path.read_bytes())
        assert r["pack_sha256"]==ACTOR and r["audit_passed"] and r["actual_transitions"]==299
        snaps=[]
        for t,v in r["snapshots_by_step_and_seat"].items():
            snaps.append(dict(t=int(t),cash_margin=CENTS(v[s]["cash"]-v[1-s]["cash"]),
                              candidate=[v[s][k] for k in ("alive_workers","alive_trees","cargo","cash")],
                              opponent=[v[1-s][k] for k in ("alive_workers","alive_trees","cargo","cash")]))
        hist.append(dict(seed=r["map"],seat=s,margin=CENTS(r["margin"]),scores=r["official_score_by_seat"],snapshots=snaps,
                         report_sha256=sha256(path.read_bytes()).hexdigest(),actions_digest_only=r["actions_sha256"]))
    return hist


def render(result):
    text=["# 原版 Master3s2 对双网 Sapling 的失守前布局","",
          "本报告使用冻结的 it50 75%＋Sapling V3 25% 双网包，身份为 `b54cc51f…`。两席都用完整的原版 Master3s2 搜索和官方计分。本次重放已有联合动作，未启动新考试或训练。天梯 26 局只提供待检验的局面特征。","",
          "## 最接近取胜的记录与逐步证据边界","",
          "历史最佳开发实分是地图 620098122 席 1 的 −70.25，同图席 0 为 −359.01。旧 recorder 只保存了联合动作摘要、终态摘要及 60/100/200/299 步快照。原版考官含未播种采样，地图种子无法恢复这两条精确行动轨迹。","",
          "| 地图 620098122 | 第60步现金差 | 第100步 | 第200步 | 终局 |","| --- | ---: | ---: | ---: | ---: |"]
    for g in result["global_best_historical_pair"]:
        text.append("| 席 "+str(g["seat"])+" | "+" | ".join(f"{q['cash_margin']:+.2f}" for q in g["snapshots"])+" |")
    text.extend(["","在最佳的席 1 第 100 步，Sapling 现金领先 588.12，手中持货 261.92；对手持货 627.81。等到第 200 步 Sapling 现金已落后 93.52，场上仍有 10 工，树只剩 2 棵，对手有 6 棵。这能定位翻转区间，不能把中间某一步归因为漏截或错种。","",
        "下面使用同一父包具备逐动作原件且双席齐全的四局里分差最好的地图 620224311。席 1 输 653.01，同图席 0 输 1,178.07。行动摘要与原档一致；两局各 299 次联合推进、300 个状态、每个状态的两席可见观测和官方终分均经官方引擎核对。","",
        "## 双席最后一次失去现金领先","",
        "持续失守定义为最后一个持平或领先状态的下一步。发生时点按实际终局回看；不作为上线时可见的触发条件。“安全近库货”是距本方任一基地三格以内、当前格不会被同载或更轻敌工下一步碰到的载货总额。树龄用实际规则的剩余寿命合计，不含对未来收入的预测。以下各对数字均按 Sapling / Master3s2。","",
        "| 席位 | 步 | 现金 | 工人数 | 树数 | 剩余树寿命合计 | 全队载货 | 安全近库货 |","| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for g in result["map_pair"]:
        for r in g["permanent"]["states"]:
            def pair(k):
                return " / ".join(f"{x[k]:,.0f}" for x in (r["candidate"],r["examiner"]))
            text.append(f"| {g['seat']} | {r['t']} | {pair('cash')} | {pair('workers')} | {pair('tree_count')} | {pair('remaining_tree_turns')} | {pair('cargo')} | {pair('near_safe_cargo')} |")
    text.extend(["","**席 0。** 第 187 步，Sapling 仅有一棵树，位于格 58，树龄 39，剩余 11 步；对方有 5 棵，多数刚种，剩余寿命合计 226。到第 192 步，己方回款把现金重新推到领先 91.68，但对方第 0 号工已带着 759.60，距最近基地 5 格。第 197 步己方 4 棵树中格 58 的老树已 49 岁，对手 8 棵树分布两侧，有 4 棵更靠近 Sapling 的基地。此时现金只落后 84.41，对手载货却多 806.27。","",
        "**席 1。** 第 232 步，两席各 10 工，Sapling 现金还领先 242.31，对方三格内安全货为 668.04，对己方 106.11。第 242 步，己方树数甚至反超至 6 对 5，安全近库货仍是 109.40 对 617.64。随后十步对方正常回款 773.34、己方 182.74。己方同期树吸收 140.77，超过对方 50.36，仍止不住分差扩大 466.78。只把树数或当期树息写进评估，会漏掉这一席的失守。","",
        "## 最贵的一次交付：什么时候还有防守窗口","",
        "席 0 第 207 步，对手第 0 号采集工在格 13，载 970.23，距基地 7 格。第 212 步它在格 28 载 1,022.28，距库 6 格；第 214 步在格 42 距 4 格；第 217 步在格 55 距 1 格；第 218 步 1,022.28 实际入库。第 217 至 227 步对方累计回款 1,474.40，己方只有 338.30，十步多丢 1,099.85。","",
        "从第 212、214 或 217 步才下令时，即使知道对方接下来实走的每个格子，也没有现存、更轻的己方采集工能在入库前赶到任一非基地交叉格。第 207 步则还有一个几何窗口：己方空载 23 号工从格 146 出发，距对方后来经过的格 41 有 7 格。对手直行到那里只需 4 步，真实途中又采货、绕行，这名己方工才有时间到达。可见的空间关系可以用于提名候选，准确的会合时间仍有对手继续采多久的不确定性。","",
        "## 从几何上限到实际兑付","",
        "对持续失守和最大十步分差窗口的起点，以及各自前 10、5、3 步，固定考察该步对方最重的采集工。离线读取其后实际路线，挑第一个尚可赶到的非基地格，选择能到的最低载货己方采集工。然后从同一个实战状态分支，只改这名工的指令，其余单位沿归档动作；过了会合步，截获者按可见威胁择近基地回库。记录 18 步内候选现金差相对原归档的变化。这里刻意利用了未来信息，只估机会的上限。","",
        "| 己方席 | 发令步 | 目标会合步/格 | 己方工 | 真实引擎截货 | 截货工本方到账（各笔，含该工原载） | 18步相对现金差 |","| --- | ---: | --- | --- | ---: | --- | ---: |"])
    for g in result["map_pair"]:
        for r in g["offline_interception_upper_bound_branches"]:
            branch=r["arms"][1]; cap=sum(v for _,v in branch["captures"])
            paid="、".join(f"t{t} {x:.2f}" for t,x in branch["all_source_shooter_deposits"]) or "无"
            text.append(f"| {g['seat']} | {r['root']} | {r['meet']} / {r['cell']} | {r['defender']} | {cap:.2f} | {paid} | {r['candidate_cash_margin_change']:+.2f} |")
    text.extend(["","前五个成功窗口都瞄准同一名对手，不能当作五个独立样本。第 207 步分支中己方 23 号工起初为空，截获后第 223 步把 1,022.28 全部送到本方基地。另一席第 237 步，对手 18 号工载 258.28、距库 2 格，表面可在第 240 步与其交叉，但实际走位器未截获、未回款，18 步现金差变化为零。两类例子都保留作回归。","",
        "## 建议先改哪里以及如何证伪","",
        "优先给空载或低载采集工增加一个**提前站位、随后回款**的候选动作。输入使用对手工的当前货量、两座基地、过去三步位置；同时比较它立即返航与继续采一到两轮的路线，到敌方基地的最后一格不计可获收益。己方已经持货、正在守住年轻高收益树或同队已占交叉点时，要从拦截收益里扣掉放弃原任务的成本。把这类候选和原双网动作放在同一实际前缀下比较。","",
        "在独立开发地图上预先固定触发条件，先记每次公开预测的交叉格、时刻与原本动作。在不看后续指令的选择下，按全体触发根报告同工最终入库、己方被抢、被送入敌库、原工放弃的正常回款和树息。与原版 Master3s2 的同图双席整局再比较官方分差及胜负。若在已知存在机会的控制例上只提升碰撞数、却不能提高本方现金净额，或原位防树/回运损失更大，应弃用该候选。","",
        "将天梯回放里固定的简单返程信号原样用于这张开发图的两席考官：载货至少 200、距库 4–8 格、同工每 10 步最多取一次。此前三步距库缩短时，考官十步内真正入库 6/10；持平或变远时为 8/22。样本来自同一地图的两场不同对局，不能据此估计总体胜率。JSON 保留每个截点的全部工人、树坐标和年龄、现金账、可见路线与分支结果。"])
    return "\n".join(text)+"\n"


def main():
    receipt=json.loads((FOLDER/"results.json").read_bytes());header=receipt["header"]
    if header["pack_sha256"]!=ACTOR:raise ValueError("frozen actor mismatch")
    with_tape=[g for g in receipt["games"] if g["valid"] and (FOLDER/g["raw_file"]).is_file()]
    best=max(with_tape,key=lambda q:q["margin"])
    selected=[g for g in with_tape if g["seed"]==best["seed"]]
    if len(selected)!=2 or {g["candidate_seat"] for g in selected}!={0,1}:raise ValueError("no matching other seat")
    result=dict(actor_sha256=ACTOR,global_best_historical_pair=historical(),best_traceable_in_exact_actor_tape_cohort=best["seed"],
        definition="Permanent loss is the first state after the last tie/lead by Sapling. Largest ten-turn swing and first 15 consecutive states at least 250 cash behind are separate retrospective views. Profiles have no hidden board carbon or future commands. Static safe shortest routes project current public enemy reach onto every cell.",
        map_pair=[analyze(g,header) for g in selected])
    out=HERE/"original_layout_turns.json"
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    (HERE/"original_layout_turns.md").write_text(render(result))
    print([(g["seat"],g["crossing_values"]) for g in result["map_pair"]])


if __name__=="__main__":
    main()
