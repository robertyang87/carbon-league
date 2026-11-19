"""Trace two Krypton-Pro games and Master6's loss to it720 on already archived tapes."""
import gzip
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = Path("/workspace/sapling-all-top5-20261119")
HIGH = "0f1196d2b1d3292cc9d4192e62f2d889-1"
LOW = "e12b395124fb7d8f976adbce3a496ad5-1"
DUEL = "8d06052179a60bcc71ab86269cea85b6-1"


def load(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def roster(frame, seat):
    return frame[0]["observation"]["players"][seat]


def first_run(series, threshold, length=10):
    for t in range(len(series) - length + 1):
        if all(v >= threshold for v in series[t:t+length]):
            return t
    return None


def next_action(frames, t, s):
    return (frames[t+1][s].get("action") or {}) if t+1 < len(frames) else {}


def xy(pos):
    return [pos%15, pos//15]


def assets(frames, t, seat):
    cash,bases,workers,trees=roster(frames[t],seat)
    commands=next_action(frames,t,seat)
    desc=lambda uid,w:dict(uid=uid,role=w[2],xy=xy(w[0]),cargo=w[1],command=commands.get(uid,"NONE"))
    return dict(cash=cash,collectors=sum(w[2]=="COLLECTOR" for w in workers.values()),
                planters=sum(w[2]=="PLANTER" for w in workers.values()),trees=len(trees),
                cargo=sum(w[1] for w in workers.values()),workers=[desc(u,w) for u,w in workers.items()],
                bases=[dict(uid=uid,xy=xy(p),command=commands.get(uid,"NONE")) for uid,p in bases.items()])


def cash_window(tape,t,u,seat):
    out=[sum(row[2][seat][j] for row in tape["turns"][t:u]) for j in range(8)]
    cash=[tape["opening_cash"]]+[r[1] for r in tape["turns"]]
    if abs(sum(out)-(cash[u][seat]-cash[t][seat]))>.011:
        raise ValueError("cash components disagree with official tape")
    return dict(zip(("bank","donation","tree","collision_cash","recruit","plant","seize","round"),
                    [round(v,4) for v in out]))


def collector_trace(fs,tape,s):
    hire=[]
    for t,f in enumerate(fs):
        workers=roster(f,s)[2]
        before=roster(fs[t-1],s)[2] if t else {}
        for uid,w in workers.items():
            if uid not in before and w[2]=="COLLECTOR":
                hire.append(dict(t=t,uid=uid,xy=xy(w[0])))
    paid=[dict(t=turn[0],uid=uid,amount=round(value,4)) for turn in tape["turns"] for seat,uid,value in turn[3] if seat==s and value>0]
    first=firstbank=None
    if paid:firstbank=paid[0]
    if hire:
        first=dict(hire[0])
        uid=first["uid"]
        loss=next((t for t in range(first["t"]+1,len(fs)) if uid not in roster(fs[t],s)[2]),None)
        first["removed_at"]=loss
        first["cash_delivered"]=round(sum(r["amount"] for r in paid if r["uid"]==uid),4)
        first["first_next_rec_after_loss"]=next((r for r in hire if loss is not None and r["t"]>loss),None)
        if loss is not None:
            first["last_state"]=next(w for w in assets(fs,loss-1,s)["workers"] if w["uid"]==uid)
    return dict(first_bank=firstbank,first_collector=first,successful_collector_hires=len(hire),all_own_bank_events=len(paid),
                actual_bank=round(sum(r["amount"] for r in paid),4))


def collapse_detail(frames,tape,seat):
    counts = [sum(w[2]=="COLLECTOR" for w in roster(frame,seat)[2].values()) for frame in frames]
    last_with = max((i for i,n in enumerate(counts) if n),default=None)
    empty_from = last_with+1 if last_with is not None and last_with<len(frames)-1 else None
    receipt = [(r[0],uid,amount) for r in tape["turns"] for who,uid,amount in r[3] if who==seat]
    collisions=[]
    shifts={"UP":(0,-1),"DOWN":(0,1),"LEFT":(-1,0),"RIGHT":(1,0),"NONE":(0,0)}
    def destination(worker):
        return [(a+b)%15 for a,b in zip(worker["xy"],shifts[worker["command"]])]
    if empty_from is not None:
        for t in range(max(1,empty_from-10),empty_from):
            before=roster(frames[t-1],seat)[2]; after=roster(frames[t+1],seat)[2]
            for w in assets(frames,t,seat)["workers"]:
                if w["role"]!="COLLECTOR" or w["uid"] in before or w["uid"] in after:
                    continue
                pos=destination(w)
                rivals=[dict(**e,survived=e["uid"] in roster(frames[t+1],1-seat)[2])
                        for e in assets(frames,t,1-seat)["workers"] if destination(e)==pos]
                collisions.append(dict(t=t,own=w,destination=pos,rivals=rivals))
    return dict(last_delivery=receipt[-1] if receipt else None, empty_from=empty_from,
                cash_at_empty=roster(frames[empty_from],seat)[0] if empty_from is not None else None,
                next_cash_bounds=([min(r[0]["observation"]["players"][seat][0] for r in frames[empty_from:]),
                                   max(r[0]["observation"]["players"][seat][0] for r in frames[empty_from:])]
                                  if empty_from is not None else None),
                newly_hired_collectors_lost_next_step=collisions)


def game(info,tape,files):
    path=CACHE/info["path"]
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=info["sha256"] or tape["replay_sha256"]!=info["sha256"]:
        raise ValueError("source checksum")
    env=json.loads(gzip.decompress(raw));fs=env["steps"]
    if len(fs)-1!=len(tape["turns"]):raise ValueError("alignment")
    scores=info["score_by_seat"]
    s=info["seat_names"].index("Krypton-Pro") if info["id"]!=DUEL else info["seat_names"].index("RyanX-AIPowercore-it720-preview")
    game_cash=[r[1] for r in tape["turns"]]
    if max(abs(a-b) for a,b in zip(game_cash[-1],scores))>.011:
        raise ValueError("only played-to-score games enter this comparison")
    files[info["id"]]=(fs,tape,s)
    return dict(id=info["id"],sha256=info["sha256"],seed=info["seed"],focus_seat=s,
                seat_names=info["seat_names"],scores=scores,transitions=len(fs)-1,
                opening=[collector_trace(fs,tape,i) for i in (0,1)], recruit_price_floor=env["configuration"]["recCollectorCost"],
                late_economy=[collapse_detail(fs,tape,i) for i in (0,1)],
                last_60_cash=[cash_window(tape,max(0,len(fs)-61),len(fs)-1,i) for i in (0,1)],
                full_cash=[cash_window(tape,0,len(fs)-1,i) for i in (0,1)])


def panel(files, k,t,u):
    fs,tape,s=files[k]
    timeline=[]
    for v in sorted({max(0,t-d) for d in (10,5,3,0)}|{u}):
        timeline.append(dict(t=v,seats=[assets(fs,v,p) for p in (0,1)]))
    deposits=[dict(t=row[0],seat=seat,uid=uid,amount=round(amount,4)) for row in tape["turns"][t:u] for seat,uid,amount in row[3]]
    return dict(start=t,end=u,by_seat=[cash_window(tape,t,u,p) for p in (0,1)],timeline=timeline,bank_receipts=deposits)


def render(results):
    games=results["games"]
    rows=["# 同名 Krypton-Pro 的两种结果与 Master6 的一场失利","",
          "三局按真实双方现金复算。表中 t 是该时点已经结算完的步数；动作栏记录从 t 到 t+1 的已发命令。坐标按左上角起算的物理格 (列, 行)，两席使用同一张地图坐标。","",
          "| 对局与物理席 | 终分 | 首次成功回库 | 首名采集工与实收 | 移除后第一名新增采集工 | 全场正常回库 / 树收入 |",
          "|---|---:|---|---|---|---:|"]
    for g in games:
        for s,o in enumerate(g["opening"]):
            first,bank=o["first_collector"],o["first_bank"]
            if first:
                uid=first["uid"];desc=f"t{first['t']} 出现，"+(f"t{first['removed_at']} 消失" if first["removed_at"] is not None else "存活至终局")+f"，回库 {first['cash_delivered']:,.2f}"
                n=first["first_next_rec_after_loss"];recruit="全场无后续招到" if not n else f"t{n['t']}，{n['uid']}"
                if first["removed_at"] is None:recruit="首工存活"
            else:desc=recruit="无"
            dep="全局为零" if bank is None else f"t{bank['t']}，{bank['amount']:,.2f}"
            b=g["full_cash"][s]
            rows.append(f"| [{g['seat_names'][s]} / {s}]({g['id']}.md) | {g['scores'][s]:,.2f} | {dep} | {desc} | {recruit} | {b['bank']:,.2f} / {b['tree']:,.2f} |")
    rows += ["","## 何时拉开",""]
    for g in games:
        snap=g["window"];s=g["focus_seat"];a=snap["by_seat"][s];b=snap["by_seat"][1-s]
        rows.append(f"- **{g['seat_names'][s]}，{g['id'][:8]}，t{snap['start']}–{snap['end']}。** 自己 / 对手正常回库 {a['bank']:,.2f} / {b['bank']:,.2f}，树收入 {a['tree']:,.2f} / {b['tree']:,.2f}，招工支出 {abs(a['recruit']):,.2f} / {abs(b['recruit']):,.2f}，种夺支出 {abs(a['plant']+a['seize']):,.2f} / {abs(b['plant']+b['seize']):,.2f}。窗口对双方现金差的净贡献为 {sum(a.values())-sum(b.values()):+,.2f}。")
    contrast=results["krypton_gap"]
    rows.extend(["",f"同名 Krypton 在两张图上的本方现金（6,087.49 局减 5.50 局）第一次达到 100 且连续保持十个已结算状态为 t{contrast['first_sustained_100']}，同口径 500 分为 t{contrast['first_sustained_500']}。两局窗口均截至首次持续达到 100 的时点，回溯此前十步。配对的是两个不同地图与物理席，这个对照只能定位收入差何时出现。", "",
                "it720 一局取其现金领先后再未被 Master6 追平的第一步，列前十步到账。首工消失后出现的新采集工可证明补员发生；实际招工动机从回放无法识别。","",
                "## 转折前账目与真实动作","",
                "| 场次 / 席 | 观察时刻 | 现金 | 采集 / 种植 / 树 | 最多载货采集工的位置、货与实发动作 | 基地实发招工 |",
                "|---|---:|---:|---:|---|---|"])
    for g in games:
        for snap in g["window"]["timeline"]:
            for seat,v in enumerate(snap["seats"]):
                workers=sorted([w for w in v["workers"] if w["role"]=="COLLECTOR"],key=lambda w:(-w["cargo"],w["uid"]))
                demo="无采集工" if not workers else f"{workers[0]['uid']} {tuple(workers[0]['xy'])}，{workers[0]['cargo']:,.2f}，{workers[0]['command']}"
                hired="、".join(b["command"] for b in v["bases"] if b["command"]!="NONE") or "无"
                rows.append(f"| {g['id'][:8]} / {seat} | {snap['t']} | {v['cash']:,.2f} | {v['collectors']} / {v['planters']} / {v['trees']} | {demo} | {hired} |")
    recover=games[1]["late_economy"][games[1]["focus_seat"]]
    first=games[0]["late_economy"][games[0]["focus_seat"]]
    rows.extend(["","## 接连补员为何仍停产","",
                 f"Krypton-Pro 的 5.50 分局最后一笔正常回库在 t{recover['last_delivery'][0]}，到账 {recover['last_delivery'][2]:.2f}。"
                 f"从 t{recover['empty_from']} 起直到比赛打满 299 步，本方一直没有采集工。"
                 f"这期间现金范围 {recover['next_cash_bounds'][0]:.2f}–{recover['next_cash_bounds'][1]:.2f}，"
                 f"重新招采集工至少要 {games[1]['recruit_price_floor']:.2f}。6,087.49 分局仍在 t{first['last_delivery'][0]} 回库。",""])
    for hit in recover["newly_hired_collectors_lost_next_step"]:
        own=hit["own"];other="、".join(f"{e['uid']} 从 {tuple(e['xy'])} 发 {e['command']}，载货 {e['cargo']:.2f}，推进后"+("存活" if e["survived"] else "消失") for e in hit["rivals"])
        rows.append(f"- t{hit['t']}，刚招到的 {own['uid']} 在 {tuple(own['xy'])}，载货 {own['cargo']:.2f}，发 {own['command']} 前往 {tuple(hit['destination'])}，下一帧消失。同格的真实对手命令为 {other or '无'}。")
    i,m=games[2]['last_60_cash']
    rows.extend(["", f"在 it720 的胜局中，末 60 步双方都还在采运。it720 / Master6 正常回库 {i['bank']:,.2f} / {m['bank']:,.2f}，树收入 {i['tree']:,.2f} / {m['tree']:,.2f}，招工支出 {-i['recruit']:,.2f} / {-m['recruit']:,.2f}，种夺支出 {-i['seize']-i['plant']:,.2f} / {-m['seize']-m['plant']:,.2f}。即使对面有更高树息，现金领先仍从 t{results['duel_first_final_lead']} 保持到终局。"])
    rows.extend(["","所有在场工的物理动作、当步仓库实发命令、真实回库工号以及八项现金分解均可逐笔复算。表内动作只使用当前可见状态定位；窗口时点由观测到的整局结果选出，不能当成实时选动作收益。[三局逐项明细](contrasts.json)",""])
    return "\n".join(rows)


def main():
    ids=(HIGH,LOW,DUEL)
    infos={g["id"]:g for g in load(HERE/"manifest.json")["games"]}
    tapes={g["id"]:g for g in load(HERE/"turn_cash.json.gz")["games"]}
    files={}
    games=[game(infos[k],tapes[k],files) for k in ids]
    def cash(k,t):
        fs,tape,s=files[k]
        return roster(fs[t],s)[0]
    common=min(games[i]["transitions"] for i in (0,1))+1
    gaps=[cash(HIGH,t)-cash(LOW,t) for t in range(common)]
    t100,t500=(first_run(gaps,v) for v in (100,500))
    if t100 is None:raise ValueError("Krypton cash paths never separated by 100")
    for g in games[:2]:
        g["window"]=panel(files,g["id"],max(0,t100-10),t100)
    fs,tape,s=files[DUEL]
    last_tied_or_behind=max(t for t,f in enumerate(fs) if roster(f,s)[0]<=roster(f,1-s)[0])
    decisive=last_tied_or_behind+1
    games[-1]["window"]=panel(files,DUEL,max(0,decisive-10),decisive)
    # Record additional pre-transition actions at the separation instant minus 3, 5 and 10.
    for g in games:
        w=g["window"]; f,_,s=files[g["id"]]
        w["timeline"]=[dict(t=t,seats=[assets(f,t,p) for p in (0,1)]) for t in sorted(
            {max(0,w["end"]-j) for j in (10,5,3,1,0)} | {w["start"]})]
    out=dict(games=games,krypton_gap=dict(first_sustained_100=t100,first_sustained_500=t500,condition="high-focus cash minus low-focus cash for 10 consecutive post-action states"),
             duel_first_final_lead=decisive)
    (HERE/"contrasts.json").write_text(json.dumps(out,ensure_ascii=False,indent=1,allow_nan=False)+"\n")
    (HERE/"contrasts.md").write_text(render(out))
    print("micro",t100,t500,decisive)


if __name__=="__main__":main()
