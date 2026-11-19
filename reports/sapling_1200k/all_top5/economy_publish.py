"""Publish a linked economic casebook after the full manifest's per-replay checks complete."""
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
import argparse
import gzip
import json

from economy_extract import HERE,KEYS,BASELINE_TOP5,atomic,compact,digest
from economy_endgame import last_assets

HEADINGS={"bank":"己工回库","donation":"对手误送本方基地","tree":"树吸收","collision_cash":"碰撞直接入账","recruit":"招工","plant":"种树","seize":"夺树","round":"舍入"}
def number(x):return f"{x:,.2f}".rstrip("0").rstrip(".")
def pair(items):return " / ".join(number(x) for x in items)
def q(x):return str(x).replace("|","／").replace("\n"," ")
def link(gid):return f"[{gid[:8]}](economy_games/{gid}.md)"
def outcome(r,s):
    if r["winner"] is None:return "平"
    return "胜" if r["winner"]==s else "负"
def write_md(path,lines):atomic(path,("\n".join(lines)+"\n").encode())


def case_page(card,r,meta):
    gid=card["id"];names=card["seat_names"];scores=card["score_by_seat"]
    lines=[f"# {q(names[0])} 对 {q(names[1])}","",
           f"`{gid}`　席 0 / 席 1　官方终分 **{pair(scores)}**。"]
    if not r:
        lines.extend(["","本场只有赛果卡，缺少通过身份校验的回放。","",f"[公开对局](https://117.72.210.134/replay/{gid.split('-')[0]}/{gid.split('-')[1]})"])
        return lines
    lines+=["",f"地图种子 `{r.get('seed')}`，{r['turns']} 个完整联合回合。回放 SHA-256 `{meta['replay_sha256']}`。"]
    if r["method"]=="unscored_replay":return lines+["",r["reason"]]
    mode="引擎重放，逐步核验现金、单位和树。" if r["method"]=="simulator_verified" else "公开行动与后继阵容核验。种树费含舍入余项；树吸收来自棋谱实际值，无法拆分敌方载货额。"
    lines+=["",mode,"","| 阶段 | 席0现金差变化 | 自家回款 0 / 1 | 树吸收 0 / 1 | 招工、种树与夺树净支出 0 / 1 |","| --- | ---: | ---: | ---: | ---: |"]
    for w in [*r["stages"],r["whole"],r["last30"]]:
        title="全场" if w is r["whole"] else "最后30步" if w is r["last30"] else f"{w['a']}–{w['b']}"
        ss=w["seats"];lines.append(f"| {title} | {w['margin0_delta']:+,.2f} | {pair(v['bank'] for v in ss)} | {pair(v['tree'] for v in ss)} | {pair(-(v['recruit']+v['plant']+v['seize']) for v in ss)} |")
    ss=r["whole"]["seats"]
    lines+=["",f"正常回库笔数为 {ss[0]['bank_count']} / {ss[1]['bank_count']}，单笔中位数 {pair(v['bank_median'] for v in ss)}。对手误送本方基地 {pair(v['donation'] for v in ss)}，碰撞时转到己方的敌货总额 {pair(v['cargo_acquired_from_enemy'] for v in ss)}，直接因碰撞入库 {pair(v['collision_cash'] for v in ss)}。敌货进工人背包与最终回库分开记账。"]
    if all(v["tree_from_enemy_carried"] is not None for v in ss):
        lines+=["",f"树吸收中直接来自敌方工人载货 {pair(v['tree_from_enemy_carried'] for v in ss)}，其余来自地面碳。"]
    terminal=r["terminal_assets"];asset=terminal["seats"]
    lines+=["","## 最后 30 步和剩余资产","",
        f"窗口 t{terminal['time_window'][0]}–{terminal['time_window'][1]}；"
        f"新种 {pair(x.get('plant',0) for x in asset)} 棵、夺树 {pair(x.get('seize',0) for x in asset)} 次、树到期或移除 {pair(x.get('expired',0) for x in asset)} 棵。"
        f"这些新购所有权在本次连续持有期内至终局共取得树吸收 {pair(x.get('last30_new_tenure_absorption',0) for x in asset)}，不含碰撞直接入账。","",
        f"终局仍在场的树 {pair(x['final_trees'] for x in asset)} 棵，仍由工人携带的碳 {pair(x['final_worker_load'] for x in asset)}。"
        "标准规则不提供主动出售树的指令，剩余寿命没有折价入库。"]
    labels=(("最大十步现金差变化",r["swing"]),("最终胜者从此持续领先的前十步",r["decisive"]))
    for caption,w in labels:
        if w is None:continue
        side=int(w["margin0_delta"]<0);other=1-side;s=w["seats"]
        diff={k:s[side][k]-s[other][k] for k in KEYS}
        largest=sorted(diff.items(),key=lambda kv:-abs(kv[1]))[:3]
        attribution="、".join(f"{HEADINGS[k]} {v:+,.1f}" for k,v in largest)
        lines+=["",f"## {caption}","",
                f"第 {w['a']}–{w['b']} 步，席 {side} 的现金差相对改善 **{number(abs(w['margin0_delta']))}**。按绝对影响取前三项为 {attribution}。",
                "","| 状态步 | 现金 0 / 1 | 单位 / 树（席0，席1） | 队内载货 0 / 1 | 三格内当前安全载货 0 / 1 | 树龄≥40棵数 0 / 1 |",
                "| ---: | ---: | --- | ---: | ---: | ---: |"]
        for pos in w["positions"]:
            z=pos["seats"]
            lines.append(f"| {pos['t']} | {pair(x['cash'] for x in z)} | {z[0]['workers']}/{z[0]['trees']}，{z[1]['workers']}/{z[1]['trees']} | {pair(x['cargo'] for x in z)} | {pair(x['safe_close_load'] for x in z)} | {pair(x['trees_age40plus'] for x in z)} |")
        for seat,event in enumerate(w["largest_receipt_paths"]):
            if event is None:
                lines.append(f"\n席 {seat} 窗口内没有正额正常回库。");continue
            trajectory=" → ".join(f"t{v[0]} 未出生或不在场" if v[1] is None else f"t{v[0]} 格{v[1]} 载{number(v[2])} 距库{v[3]}" for v in event["path"])
            lines.append(f"\n席 {seat} 最大交付是 {event['uid']} 第 {event['t']} 步入账 {number(event['amount'])}。{trajectory}。")
    lines+=["",f"终场实际现金 {pair(r['cash_end'])}。官方奖励会计入提前灭绝等终局规则。若赢家从初期起就一直领先，第二个窗口会短于十步。上述回溯按事后现金差选择；不能单凭一次回款判断当时只有一条好路线。",
            "",f"[公开对局](https://117.72.210.134/replay/{gid.split('-')[0]}/{gid.split('-')[1]}) · [同批机器可读证据](../economy_data/{gid[0]}.jsonl.gz) · [全库总览](../economy_report.md)"]
    return lines


def accum(groups,key,g,seat):
    row=groups[key];w=g["whole"]["seats"][seat];opp=g["whole"]["seats"][1-seat]
    row["n"]+=1; row[outcome(g,seat)]+=1
    for k in ("bank","tree","cargo_acquired_from_enemy","collision_cash","plant","seize","recruit"):
        row[k]+=w[k]
    row["margin"]+=g["rewards"][seat]-g["rewards"][1-seat]
    row["normal_bank_edge"]+=w["bank"]-opp["bank"]
    row["tree_edge"]+=w["tree"]-opp["tree"]
    for k,(hits,total) in g["alerts"][seat].items():
        row[k+"_hits"]+=hits;row[k+"_n"]+=total


def group_table(groups):
    lines=["| 分组 | 出场次 | 胜/平/负 | 胜率 | 平均官方分差 | 己方回款差/局 | 己方树吸收差/局 | 己方劫货/局（含未兑现） |",
           "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name,key in groups:
        n=key["n"]
        lines.append(f"| {q(name)} | {n} | {key['胜']}/{key['平']}/{key['负']} | {100*key['胜']/n:.1f}% | {key['margin']/n:+,.1f} | {key['normal_bank_edge']/n:+,.1f} | {key['tree_edge']/n:+,.1f} | {key['cargo_acquired_from_enemy']/n:,.1f} |")
    return lines


def main(manifest,work,cache):
    raw=manifest.read_bytes();catalog=json.loads(raw)
    cards=catalog["games"];groups=defaultdict(Counter)
    cohorts=Counter();methods=Counter();rows=[];missing=[]
    pieces=defaultdict(list);pages=defaultdict(list);times=[];swing_diff=[];before=[]
    src=digest((HERE/"economy_extract.py").read_bytes());alerts=defaultdict(lambda:[0,0])
    ending=defaultdict(Counter);common=Counter();extremes=[]
    for c in cards:
        gid=c["id"];file=work/(gid+".json.gz")
        skip=(c.get("frame_verified") is False and not c.get("sha256"))
        if skip or not file.exists():
            if c.get("frame_verified") or c.get("sha256"):
                raise ValueError("verified replay still waiting for economic extraction: "+gid)
            missing.append(dict(id=gid,reason="no verified replay",score_by_seat=c["score_by_seat"]))
            write_md(HERE/"economy_games"/(gid+".md"),case_page(c,None,None));pages[gid[0]].append((c,None));continue
        doc=json.loads(gzip.decompress(file.read_bytes()))
        if c.get("sha256")!=doc["replay_sha256"] or doc["extractor_sha256"]!=src:raise ValueError(gid+" inconsistent seal")
        r=doc["record"];methods[r["method"]]+=1;times.append(doc["cpu_seconds"])
        if r["method"]!="unscored_replay":
            rawpath=cache/(gid+".json.gz")
            if not rawpath.exists() or digest(rawpath.read_bytes())!=doc["replay_sha256"]:raise ValueError(gid+" final30 source mismatch")
            r["terminal_assets"]=last_assets(rawpath)
            terminal=r["terminal_assets"]
            for s,x in enumerate(terminal["seats"]):
                if c["seat_names"][s] not in BASELINE_TOP5:continue
                bucket=ending["scheduled_horizon" if terminal["scheduled_horizon"] else "early_termination"]
                bucket.update(x); bucket["appearances"]+=1
                bucket["held_trees_at_end"]+=int(x["final_trees"]>0)
                z=r["last30"]["seats"][s]
                for k in ("bank","tree","plant","seize","recruit"):
                    bucket["cash_"+k]+=z[k]

        pages[gid[0]].append((c,r));write_md(HERE/"economy_games"/(gid+".md"),case_page(c,r,doc))
        evidence={k:v for k,v in doc.items() if k!="cpu_seconds"}
        evidence.update(seat_names=c["seat_names"],cohort=c["cohort"])
        pieces[gid[0]].append(compact(evidence)+b"\n")
        if r["method"]=="unscored_replay":continue
        cohorts[c["cohort"]]+=1
        side=int(r["swing"]["margin0_delta"]<0);seats=r["swing"]["seats"]
        swing_diff.append(abs(r["swing"]["margin0_delta"]))
        common["beneficiary_more_normal_cash"]+=seats[side]["bank"]>seats[1-side]["bank"]
        common["beneficiary_more_tree_cash"]+=seats[side]["tree"]>seats[1-side]["tree"]
        common["beneficiary_won"]+=r["winner"]==side
        original_run=r['turns']; common["turns"]+=original_run
        roots=r["swing"]["positions"];start=r["swing"]["a"]
        for k in (3,5,10):
            if start<k:continue
            p=next(x["seats"] for x in roots if x["t"]==start-k)
            common[f"viewed_{k}"]+=1
            common[f"positive_safe_{k}"]+=p[side]["safe_close_load"]>p[1-side]["safe_close_load"]
        extremes.append((abs(r["swing"]["margin0_delta"]),gid,side,r["swing"]["a"],r["swing"]["b"]))
        for side,name in enumerate(c["seat_names"]):
            if name not in BASELINE_TOP5:continue
            for key in ("玩家 "+name, "席位 "+str(side),"赛果 "+outcome(r,side),"对手 "+c["seat_names"][1-side],
                       "对局类型 "+c["cohort"], "所有 Top5 出场","组合 "+name+" / 席"+str(side)+" / "+outcome(r,side)):
                accum(groups,key,r,side)
            for k,v in r["alerts"][side].items():
                alerts[k][0]+=v[0];alerts[k][1]+=v[1]
    for prefix,records in pieces.items():atomic(HERE/"economy_data"/(prefix+".jsonl.gz"),gzip.compress(b"".join(records),compresslevel=6,mtime=0))
    for prefix,records in pages.items():
        out=[f"# 逐局经济索引 {prefix}","","| 对局 | 席0 / 席1 | 官方分数 | 最大现金差窗口 |","| --- | --- | ---: | --- |"]
        for card,r in records:
            summary=f"t{r['swing']['a']}–{r['swing']['b']}，{number(abs(r['swing']['margin0_delta']))}" if r and "swing" in r else "仅赛果或无有效局面"
            out.append(f"| {link(card['id'])} | {q(' / '.join(card['seat_names']))} | {pair(card['score_by_seat'])} | {summary} |")
        write_md(HERE/("economy_index_"+prefix+".md"),out)
    n=sum(cohorts.values());idx=" · ".join(f"[{pr}（{len(pages[pr])}）](economy_index_{pr}.md)" for pr in sorted(pages))
    report=["# Top5 全量逐局经济复盘","",
            f"冻结 GitHub `{catalog['revision'][:12]}` 共 {len(cards):,} 张唯一赛果卡，{sum(methods.values()):,} 份回放完成处理，{n:,} 场得到完整现金账；仅有赛果卡 {len(missing)} 场。总计核对 {common['turns']:,} 次联合推进。每场均有独立阅读页，以下按对局 ID 首字符索引。","",idx,"",
            f"引擎逐步匹配两席现金、全部工人和树的 {methods['simulator_verified']:,} 场；经公开规则和下一步阵容核对的替代路径 {methods['public_roster_verified']:,} 场；无可计经济量的回放 {methods['unscored_replay']} 场。回放相同对局 ID 只消费一次。每个阶段的现金科目重新加总核平后才发布。","",
            "## 收益在什么时候拉开","",
            f"每局各取现金差变化幅度最大的连续十步（提前结束的短局用所有现有步数）。在 {n:,} 场中，窗口受益席的正常回库更多有 {common['beneficiary_more_normal_cash']:,} 场，树吸收更多有 {common['beneficiary_more_tree_cash']:,} 场；窗口受益席最后获胜 {common['beneficiary_won']:,} 场。窗口现金差变化绝对值中位数 {number(median(swing_diff))}。这是按结果回看的定位手段。","",
            "事件往往在比分可见变化前酝酿。距基地三格内、且当前格不在轻载敌工下一步触及范围内的货定义为安全近库货。不预知后续路线，也没有给同一局的多个工人赋予独立地图权重。","",
            "| 在窗口开始之前 | 受益席安全近库货较多 | 有完整前视距的地图数 |","| --- | ---: | ---: |"]
    for lag in (10,5,3):report.append(f"| {lag} 步 | {common[f'positive_safe_{lag}']} | {common[f'viewed_{lag}']} |")
    report+=["","以前 7 局的固定信号复算。工人当前载 ≥200，距己方任一基地 4–8 格，同一工人至少间隔 10 步；按前三步距库是否缩短分组，核该工之后十步有无正额正常回库。此全量分析包含早期训练、开发和既有验证资料，属于描述性总体。","",
             "| Top5 工人观察 | 十步内到账 / 观察次数 | 比例 |","| --- | ---: | ---: |"]
    for label,k in (("正向回库","approaching"),("持平或变远","other")):
        a,b=alerts[k];report.append(f"| {label} | {a:,} / {b:,} | {100*a/b:.1f}% |" if b else f"| {label} | 0 / 0 | — |")
    report+=["","## 最后 30 步的现金化","",
        "引擎给工人的可选动作是移动或停留。种树工停留可触发种树或夺树；树达到寿命时，碳返回对应格子。通常终局按玩家账户现金计分，灭绝另计。没有主动售树指令，也没有终局存树残值。以下按 Top5 一次出场计数，提前终止单列。"
        " [工人指令及寿命处理](../../../code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/helpers.py) · [官方计分](../../../code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.py)","",
        "| 终局类型 | Top5 出场 | 种树/夺树次数 | 搬运入库/场 | 树吸收/场 | 种树与夺树支出/场 | 终局留树/场 | 未兑付工人载货/场 |","| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for k,title in (("scheduled_horizon","到规定步限"),("early_termination","提前终止")):
        v=ending[k];den=v["appearances"]
        if den:report.append(f"| {title} | {den:,} | {v['plant']:,} / {v['seize']:,} | {v['cash_bank']/den:,.1f} | {v['cash_tree']/den:,.1f} | {-(v['cash_plant']+v['cash_seize'])/den:,.1f} | {v['final_trees']/den:.2f} | {v['final_worker_load']/den:.2f} |")
    report+=["","每局页另列末期新购树在本次持有期间实际取得的树吸收。购新树尚可缩小对手树息，是否值得支付需独立比较同一局面的替代动作。"]
    report+=["","## 按 Top5 玩家、席位和赛果分层","","互打每局在出场表计两席，所有逐局窗口和胜者总数仍按唯一比赛计一次。树从敌工载货抽取的碳已计在树吸收中。","",
             *group_table(sorted([(k,v) for k,v in groups.items() if k.startswith(("玩家 ","席位 ","赛果 ","对局类型 ","所有 "))])),"",
             "[全部对手及玩家×席位×胜负分层](economy_splits.md)","","## 最大的五个现金差窗口",""]
    for v,gid,w,a,b in sorted(extremes,reverse=True)[:5]:
        report.append(f"- {link(gid)} 在第 {a}–{b} 步由席 {w} 拉开 {number(v)}。")
    report+=["","各局页还列出最后 30 步、终局实际现金、树吸收、植树/夺树/招工投入、正常回库和误送敌库、碰撞已兑现现金与仍在工人身上的战利品。持续领先时点和最大现金窗口采用不同定义。原始归档与完整身份见本目录 manifest；经济证据按 ID 首字符分片。"]
    write_md(HERE/"economy_report.md",report)
    splits=["# 对手与席位分层","","按 Top5 出场统计。对手名字按原始赛果记录。","",*group_table(sorted([(k,v) for k,v in groups.items() if k.startswith("对手 ")],key=lambda kv:(-kv[1]["n"],kv[0]))),"","## 玩家、席位与赛果","","",*group_table(sorted([(k,v) for k,v in groups.items() if k.startswith("组合 ")]))]
    write_md(HERE/"economy_splits.md",splits)
    receipt=dict(manifest_sha256=digest(raw),revision=catalog["revision"],extractor_sha256=src,cards=len(cards),economic_games=n,
                 methods=methods,cohorts=cohorts,terminal_economy=ending,missing=missing,common_counts=common,group_totals=groups,alerts=alerts,
                 compute=dict(process_cpu_seconds=round(sum(times),3),median_cpu_seconds=round(median(times),4) if times else 0,
                              mean_cpu_seconds=round(sum(times)/len(times),4) if times else 0))
    atomic(HERE/"economy_coverage.json",compact(receipt)+b"\n")
    print(n,len(missing),methods,receipt["compute"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest",type=Path);p.add_argument("work",type=Path);p.add_argument("cache",type=Path)
    a=p.parse_args();main(a.manifest,a.work,a.cache)
