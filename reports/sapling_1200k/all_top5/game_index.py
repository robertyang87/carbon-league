"""Join published analyses into a single game-level index and paginated physical-seat views."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile

HERE=Path(__file__).resolve().parent
PAGE_SIZE=55
ID_PATTERN=re.compile(r"^[0-9a-f]{32}-[0-9]+$")
MEASURES=("bank","tree","donation","collision_cash","recruit","plant","seize","round",
          "cargo_acquired_from_enemy","tree_from_enemy_carried","bank_count")
DEFENSE=("acquired_collector","collector_stolen_banked","collector_stolen_tax_paid_to_enemy_trees",
         "collision_deaths","mixed_deaths","loaded_stay_under_threat_with_safe_move",
         "chose_geometrically_safe_move","enemy_trees_seized","trees_lost_to_enemy",
         "same_side_deaths","collector_deaths","planter_deaths","loaded_decisions",
         "collector_seizure_lots","collector_lots_positive_cash","collector_stolen_carried_at_terminal",
         "collector_stolen_opponent_worker","collector_stolen_opponent_cash","collector_stolen_ground_or_destroyed",
         "acquired_tree","acquired_base","mixed_cargo_left_on_ground","enemy_base_refund_at_contact")


def unpack(file):
    with (gzip.open(file,"rt") if file.suffix==".gz" else file.open()) as stream:
        return json.load(stream)


def json_lines(file):
    with gzip.open(file,"rt") as stream:
        for line in stream:
            if line.strip(): yield json.loads(line)


def keyed(rows):
    result={}
    for r in rows:
        gid=r["id"]
        if gid in result: raise ValueError(f"duplicate game ID {gid}")
        result[gid]=r
    return result


def number(v):
    return "—" if v is None or v=="" else f"{float(v):,.2f}".rstrip("0").rstrip(".")


def cell(v):
    return str(v).replace("\\","\\\\").replace("|","\\|").replace("[","\\[").replace("]","\\]").replace("<","&lt;").replace(">","&gt;").replace("\n"," ")


def safe_json(v):
    return "" if v is None else json.dumps(v,separators=(",",":"),ensure_ascii=False,allow_nan=False)


def emit(file, data):
    if isinstance(data,str): data=data.encode()
    file.parent.mkdir(parents=True,exist_ok=True)
    if file.exists() and file.read_bytes()==data:return
    with tempfile.NamedTemporaryFile(dir=file.parent,delete=False) as tmp:
        tmp.write(data)
    Path(tmp.name).replace(file)


def ledger(source, partial):
    files={}
    def data(name):
        file=source/name
        if file.exists(): files[name]=file; return file
        if not partial: raise FileNotFoundError(file)
        return None
    manifest=unpack(data("manifest.json"))
    if not partial and manifest["state"]!="complete":raise ValueError("replay coverage still changing")
    valfile=data("validation_games.jsonl.gz")
    validation=keyed(json_lines(valfile)) if valfile else {}
    layoutfile=data("layout.json.gz")
    layout=unpack(layoutfile) if layoutfile else {}
    layouts=keyed(layout.get("games",[]))
    shards=sorted((source/"economy_data").glob("*.jsonl.gz"))
    if not shards and not partial: raise FileNotFoundError("economy_data/*.jsonl.gz")
    for f in shards:files[f.relative_to(source).as_posix()]=f
    economy=keyed(r for f in shards for r in json_lines(f))
    deffile=data("defense_games.tsv")
    defense={}
    if deffile:
        with deffile.open() as fh:
            for line,r in enumerate(csv.DictReader(fh,delimiter="\t"),2):
                key=(r["id"],int(r["seat"]))
                if key in defense or key[1] not in (0,1): raise ValueError(f"invalid defense row {key}")
                defense[key]=(line,r)
    return manifest,validation,layout,layouts,economy,defense,files


def build(source,output,partial=False):
    m,valid,layout,layouts,econ,defense,files=ledger(source,partial)
    cards=keyed(m["games"]); top=m["top5"]
    if len(cards)!=m["coverage"]["score_cards"] or len(top)!=5:raise ValueError("manifest counts differ")
    for label,keys in (("validation",valid),("layout",layouts),("economy",econ),("defense",{k[0] for k in defense})):
        if set(keys)-cards.keys():raise ValueError(f"{label} has unknown match IDs")
    by_rank=defaultdict(list); rankby={p["id"]:p["rank"] for p in top}
    rows=[]; outcomes=Counter(); completeness=Counter()
    for gid,c in sorted(cards.items()):
        if not ID_PATTERN.fullmatch(gid): raise ValueError(f"invalid match/game ID {gid}")
        ranks=[rankby.get(pid) for pid in c["seat_strategy_ids"]]
        seen={rank for rank in ranks if rank is not None}
        if not seen:raise ValueError(f"no Top Five seat for {gid}")
        scores=c["score_by_seat"]; names=c["seat_names"]
        outcome= ("平" if scores[0]==scores[1] else ("席0胜" if scores[0]>scores[1] else "席1胜")) if all(isinstance(x,(int,float)) for x in scores) else "待核"
        v=valid.get(gid); lay=layouts.get(gid); e=econ.get(gid); rec=e["record"] if e else {}
        for ref in (v,e):
            if not ref:continue
            if ref.get("seat_names",names)!=names:raise ValueError(f"physical seats mismatch for {gid}")
        if v:
            if v["score_by_seat"]!=scores:raise ValueError(f"score mismatch {gid}")
            for key,expected in (("compressed_sha256",c.get("sha256")),("canonical_sha256",c.get("canonical_sha256"))):
                if v.get(key) and expected and v[key]!=expected:raise ValueError(f"validation hash mismatch {gid}/{key}")
        if lay and lay["scores"]!=scores:raise ValueError(f"layout score mismatch {gid}")
        if e and rec.get("rewards") is not None and rec["rewards"]!=scores:raise ValueError(f"economy score mismatch {gid}")
        if c.get("sha256") and any(h and h!=c["sha256"] for h in (lay.get("sha256") if lay else None,e.get("replay_sha256") if e else None)):
            raise ValueError(f"replay hashes disagree for {gid}")
        replay=f"https://117.72.210.134/replay/{gid.split('-')[0]}/{gid.split('-')[1]}"
        detail=f"economy_games/{gid}.md" if (source/"economy_games"/(gid+".md")).exists() else ""
        missing=[]
        if not c.get("frame_verified"):missing.append("缺逐帧")
        if not v:missing.append("缺独立验证")
        if not lay:missing.append("缺布局")
        if not rec.get("whole"):missing.append("缺经济渠道")
        if not rec.get("swing"):missing.append("缺转折")
        if any((gid,s) not in defense for s in (0,1)):missing.append("缺防守证据")
        if not detail:missing.append("缺逐局经济页")
        row={"id":gid,"cohort":c["cohort"],"seed":c.get("seed"),"created_unix":c["card"].get("created"),
             "replay_url":replay,"frame_verified":bool(c.get("frame_verified")),"transitions":c.get("transitions"),
             "raw_sha256":c.get("sha256"),"canonical_sha256":c.get("canonical_sha256"),
             "validation":v.get("verdict") if v else None,"validation_terminal":v.get("terminal_kind") if v else None,
             "validation_frames":v.get("frames") if v else None,"validation_engine_steps":v.get("engine_steps_matched") if v else None,
             "validation_issues":safe_json(v.get("issues",[])) if v else None,
             "initial_terrain_u8_sha256":v.get("initial_terrain_u8_sha256") if v else None,
             "initial_observation_map_sha256":v.get("initial_map_sha256") if v else None,
             "economy_method":rec.get("method"),"outcome":outcome,"evidence_status":"；".join(missing) if missing else "齐全",
             "economy_page":detail}
        margin=round(scores[0]-scores[1],2) if outcome!="待核" else None
        for s in (0,1):
            row[f"seat{s}_rank"]=ranks[s]; row[f"seat{s}_name"]=names[s]
            row[f"seat{s}_model_id"]=c["seat_strategy_ids"][s]; row[f"seat{s}_score"]=scores[s]
            row[f"seat{s}_margin"]=margin if s==0 or margin is None else -margin
            row[f"seat{s}_result"]="平" if outcome=="平" else "待核" if outcome=="待核" else "胜" if outcome==f"席{s}胜" else "负"
            totals=rec["whole"]["seats"][s] if rec.get("whole") else {}
            for k in MEASURES:row[f"seat{s}_{k}"]=totals.get(k)
            de=defense.get((gid,s)); dr=de[1] if de else {}
            if de and (dr["player"]!=names[s] or dr["opponent"]!=names[1-s] or
                       (c.get("sha256") and dr["raw_sha256"]!=c["sha256"])):
                raise ValueError(f"defense seat or hash mismatch {gid}/{s}")
            row[f"seat{s}_defense_line"]=de[0] if de else None
            for k in DEFENSE:row[f"seat{s}_defense_{k}"]=dr.get(k)
            stages=lay["snapshots"] if lay else {}
            for t in layout.get("snapshot_steps",[]):
                snap=stages.get(str(t))
                row[f"seat{s}_t{t}_assets"]=safe_json(snap[s] if snap else None)
        swing=rec.get("swing") or {}
        row.update(swing_from=swing.get("a"),swing_to=swing.get("b"),swing_margin0=swing.get("margin0_delta"),
                   permanent_cash_lead_at=rec.get("persistent_cash_lead_t"),
                   income_by_stage=safe_json(rec.get("stages")),last30=safe_json(rec.get("last30")),
                   terminal_assets=safe_json(rec.get("terminal_assets")),
                   permanent_lead_decisive_window=safe_json(rec.get("decisive")),
                   swing_seat_channels=safe_json(swing.get("seats")),swing_receipt_paths=safe_json(swing.get("largest_receipt_paths")))
        poses={v["t"]:v["seats"] for v in swing.get("positions",[])}
        for back in (10,5,3,0):
            t=max(0,swing["a"]-back) if swing else None
            row[f"swing_minus_{back}_actual_t"]=t
            row[f"swing_minus_{back}_positions"]=safe_json(poses.get(t))
        rows.append(row)
        for rank in seen: by_rank[rank].append(row)
        outcomes[outcome]+=1;completeness[row["evidence_status"]]+=1
    master=io.StringIO(newline="")
    writer=csv.DictWriter(master,fieldnames=list(rows[0]),delimiter="\t",lineterminator="\n")
    writer.writeheader();writer.writerows(rows)
    emit(output/"index_games.tsv.gz",gzip.compress(master.getvalue().encode(),mtime=0,compresslevel=6))
    section=output/"index"; pages=[]; nav=[]
    for user in top:
        rank=user["rank"];ordered=sorted(by_rank[rank],key=lambda r:(-(r["created_unix"] or 0),r["id"]))
        page_count=(len(ordered)+PAGE_SIZE-1)//PAGE_SIZE
        links=[f"[{i+1}](rank{rank}-p{i+1:02d}.md)" for i in range(page_count)]
        appearances=Counter(r[f"seat{s}_result"] for r in ordered for s in (0,1) if r[f"seat{s}_rank"]==rank)
        record=f"{appearances['胜']} / {appearances['平']} / {appearances['负']}"
        nav.append(f"| {rank} | {cell(user['name'])} | {len(ordered)} | {record} | "+" · ".join(f"[{i+1}](index/rank{rank}-p{i+1:02d}.md)" for i in range(page_count))+" |")
        for ix in range(page_count):
            sub=ordered[ix*PAGE_SIZE:(ix+1)*PAGE_SIZE]
            target=f"rank{rank}-p{ix+1:02d}.md";pages.append(target)
            lines=[f"# {cell(user['name'])} · 第 {ix+1} 页","",
                   f"按开局时间从近到远。本页 {len(sub)} 场，共 {len(ordered)} 场。比分按物理席 0、1 排列。分差与胜负从该选手所在席计算，转折则始终报告席 0 的现金领先变化。","",
                   "[回到五人总览](../index.md) · "+" · ".join(links),"",
                   "| 对局 | 物理席 0 | 物理席 1 | 该选手席 / 结果 / 分差 | 席 0 / 1 终局分 | 最剧烈十步及席0变化 | 逐局证据 |",
                   "| --- | --- | --- | --- | --- | --- | --- |"]
            for r in sub:
                seats=[s for s in (0,1) if r[f"seat{s}_rank"]==rank]
                own="、".join(f"{s} / {r[f'seat{s}_result']} / {number(r[f'seat{s}_margin'])}" for s in seats)
                gid=r["id"]; token=f"{gid[:8]}-{gid.rsplit('-',1)[1]}"
                label="—" if r["swing_from"] is None else f"{r['swing_from']}–{r['swing_to']} / {number(r['swing_margin0'])}"
                evidence=[]
                if r["economy_page"]:evidence.append(f"[经济与回溯](../{r['economy_page']})")
                dl=[f"[防{s}](../defense_games.tsv?plain=1#L{r[f'seat{s}_defense_line']})" for s in (0,1) if r[f"seat{s}_defense_line"]]
                evidence.extend(dl)
                evidence.append(cell(r["validation"] or "缺验证"))
                if r["evidence_status"]!="齐全":evidence.append(r["evidence_status"])
                lines.append(f"| [{token}]({r['replay_url']}) | {cell(r['seat0_name'])} | {cell(r['seat1_name'])} | {own} | "
                             f"{number(r['seat0_score'])} / {number(r['seat1_score'])} | {label} | "+" · ".join(evidence)+" |")
            emit(section/target,"\n".join(lines)+"\n")
    info={"games":len(cards),"cohorts":dict(Counter(r["cohort"] for r in rows)),"page_size":PAGE_SIZE,
          "top_player_view_rows":sum(map(len,by_rank.values())),"physical_outcomes":dict(outcomes),
          "evidence_status":dict(completeness),"validation":dict(Counter(r["validation"] or "missing" for r in rows)),
          "layout_field_order":layout.get("field_order"),"stage_steps":layout.get("snapshot_steps"),
          "sources_sha256":{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in sorted(files.items())},
          "data_sha256":hashlib.sha256((output/"index_games.tsv.gz").read_bytes()).hexdigest(),"pages":pages}
    emit(output/"index_sources.json",json.dumps(info,ensure_ascii=False,indent=2)+"\n")
    overview=["# 天梯前五逐局索引","",f"共 {len(rows):,} 个独立对局 ID，其中前五互打 {info['cohorts'].get('mutual',0)} 局，对外 {info['cohorts'].get('external',0)} 局。"
              f"五个选手视图合计 {info['top_player_view_rows']:,} 条，因为互打局在两位选手名下都可查到。下载表每个 ID 只有一行，保留两物理席。","",
              "| 排名 | 选手 | 对局数 | 胜 / 平 / 负 | 页码 |","| ---: | --- | ---: | --- | --- |",*nav,"","## 后续补编","",
              "[新增八局](appendix_499886f1/README.md)和[最后十五局](appendix_4bfdabf5/README.md)逐局附相同的原始回放、资金流和出手背景，三批合计 1,347 局。","","## 数据与口径","",
              "[逐局双席 TSV 压缩表](index_games.tsv.gz) · [来源与哈希](index_sources.json) · [逐帧清单](manifest.json)","",
              "打开选手分页，点对局 ID 看天梯回放，点“经济与回溯”看该局收入明细及转折前的布局。“防0/防1”跳到物理席防守统计。缺少证据的局保留在索引中，空值不得当作零。","",
              f"逐局证据覆盖为 "+
              "，".join(f"{cell(k)} {n} 局" for k,n in sorted(completeness.items()))+"。独立验证状态为 "+
              "，".join(f"{cell(k)} {n} 局" for k,n in sorted(info["validation"].items()))+"。","",
              "转折是整局绝对值最大的连续十步席0现金领先变化；短于十步的对局取全段。到达转折起点前 3、5、10 步的资产位置均保存在 TSV。若回溯早于开局则取第0步，保留实际步号。"
              "终局分差按官方奖励计算；灭绝时它与最终盘面现金差可能不同。","",
              "以 seat0_ 和 seat1_ 开头的列始终对应物理席。收入金额按现金账分类，抢到的货物在实际入账前仍属在途；最后三十步与三段区间存为 JSON。布局资产快照的数组顺序为 "
              +"、".join(layout.get("field_order",[]))+"。布局快照只记录对局实际到达的步数。"]
    emit(output/"index.md","\n".join(overview)+"\n")
    print(info["games"], info["cohorts"], info["evidence_status"], len(pages),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=HERE)
    p.add_argument("--output",type=Path,default=HERE)
    p.add_argument("--allow-partial",action="store_true",help="Preview the join with missing published layers.")
    a=p.parse_args();build(a.source,a.output,a.allow_partial)
