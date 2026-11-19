"""Capture complete, dated public score indexes for the heavy Top Five and verify new fixtures."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import gzip
import io
import json
from pathlib import Path
import tarfile
from urllib.parse import urlencode
from latest5_inventory_github import (BASE,HEAD,CACHE,DATA,REPORT,ROOT,SERVER,atomic,batch,canonical,dump,rows,
                                      git,identity,key_of,read_url,references,sha,walk_matches)


def paginate(rank):
    sid=rank["id"];name=rank["name"];offset=0;seen=set();meta=[];rows={}
    while True:
        if offset in seen: raise ValueError("repeated offset")
        seen.add(offset)
        url=SERVER+"/api/matches?"+urlencode(dict(compact=1,limit=100,offset=offset,strategy=sid))
        buf,date=read_url(url)
        if buf.startswith(b"\x1f\x8b"):buf=gzip.decompress(buf)
        page=json.loads(buf)
        for r in page["matches"]:
            if r["state"]!="done":continue
            if name not in (r["name_a"],r["name_b"]):raise ValueError("unrelated strategy identity")
            k=key_of(r)
            if k in rows:raise ValueError("overlapping pages "+sid)
            rows[k]=r
        meta.append(dict(offset=offset,count=len(page["matches"]),http_date=date,decoded_sha256=sha(buf)))
        if page["next_offset"] is None:break
        offset=page["next_offset"]
    return dict(id=sid,name=name,pages=meta,cards=rows)


def known_seeds(prior):
    refs={}
    def add(s,tag):
        if isinstance(s,int):refs.setdefault(s,set()).add(tag)
    for path in (REPORT/"top5_h2h_data.json",REPORT/"elite_bc_data.json",REPORT/"cash_dagger_data.json",REPORT/"tree_dagger_data.json",
                 REPORT/"elite_3s2_agreement.json"):
        if not path.exists():continue
        d=json.loads(path.read_bytes())
        for g in d.get("games",[]):
            for k in ("random_seed","seed","map_seed"):add(g.get(k),path.name)
    for path in sorted(REPORT.glob("*/results.json")):
        d=json.loads(path.read_bytes())
        for g in d.get("games",[]):add(g.get("seed"),"recorded development/formal "+path.parent.name)
    for n in range(620113400,620113420):add(n,"cash training")
    for n in range(620124200,620124224):add(n,"tree training")
    for n in range(620176000,620176024):add(n,"soft training")
    sources=json.loads(git(HEAD,"data/sapling_top5_h2h_corpus/sources.json"))["games"]
    split=json.loads(git(HEAD,"data/sapling_top5_h2h_corpus/later7.json"))["games"]
    past_uuids=set()
    for g in sources:
        f=Path("/workspace/sapling-verified108-fromrepo")/g["cohort"]/(g["id"]+".json.gz")
        blob=f.read_bytes() if f.exists() else git(HEAD,g["path"])
        if sha(blob)!=g["packed_sha256"]:raise ValueError("prior replay differs from registered Git copy "+g["id"])
        env=json.loads(gzip.decompress(blob))
        add(env["configuration"]["randomSeed"],"previous mutual "+g["id"])
        past_uuids.add(env["id"])
    for g in sources+split:
        for k in ("seed","random_seed","map_seed"):add(g.get(k),"previous mutual")
    older=json.loads((REPORT/"ladder_cash_rules.json").read_bytes())
    # Previous 2 had detailed records but no explicit map seeds in that report.
    old_names={"410d7b223933838e6ddc195d52beb155-1":"sapling-online-master5.body","f70ac81c272f7310afe225589cfc2f88-1":"sapling-online-master6.body"}
    from recover_public import extract
    for g in older["live"]:
        k=g["game"]; f=Path("/tmp")/old_names[k]
        buf=f.read_bytes() if f.exists() else read_url(SERVER+"/replay/"+k.replace("-","/"))[0]
        env,_,_=extract(buf);add(env['configuration']['randomSeed'],"previous single-side "+k)
        past_uuids.add(env["id"])
    if len(past_uuids)!=len(prior):raise ValueError("previous 110 distinct environment UUIDs required")
    return refs,past_uuids


def arch(games,target):
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode="w",format=tarfile.USTAR_FORMAT) as tar:
        for g in games:
            data=(CACHE/g["path"]).read_bytes()
            if sha(data)!=g["sha256"]:raise ValueError("changed replay cache")
            item=tarfile.TarInfo(g["path"]); item.size=len(data);item.mtime=0;item.mode=0o644
            tar.addfile(item,io.BytesIO(data))
    blob=gzip.compress(buf.getvalue(),compresslevel=1,mtime=0)
    if len(blob)>50_000_000:raise ValueError("archive size")
    atomic(target,blob)
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        if len(tar.getnames())!=len(games):raise ValueError("archive count")
        for g in games:
            if sha(tar.extractfile(g["path"]).read())!=g["sha256"]:raise ValueError("archive digest")
            g["archive"]=str(target.relative_to(ROOT));g["member"]=g["path"]
    return dict(path=str(target.relative_to(ROOT)),sha256=sha(blob),bytes=len(blob),replays=len(games))


def main(as_of):
    names,base,stage,prior=references()
    head=rows(as_of,names)
    raw=(CACHE/"leaderboard.response").read_bytes()
    rd=json.loads(raw)
    ranks=sorted((r for r in rd["strategies"] if r["board"]=="heavy"),key=lambda r:r["board_rank"])[:5]
    if [r["board_rank"] for r in ranks]!=list(range(1,6)):raise ValueError("rank discontinuity")
    frozen=json.loads(git(HEAD,"reports/master_family/tactical_rules/top5_inventory.json"))["ranks"]
    if [(r["id"],r["name"]) for r in frozen]!=[(r["id"],r["name"]) for r in ranks]:
        raise ValueError("lineup changed; map all effective date indices")
    snapfile=REPORT/"latest5_inventory_sources.json.gz"
    if snapfile.exists():
        dump_raw=snapfile.read_bytes(); snapshots=json.loads(gzip.decompress(dump_raw))
        if snapshots["board_sha256"]!=sha(raw):raise ValueError("captured leaderboard changed")
        histories=snapshots["histories"]
    else:
        with ThreadPoolExecutor(max_workers=5) as pool:
            histories=list(pool.map(paginate,ranks))
        snapshots=dict(as_of_board_utc="2026-11-19T02:53:08Z",board_sha256=sha(raw),leaderboard=rd,
                       acquisition_finished_utc=datetime.now(timezone.utc).isoformat(),histories=histories)
        dump_raw=gzip.compress(canonical(snapshots),compresslevel=6,mtime=0)
        atomic(snapfile,dump_raw)
    live={}
    membership={}
    for history in histories:
        for k,row in history["cards"].items():
            if k in live and identity(live[k])!=identity(row):raise ValueError("live index disagreement")
            live[k]=row;membership.setdefault(k,[]).append(history["id"])
    for k in live.keys()&head.keys():
        if identity(live[k])!=identity(head[k]):raise ValueError("github/live score disagreement "+k)
    mutual=lambda d:{k for k,r in d.items() if r["name_a"]!=r["name_b"] and {r["name_a"],r["name_b"]}<=names}
    mh=mutual(head);ml=mutual(live)
    for k in ml:
        if len(membership[k])!=2:raise ValueError("live H2H lacks both membership sources "+k)
    # Prioritize the frozen GitHub new batch. The live tail may add mixed opponents;
    # on indexes truncated to recent pages, do not call historical holes new matches.
    water={n:max(r["created"] for r in head.values() if n in (r["name_a"],r["name_b"])) for n in names}
    extra=set(live)-set(head)-prior
    new_tail={k for k in extra if all(live[k]["created"]>water[n] for n in names&{live[k]["name_a"],live[k]["name_b"]})}
    eligible_ids=(new_tail|((mh|ml)-prior))-set(head)
    original=json.loads((REPORT/"latest5_inventory_github.json").read_bytes())["games"]
    add=batch([(k,live[k],"live_top5_mutual" if k in ml else "live_top5_external") for k in sorted(eligible_ids)])
    new_gh=(mh|(set(head)-set(base)))-prior-{g["id"] for g in original}
    original+=batch([(k,head[k],"github_top5_mutual" if k in mh else "github_top5_external") for k in sorted(new_gh)])
    games=original+add
    seed_refs,old_envs=known_seeds(prior)
    mult=Counter(g["seed"] for g in games);uids=Counter(g["environment_uuid"] for g in games)
    for g in games:
        tags=sorted(seed_refs.get(g["seed"],()))
        if 1_887_000_000<=g["seed"]<1_888_000_000:tags.append("reserved formal map interval")
        if g["environment_uuid"] in old_envs:tags.append("previously used replay environment UUID")
        if 20291101<=g["seed"]<=20292100:tags.append("old fixed exam block")
        g["seed_conflicts"]=tags
        g["eligible"]=not (tags or mult[g["seed"]]>1 or uids[g["environment_uuid"]]>1)
        g["cohort"]="github_top5_mutual" if g["id"] in mh and g["id"] in head else g["cohort"]
    archives=[]
    for prefix in ("github","live"):
        group=[g for g in games if g["cohort"].startswith(prefix) and g["eligible"]]
        if group:archives.append(arch(group,DATA/(prefix+"_replays.tar.gz")))
    out=dict(rankings_observed_utc=snapshots["as_of_board_utc"],rankings_http_url=SERVER+"/api/leaderboard",
             ranked_board="heavy",rankings=[{k:r[k] for k in ("id","name","board_rank","games","rating")} for r in ranks],
             matches_frozen_historical_five=True,source_snapshot="latest5_inventory_sources.json.gz",
             source_snapshot_sha256=sha(dump_raw),github_before=BASE,github_interim=HEAD,github_after=as_of,source_pages_read_utc=sorted({p["http_date"] for h in histories for p in h["pages"]}),
             total_counts=dict(github_before=len(base),github_interim=len(stage),github_after=len(head),github_before_mutual=len(mutual(base)),
                               github_after_mutual=len(mh),live_unique=len(live),live_mutual=len(ml),previous_studied=len(prior),
                               incremental_gh=len(set(head)-set(base)),live_missing_from_github=len(extra),live_newer_than_gh_per_id=len(new_tail),
                               newly_verified_cross=len((mh|ml)-prior),previous_108_missing_from_live=sorted(mh-ml)),
             counts_by_cohort=dict(Counter(g["cohort"] for g in games)),previously_studied_github_additions=sorted((set(head)-set(base))&prior),
             live_missing_from_github_not_newer=sorted(extra-new_tail),not_in_new_training=True,eligible_definition="Clean terminal, correct score and order, distinct map and environment in batch, no recorded known prior training/evaluation seed. Public demonstration data exposed through downstream diagnostic analysis; never an independent final map.",
             known_seed_count=len(seed_refs),prior_distinct_replay_uuids=len(old_envs),archives=archives,games=games)
    dump(REPORT/"latest5_inventory.json",out)
    print(out["total_counts"],out["counts_by_cohort"],flush=True)


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--git-as-of",default="fde97ddac19733594b5d943d8cf1d6970c035b9a")
    main(p.parse_args().git_as_of)
