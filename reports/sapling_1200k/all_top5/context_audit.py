"""Verify every archived decision's input and label alignment without rerunning an opponent or simulator."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import copy
import gzip
import json
from pathlib import Path

import context as api

C=None


def worker_init(index):
    global C
    C=api.Corpus(index=index)


def scan(gid):
    game=C.open(gid);counts=Counter()
    unknown_rules=set(game._data["configuration"])-api.PUBLIC_RULES-{"randomSeed"}
    if unknown_rules:raise ValueError(f"unclassified rule keys: {unknown_rules}")
    platform=game._data.get("info",{}).get("platform_time_control",{})
    counts["games_with_archived_platform_time_control"]+=bool(platform)
    counts["games_with_separate_declared_init_seconds"]+=bool({"init_seconds","initialization_seconds"} & set(platform))
    for t in range(game.turns):
        frame=game._frames[t]
        next_frame=game._frames[t+1]
        true=frame[0]["observation"]["full_carbon"]
        for seat in (0,1):
            obs=game._obs(t,seat)
            if obs["step"]!=t or obs["player"]!=seat or set(obs)!=api.OBS_FIELDS:
                raise ValueError(f"observation scope or ordering differs {gid}/{t}/{seat}")
            if C.locate(C.ordinal(gid,t,seat))!=(gid,t,seat):raise ValueError("broken decision address")
            commands=next_frame[seat]["action"]
            info=next_frame[seat].get("info") or {}
            phase="opening" if t==0 else "subsequent"
            counts[phase+"_post_action_nonempty_info"]+=bool(info)
            counts[phase+"_post_action_time_named_fields"]+=any("time" in str(k).lower() or "duration" in str(k).lower() for k in info)
            if not isinstance(commands,dict) or frame[seat]["status"]!="ACTIVE":
                raise ValueError(f"invalid action boundary {gid}/{t}/{seat}")
            ids=obs["players"][seat][1].keys() | obs["players"][seat][2].keys()
            if set(commands)-ids:raise ValueError(f"t+1 command references an absent actor {gid}/{t}/{seat}")
            mask=api.visibility(obs["players"][seat][2],game.size)
            if len(true)!=game.size**2 or obs["carbon"]!=[v if b else 0 for v,b in zip(true,mask,strict=True)]:
                raise ValueError(f"visibility mismatch {gid}/{t}/{seat}")
            visible=sum(mask);positive=sum(val!=0 for val in obs["carbon"])
            counts["decisions"]+=1;counts["visible_cells"]+=visible;counts["visible_zero_cells"]+=visible-positive
            counts["unknown_cells"]+=len(mask)-visible
            counts["mixed_observed_zero_and_unknown"]+=int(positive<visible<len(mask))
            counts["nonempty_recorded_commands"]+=bool(commands)
    try:game.decision(game.turns,0)
    except IndexError:counts["terminal_root_rejections"]+=1
    else:raise ValueError("terminal frame accepted as a decision")
    return gid,game.turns,counts


def examples(c):
    result={}
    # Pick a real match with a sizeable audited interception. All source coordinates are
    # confirmed against the actual pre-action state and subsequent replay observation.
    with gzip.open(api.HERE/"defense_traces.jsonl.gz","rt") as f:
        for line in f:
            entry=json.loads(line)
            if entry["id"] not in c.rows or entry["cohort"]!="mutual":continue
            for lot in entry["lots"]:
                if (lot["method"]=="collector" and lot["mass"]>=120 and lot["fate"].get("banked",0)>0
                        and entry["side_names"][lot["beneficiary"]]=="claude-AlphaCarbon-Master6"):
                    g=c.open(entry["id"]);t=lot["t"]-1;s=lot["beneficiary"];obs=g._obs(t,s)
                    if lot["victim"] not in obs["players"][1-s][2]:continue
                    result["interception"]={"id":entry["id"],"turn":t,"seat":s,"unit":lot["hunter"],
                       "pre_action_workers":[obs["players"][s][2].get(lot["hunter"]),obs["players"][1-s][2][lot["victim"]]],
                       "outcome_label":{"victim":lot["victim"],"captured_cargo":lot["mass"],"eventual_banked":lot["fate"]["banked"]},
                       "source":"defense_traces.jsonl.gz"}
                    break
            if result:break
    for path in sorted((api.HERE/"economy_data").glob("*.jsonl.gz")):
        with gzip.open(path,"rt") as f:
            for line in f:
                entry=json.loads(line)
                if (entry["id"] not in c.rows or entry["cohort"]!="mutual"
                        or entry["id"]==result["interception"]["id"]):continue
                for s,bank in enumerate(entry["record"]["swing"]["largest_receipt_paths"]):
                    if bank and bank["amount"]>=200 and entry["seat_names"][s]=="claude-AlphaCarbon-Master5s":
                        g=c.open(entry["id"]);t=bank["t"]-1;uid=bank["uid"];worker=g._obs(t,s)["players"][s][2].get(uid)
                        if not worker:continue
                        result["delivery"]={"id":entry["id"],"turn":t,"seat":s,"unit":uid,"pre_action_worker":worker,
                            "outcome_label":{"banked":bank["amount"]},"source":f"economy_games/{entry['id']}.md"}
                        break
                if "delivery" in result:break
        if "delivery" in result:break
    for gid in c.order[:30]:
        if any(gid==v["id"] for v in result.values()):continue
        g=c.open(gid)
        for t in range(g.turns):
            for s in (0,1):
                old=g._obs(t,s)["players"][s];new=g._obs(t+1,s,terminal=True)["players"][s]
                for tid in new[3].keys()-old[3].keys():
                    if tid in g._obs(t,s)["players"][1-s][3]:continue
                    pos,age=new[3][tid]
                    workers=[w for w,a in old[2].items() if a[0]==pos and a[2]=="PLANTER"]
                    if not workers:continue
                    result["plant"]={"id":gid,"turn":t,"seat":s,"unit":workers[0],"pre_action_cash":old[0],
                        "outcome_label":{"new_tree":tid,"pos":pos,"age":age},"source":f"economy_games/{gid}.md"}
                    break
                if "plant" in result:break
            if "plant" in result:break
        if "plant" in result:break
    if len(result)!=3:raise ValueError("expected three witnessed tactics")
    for x in result.values():
        g=c.open(x["id"]);s=x["seat"];t=x["turn"]
        x["ordinal"]=c.ordinal(x["id"],t,s)
        x["recorded_command_after_query"]=g.labels(t,s,["actions"])["joint_commands_from_t"][s].get(x["unit"])
        x["last_observed_entity_states"]=g.entity(t,s,x["unit"])[-3:]
    return result


def leakage_checks(c,gid):
    g=c.open(gid);t=min(30,g.turns-2);seat=1
    args={"history":True,"memory":True}
    expected=g.decision(t,seat,**args);past=g.entity(t,seat,next(iter(g._obs(t,seat)["players"][seat][1])))
    # Change only forbidden knowledge for this seat, from opening through t, plus the entire future.
    d=copy.deepcopy(g._data);d["configuration"]["randomSeed"]=123456;d["id"]="bad-map-id";d["rewards"]=[-9e8,9e8]
    d["info"]["opponent_name"]="poison";d["info"]["platform_time_control"]["future_time_trace"]=[-1]
    for frame in d["steps"][:t+1]:
        frame[0]["observation"]["full_carbon"]=[999]*g.size**2
        frame[0]["observation"]["carbon"]=[998]*g.size**2
        frame[0]["observation"]["remainingOverageTime"]=-2
        frame[0]["action"]={"future":"invalid"}  # enemy submitted commands are private even when historical
    d["steps"][t+1][seat]["action"]={"future":"invalid"}
    d["steps"][t+1][0]["observation"]["players"][0][0]=-9999
    other=api.Game(d)
    if other.decision(t,seat,**args)!=expected or other.entity(t,seat,next(iter(g._obs(t,seat)["players"][seat][1])))!=past:
        raise ValueError("labels, provenance or opposite private record changed decision inputs")
    diagnostic=other.labels(t,seat,api.LABEL_GROUPS)
    if diagnostic["joint_commands_from_t"][seat]!={"future":"invalid"} or diagnostic["final_reward_by_physical_seat"]!=d["rewards"]:
        raise ValueError("explicit label opt-in did not see the mutations")
    for s in (0,1):
        for k in (0,g.turns-1):
            pa=g.decision(k,s,(3,5,10));b=g._frames[k][0]["observation"]
            if pa["at"]["observation"]["players"]!=b["players"] or pa["at"]["observation"]["trees"]!=b["trees"]:
                raise ValueError("shared seat views differ")
            if g.labels(k,s,["actions"])["joint_commands_from_t"]!=[x["action"] for x in g._frames[k+1]]:
                raise ValueError("joint action did not use the next frame")
        if g.decision(0,s)["at"]["own_previous_command"] is not None:raise ValueError("opening own history")
    return {"future_commands_and_cash":True,"outcome_and_seed":True,"opponent_private_grid_time_past_commands":True,
            "hidden_grid":True,"metadata_isolation":True,"both_seats_open_last_action":True,"explicit_label_positive_control":True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index",type=Path,default=api.HERE/"context_index.tsv.gz")
    p.add_argument("--validation",type=Path,default=api.HERE/"validation_games.jsonl.gz")
    p.add_argument("--out",type=Path,default=api.HERE/"context_receipt.json")
    p.add_argument("--jobs",type=int,default=2)
    p.add_argument("--core-examples",action="store_true")
    args=p.parse_args();c=api.Corpus(index=args.index)
    cert={}
    if args.validation.is_file():
        with gzip.open(args.validation,"rt") as f:cert={r["id"]:r for r in map(json.loads,f)}
    if set(cert)!=set(c.order):raise ValueError("independent certificate coverage differs from catalog")
    totals=Counter();cert_count=0
    with ProcessPoolExecutor(args.jobs,initializer=worker_init,initargs=(args.index,)) as pool:
        for i,(gid,n,cnt) in enumerate(pool.map(scan,c.order,chunksize=4),1):
            if gid in cert:
                v=cert[gid]
                if (v["compressed_sha256"]!=c.rows[gid]["replay_sha256"] or v["verdict"]!="consistent" or v["issues"]
                    or v["engine_steps_matched"]!=n or v["joint_action_seats"]!=2*n):
                    raise ValueError(f"independent certificate mismatch {gid}")
                cert_count+=1
            totals.update(cnt)
            if i%200==0:print(i,totals["decisions"],flush=True)
    if totals["decisions"]!=c.size or totals["terminal_root_rejections"]!=len(c.order):
        raise ValueError("incomplete address coverage")
    if c.size and not totals["mixed_observed_zero_and_unknown"]:raise ValueError("visibility mask never distinguished zero")
    ex=examples(c) if args.core_examples else {}
    report={"games":len(c.order),"joint_transitions":c.size//2,"counts":dict(totals),
            "validated_game_certificates_reused":cert_count,"index_sha256":api.digest(args.index.read_bytes()),
            "validation_ledger_sha256":api.digest(args.validation.read_bytes()),
            "source_sha256":{p.name:api.digest(p.read_bytes()) for p in (Path(api.__file__),Path(__file__),
                         api.HERE/"context_catalog.py",api.RULE_FILE,api.RULE_FILE.with_name("carbon.py"),
                         api.RULE_FILE.with_name("helpers.py"),api.RULE_FILE.parents[2]/"core.py")},
            "leakage_checks":leakage_checks(c,c.order[0]),"examples":ex}
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(len(c.order),dict(totals),flush=True)


if __name__=="__main__":main()
