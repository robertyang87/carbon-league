"""Match historical opportunities when real confiscation occurred versus otherwise similar safe parent orders."""
from collections import Counter
from shared import HERE,PARENT,read,save,sha,dig

def key(c):
    return dig([c["map"],c["seat"],c["proposal"]["uid"],c["t"],"loaded-defense-v1"])

def gap(c,r):
    return (c["seed"],c["seat"])!=(r["seed"],r["seat"]) or abs(c["t"]-r["t"])>=10

def main():
    hist=read(HERE/"historical_train_losses.json.gz")["cases"]
    events={(r["seed"],r["seat"]):r["events"] for r in hist}
    panels=read(HERE/"all_opportunities.json.gz")["groups"]
    positives=[];controls=[];held=[];held_all=[]; coverage=Counter()
    for panel in panels:
        k=panel["seed"],panel["seat"]
        if panel["split"]=="held":
            p=next(iter(panel["cards"]),None)
            if p and not p["excluded"]:held.append({**p,"stratum":"held_first_real_guard"})
            held_all.append({"seed":k[0],"seat":k[1],"map":panel["map"],"first_changed_card":p and p["id"],
                             "excluded_first":bool(p and p["excluded"])})
            continue
        for c in panel["cards"]:
            if c["excluded"]:continue
            wid=c["proposal"]["uid"]
            hit=[e for e in events[k] if e["uid"]==wid and c["t"]<=e["t"]<=c["t"]+3 and e["actual_transferred"]>=80]
            if hit:
                e=min(hit,key=lambda t:t["t"])
                positives.append({**c,"stratum":"rescue","source_outcome":{"t":e["t"],"lead":e["t"]-c["t"],"uid":wid,"amount":e["actual_transferred"],
                                                                         "enemy":e["winner"],"enemy_prior_cargo":e["enemy_pre_collision_cargo"]}})
            else:
                controls.append({**c,"stratum":"control","source_outcome":{"no_direct_capture_of_at_least_80_within_3":True}})
    keep=[];maps=set()
    for c in sorted(positives,key=lambda c:(c["source_outcome"]["lead"],key(c))):
        if c["map"] in maps or any(c["proposal"]["uid"]==p["proposal"]["uid"] and c["raw_sha256"]==p["raw_sha256"] for p in keep):continue
        maps.add(c["map"]);keep.append(c)
        if len(keep)==3:break
    negatives=[];safe_maps=set()
    for rescue in keep:
        opt=[c for c in controls if c["map"] not in safe_maps and all(gap(c,p) for p in keep+negatives)]
        if not opt:break
        def match(c):
            return (0 if c["map"]==rescue["map"] and c["seat"]!=rescue["seat"] else 1,
                    abs(c["start_cargo"]-rescue["start_cargo"])/100+abs(c["t"]-rescue["t"])/60+
                    abs(len(c["proposal"]["threat"]["strict"])-len(rescue["proposal"]["threat"]["strict"])),key(c))
        c=min(opt,key=match);negatives.append(c);safe_maps.add(c["map"])
    cards=keep+negatives
    for cohort in (keep,negatives):
        if cohort:
            c=cohort[0];cards.append({**c,"id":c["id"]+"-s1","diagnostic":True,"primary_id":c["id"],"stream":1})
    cards+=held
    hist_by_root={e["uid"]+"@"+str(r["seed"])+"@"+str(e["t"]):[r["seed"],r["seat"],e["t"],e["actual_transferred"]]
                  for r in hist for e in r["events"]}
    support={"train_map_seats":len(hist),"historical_direct_worker_thefts":hist_by_root,"public_train_options":len(controls)+len(positives),
             "positive_root_opportunities":len(positives),"distinct_train_robberies_covered":len({(c["raw_sha256"],c["source_outcome"]["t"],c["proposal"]["uid"]) for c in positives}),
             "selected_rescue_ids":[c["id"] for c in keep],"selected_control_ids":[c["id"] for c in negatives],"all_held_seats":held_all,
             "native_diagnostic_possible":bool(keep),"train_gate_minimum_reachable":len(keep)>=2 and len(negatives)>=2,
             "source_sha256":sha(HERE/"source/inventory.json"),"public_panel_sha256":sha(HERE/"all_opportunities.json.gz")}
    save(HERE/"coverage.json",support)
    save(HERE/"cards.json",{"cases":cards})
    save(HERE/"native_input.json",{"cards":"cards.json","cards_sha256":sha(HERE/"cards.json"),"parent_pack_sha256":sha(PARENT),
                                 "max_followup_turns":119,"wall_seconds":2100,"protocol_sha256":sha(HERE/"protocol.json")})
    print("confirmed capture/control/all held seat opportunities",len(keep),len(negatives),len(held),"of",len(held_all),flush=True)

if __name__=="__main__":main()
