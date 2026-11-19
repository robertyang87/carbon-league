"""Freeze deployment-supported counterfactual cards without reading any alternative outcome."""
import sys
from collections import Counter

import actions
from core import HERE, ROOT, BASE, OLD, TRAIN_SEEDS, HOLD_SEEDS, KINDS, read, save, sha, value_hash

sys.path.insert(0,str(OLD/"native"))
import run as frozen_v1


def public_roots(row):
    raw=ROOT/row["raw"]
    assert sha(raw)==row["raw_sha256"] and sha(ROOT/row["official"])==row["official_sha256"]
    rec=read(raw)
    assert rec["seed"]==row["seed"] and rec["candidate_seat"]==row["seat"]
    assert rec["initial_terrain_u8_sha256"]==row["map"] and rec["pack_sha256"]==row["pack_sha256"]
    seat=row["seat"];actor,ctx,validate=frozen_v1.fresh_actor()
    env=frozen_v1.live.begin(rec)
    for t,joint in enumerate(rec["actions"]):
        if t>295: break
        obs=frozen_v1.live.public(env,seat)
        if t==0 or t>=255:
            base=actor.agent(obs,rec["game_configuration"])
            assert base==joint[seat], "source root differs from modern frozen parent"
        else:
            actor._FEAT.observe(obs)
        if t>=255 and t not in row["excluded_exact_physical_steps"]:
            choices={}
            for p in actions.proposals(obs,rec["game_configuration"],base):
                if p["kind"] in choices:
                    continue
                out,why=validate(obs,rec["game_configuration"],base,{p["uid"]:p["action"]},*ctx,actor._FEAT)
                if why!="accepted": continue
                assert len({u for u in base.keys()|out.keys() if base.get(u)!=out.get(u)})==1
                band=next(i for i,(lo,hi) in enumerate(actions.BANDS) if lo<=t<=hi)
                wid=p["uid"]; pl=obs["players"][seat]; foes=obs["players"][1-seat][2]
                pos,cargo,_=pl[2][wid]
                threat=min((actions.distance(pos,w[0],15) for w in foes.values() if w[2]=="COLLECTOR" and w[1]<=cargo),default=99)
                state_id=f'{row["seed"]}-{seat}-{t:03d}'
                card={k:row[k] for k in ("split","seed","map","seat","raw","raw_sha256","official","official_sha256")}
                card.update(id=f'{state_id}-{p["kind"]}',state_id=state_id,t=t,band=band,base=base,alt=out,root_enemy=joint[1-seat],
                            edits=[{k:p[k] for k in ("uid","original","action")}],proposal=p,
                            root_own_public_sha256=value_hash(obs),root_public_enemy_sha256=value_hash(frozen_v1.live.public(env,1-seat)),
                            priority_rank=len(choices),start_cargo=cargo,light_enemy_distance=threat,stream=0,diagnostic=False)
                choices[p["kind"]]=card
            yield t, choices
        env.step(joint)


def main():
    plan=read(HERE/"protocol.json")
    assert sha(HERE/"source/inventory.json")==plan["inventory_sha256"]
    for script,file_hash in plan["frozen_method"].items():
        assert sha(HERE/script)==file_hash,(script,file_hash)
    inventory=read(HERE/"source/inventory.json")
    all_cards=[];support=[]
    for row in sorted(inventory["games"],key=lambda r:(r["split"]!="train",r["seed"],r["seat"])):
        split,seed,seat=row["split"],row["seed"],row["seat"]
        roots=list(public_roots(row))
        group={"seed":seed,"seat":seat,"split":split,"roots_with_opportunities":sum(bool(group) for t,group in roots),
               "by_kind":dict(Counter(k for t,g in roots for k in g)),"training_bands":[]}
        if split=="train":
            for b,(lo,hi) in enumerate(actions.BANDS):
                order=(2*TRAIN_SEEDS.index(seed)+seat+b)%4
                prefer=KINDS[order]
                eligible=[card for t,g in roots if lo<=t<=hi for card in g.values()]
                if not eligible:
                    group["training_bands"].append({"band":b,"status":"no_opportunities"});continue
                # The time key is independent of the demonstrated future and each branch outcome.
                def rank(c):
                    return ((KINDS.index(c["proposal"]["kind"])-order)%4,
                            value_hash(["strata_root_v1", row["map"],seat,b,c["t"]]), c["priority_rank"])
                card=min(eligible,key=rank)
                all_cards.append(card)
                if seed==TRAIN_SEEDS[0] and seat==0 and b in (0,2):
                    all_cards.append({**card,"id":card["id"]+"-s1","stream":1,"diagnostic":True,"primary_id":card["id"]})
                group["training_bands"].append({"band":b,"prescribed":prefer,"actual":card["proposal"]["kind"],
                                                "selected_t":card["t"],"opportunities":len(eligible)})
        else:
            all_cards.extend(card for _,g in roots for card in g.values())
        support.append(group)
    prim=[c for c in all_cards if c["split"]=="train" and not c["diagnostic"]]
    checks={"at_least_18":len(prim)>=18,"four_maps":len({c["map"] for c in prim})==4,
            "three_kinds":len({c["proposal"]["kind"] for c in prim})>=3,
            "all_time_bands":all(sum(c["band"]==b for c in prim)>=4 for b in (0,1,2)),
            "distinct_situations":len({c["t"] for c in prim})>=8,
            "two_cargo_buckets":len({min(int(c["start_cargo"]//50),2) for c in prim})>=2,
            "both_threat_buckets":{c["light_enemy_distance"]<=2 for c in prim}=={True,False}}
    save(HERE/"support.json",{"eligible":all(checks.values()),"checks":checks,"panels":support,"train_primary":len(prim),
                            "train_diagnostic":sum(c["diagnostic"] for c in all_cards),"held_legal_options":sum(c["split"]=="held" for c in all_cards)})
    save(HERE/"cards.json",{"cases":all_cards})
    save(HERE/"native_input.json",{"cards":"cards.json","cards_sha256":sha(HERE/"cards.json"),
                                 "weights_file":"weights.json","parent_pack_sha256":inventory["accepted_raw_pack_sha256"],
                                 "max_followup_turns":44,"wall_seconds":2100,"protocol_sha256":sha(HERE/"protocol.json")})
    print("CC",len(prim),len(all_cards),checks,flush=True)


if __name__=="__main__":
    main()
