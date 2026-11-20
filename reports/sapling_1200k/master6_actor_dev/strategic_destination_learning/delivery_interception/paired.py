"""Inspect every physical training-map intersection in the signed intervention registries."""
from analyze import HERE, IN, INVENTORY, NATIVE, ROOT, FastCarbon, compact, digest, gzip, json, read, select, sha, write
from menu import board, threats, menu_for


def norm(commands):
    return {k:v for k,v in commands.items() if v and v!="NONE"}


def audit(obs, options):
    seat = obs["player"]
    changes = sorted(k for k in set(options[0])|set(options[1]) if options[0].get(k,"NONE")!=options[1].get(k,"NONE"))
    workers = obs["players"][seat][2]
    label = [workers[u][2] if u in workers else "BASE" for u in changes]
    pairs = [norm(c) for c in options]
    possible = []
    selected = []
    for index,a in enumerate(pairs):
        chosen = select(obs,a)
        if chosen and norm(chosen["options"][1])==pairs[1-index]:
            selected.append(chosen["target_uid"])
        if len(changes)!=1 or label!=["COLLECTOR"]:
            continue
        for target in threats(obs):
            # Fix the changed worker; the static pair must have the same legal intercept and avoidance support.
            off = menu_for(obs,a,target)
            if off and off["focal_uid"]==changes[0] and norm(off["options"][1])==pairs[1-index]:
                possible.append(target["target_uid"])
    return dict(changed_uids=changes,roles=label,visible_qualifying_cargo=len(threats(obs)),
                alternate_is_frozen_menu_choice=sorted(set(selected)),matched_selected_offer_by_target=sorted(set(possible)))


def main():
    terrains={g["terrain"] for g in read(INVENTORY)["rows"] if g["original_role"]=="registered_native_training"}
    table=NATIVE/"terminal_direction/commands.json.gz"
    additional=NATIVE/"return_action_labels/labels.json"
    other=read(additional)["rows"]
    assert not terrains & {r["terrain"] for r in other}
    rows=read(table); findings=[]
    for r in rows:
        if r["terrain"] not in terrains:continue
        datum=read(ROOT/r["source"]); sim=FastCarbon(datum["game_configuration"]).reset(seed=r["seed"])
        assert sha(bytes(map(int,sim.carbon)))==r["terrain"]
        history=datum["actions"][:r["t"]]
        assert digest(history)==r["prefix_sha256"]
        for orders in history:sim.step(*orders)
        obs=sim.obs(r["seat"]); assert digest({k:v for k,v in obs.items() if k!="remainingOverageTime"})==r["public_sha256"]
        findings.append(dict(source=r["case"],seed=r["seed"],seat=r["seat"],step=r["t"],accepted_by_original_study=True,
                             **audit(obs,r["archived_commands"])))
    for directory,filename,group in ((IN/"live_curriculum/terminal_pairs","settlement.json","roots"),
                                     (IN/"economy_proposals/terminal_intake","verify.json","fixtures")):
        for r in read(directory/filename)[group]:
            assert r["terrain"] in terrains
            card=dict(source=str((directory/filename).relative_to(ROOT)),seed=r["seed"],seat=r["seat"],step=r.get("root_step"),
                      accepted_by_original_study=r["eligible"],source_status=r["status"])
            if r.get("root_step") is not None:
                p=directory/"outputs"/f"{r['seed']}_{r['seat']}"
                prefix=read(p/"prefix.json")
                m=prefix["root_menu"]; sim=FastCarbon().reset(seed=r["seed"])
                assert sha(bytes(map(int,sim.carbon)))==r["terrain"]
                for raw in gzip.open(p/"prefix.jsonl.gz","rb"):sim.step(*json.loads(raw)["orders"])
                assert sim.step_count==r["root_step"]
                card.update(audit(sim.obs(r["seat"]),m["options"]))
            findings.append(card)
    valid=[x for x in findings if x["accepted_by_original_study"] and x["alternate_is_frozen_menu_choice"]]
    out=dict(physical_train_maps=len(terrains),historical_signed_rows=len(rows),additional_signed_rows_on_disjoint_maps=len(other),
             eligible_two_terminal_labels_under_frozen_menu=len(valid),rows=findings,scope="Physical-map test across signed one-step terminal interventions and recent live common-root trials; full-game scores or one-step immediate rewards cannot fill unobserved counterfactuals.")
    write(HERE/"signed_support.json",out);print(json.dumps(out))
if __name__=="__main__":main()
