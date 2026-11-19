"""Independently count where the focal starting cargo went, distinguishing worker robbery from all other losses."""
from collections import Counter
import types

def augment(account,snapshot,cfg,orders,seat,focal):
    cert=account.account_arm(snapshot,cfg,orders,seat,focal)
    sim=account.settled.FastCarbon(cfg);sim.set_state(snapshot);start=sim.obs(seat)
    tracks={uid:{"holder":uid,"amount":float(start["players"][seat][2][uid][1]),"relayed":False,"final":Counter(),"events":[]} for uid in focal}
    for j in orders:
        t=sim.step_count;old,funding,_=account.settled.one_step(sim,j,seat,verify=False)
        for uid,z in tracks.items():
            if z["amount"]<=1e-12:continue
            holder=z["holder"];rows,following,remaining=funding[holder];principal=z["amount"]
            assert holder in old and principal<=old[holder]+1e-6
            for cat,why,payer,payee,share in rows:
                value=principal*share
                if cat=="relay":
                    z["relayed"]=True
                else:
                    if cat=="own_direct" and z["relayed"]:cat="own_relay"
                    z["final"][cat]+=value
                z["events"].append({"t":t,"kind":cat,"reason":why,"holder":payer,"recipient":payee,
                                    "amount":value,"via_friendly_relay":z["relayed"]})
            z["amount"]*=remaining;z["holder"]=following
    aggregate={"principal_to_enemy_worker":0.,"principal_to_enemy_worker_at_root":0.,
               "principal_lost_to_enemy":0.,"principal_home_direct":0.,"principal_to_own_treasury":0.,
               "principal_enemy_tree_tax":0.,"principal_at_terminal":0.}
    data={}
    for uid,z in tracks.items():
        base=cert["per_worker"][uid];z["final"]["terminal_inventory"]+=z["amount"]
        for k,v in base["principal_fates"].items():account.settled.equal(z["final"][k],v,"opening capital custody")
        account.settled.equal(sum(z["final"].values()),base["principal"],"all opening capital")
        captures=[ev for ev in z["events"] if ev["kind"]=="lost_to_enemy" and ev["reason"]=="enemy_worker"]
        specific={"principal_to_enemy_worker":sum(ev["amount"] for ev in captures),
                  "principal_to_enemy_worker_at_root":sum(ev["amount"] for ev in captures if ev["t"]==cert["root_t"]),
                  "principal_lost_to_enemy":z["final"]["lost_to_enemy"],"principal_home_direct":base["strict_principal_delivered"],
                  "principal_to_own_treasury":z["final"]["own_direct"]+z["final"]["own_relay"],
                  "principal_enemy_tree_tax":z["final"]["tax"],"principal_at_terminal":z["final"]["terminal_inventory"]}
        for k,v in specific.items():aggregate[k]+=v
        data[uid]={"starting_principal":base["principal"],"events":z["events"],"final_holder":z["holder"],"measures":specific}
    cert["defense"]={"unit_accounts":data,"measures":aggregate,"definition":"All quantities follow opening focal cargo only. enemy_worker means transfer to a surviving enemy worker, including any subsequent loss after a friendly relay. Enemy tree tax and opponent base receipts are separate fates. strict direct deposit is made by the original worker at its own base. A captured worker's cargo is not immediately cash."}
    return cert

def main(h,split):
    inp,cards=h.panel();selected,sel=h.selected(inp,cards,split);receipt=h.read(h.HERE/f"{split}_complete.json")
    h.check(receipt=={"active":[c["id"] for c in selected],"frozen_sha256":h.frozen_hash(),"held_selection_sha256":sel},"wrong stage receipt")
    measure=h.load("native_generic_lot_verification",h.PREVIOUS/"labels.py").measure
    verifier=types.SimpleNamespace(**vars(h.prior))
    original=h.prior.account
    verifier.account=types.SimpleNamespace(settled=original.settled,account_arm=lambda snapshot,cfg,joints,s,focal:augment(original,snapshot,cfg,joints,s,focal))
    rows=[]
    for c in selected:
        d=h.HERE/c["id"];pair=h.read(d/"pair.json");arms={};hashes={}
        h.check(pair["files"]=={a:h.sha(d/f"{a}.json.gz") for a in ("base","alt")} and pair["card_sha256"]==h.dig(c),"altered root or arms")
        for a in ("base","alt"):
            z,hashes[a]=measure(verifier,c,a,d)
            account=h.read(d/f"{a}_account.json.gz")["account"]
            arms[a]={**z,**account["defense"]["measures"]}
        fields=("id","state_id","split","seed","seat","map","t","proposal","band","kind","stratum","source_outcome","start_cargo","stream","diagnostic","primary_id")
        row={k:c.get(k) for k in fields}
        row.update(stream=c.get("stream",0),diagnostic=bool(c.get("diagnostic")),cache=pair["cache"],
                   status="paired_full_original",arms=arms,account_sha256=hashes,
                   delta={k:arms["alt"][k]-arms["base"][k] for k in arms["base"]})
        row["principal_saved_from_enemy_worker"]=-row["delta"]["principal_to_enemy_worker"]
        rows.append(row)
    h.freeze(h.HERE/f"{split}_labels.json",{"split":split,"cases":rows,"frozen_sha256":h.frozen_hash(),"runner_sha256":h.sha(h.HERE/"run.py"),
            "counter_sha256":h.sha(h.prior.OLD/"evidence/account.py"),"verifier_sha256":h.sha(h.HERE/"labels.py"),
            "generic_verifier_sha256":h.sha(h.PREVIOUS/"labels.py"),"source_protocol_sha256":h.sha(h.DEV/"native_input.json"),
            "held_selection_sha256":sel,"direction":"delta is alt minus base for all measures; principal_saved_from_enemy_worker equals base minus alt worker robbery."})
    print("verified full native and principal",split,len(rows),flush=True)
