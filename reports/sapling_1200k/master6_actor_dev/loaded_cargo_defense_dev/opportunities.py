"""Enumerate a public deployable threat guard on independently registered v1 maps."""
import importlib.util
import sys
from collections import Counter
from shared import HERE,ROOT,ACTOR,PARENT,read,save,sha,dig,source
import actions

OLD=ACTOR/"late_local_residual_dev"
spec=importlib.util.spec_from_file_location("frozen_parent_driver",OLD/"native/run.py")
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)

def one(row):
    tape=source(row);s=row["seat"];cfg=tape["game_configuration"]
    env=old.live.begin(tape);actor,ctx,valid=old.fresh_actor()
    actions.install_parent_trace(actor,PARENT.read_bytes(),valid)
    cards=[]; counters=Counter()
    for t,j in enumerate(tape["actions"]):
        if t>295:break
        obs=actions.public(old.live.public(env,s))
        if t==0 or 180<=t<=295:
            parent=actor.agent(obs,cfg)
            assert parent==j[s],f"source differs from frozen parent at {row['seed']}/{s}/{t}"
        else:
            actor._FEAT.observe(obs)
        if 180<=t<=295:
            choices=actions.candidates(obs,cfg,parent,actor,valid)
            p=next((p for p in choices if p["threat"]["strict"]),None)
            if p is None:
                counters["no_strict_safe_alternative"]+=1
            else:
                alt,why=valid(obs,cfg,parent,{p["uid"]:p["action"]},*ctx,actor._FEAT)
                assert why=="accepted" and alt!=parent
                c={k:row[k] for k in ("split","map","seed","seat","raw","raw_sha256","official","official_sha256")}
                state_id=f'{row["seed"]}-{s}-{t:03d}'
                excluded=t in row["excluded_exact_physical_steps"]
                c.update(state_id=state_id,id=state_id+"-protected",t=t,root_enemy=j[1-s],base=parent,alt=alt,proposal=p,
                         edits=[{k:p[k] for k in ("uid","original","action")}],root_own_public_sha256=dig(obs),
                         start_cargo=p["carried"],excluded=excluded,stream=0,diagnostic=False)
                cards.append(c); counters["previously_exposed" if excluded else "eligible"]+=1
                if row["split"]=="held":break
        env.step(j)
    return {"seed":row["seed"],"seat":s,"map":row["map"],"split":row["split"],"cards":cards,"counts":dict(counters)}

def main():
    plan=read(HERE/"protocol.json")
    assert sha(HERE/"actions.py")==plan["actions_sha256"] and sha(HERE/"source/inventory.json")==plan["source_sha256"]
    inv=read(HERE/"source/inventory.json")
    groups=[]
    for row in inv["games"]:
        if row["split"] not in ("train","held"):continue
        # A held game that never qualifies stays explicitly in the population with a null first dispatch.
        receipt=HERE/"public_sources"/f'{row["seed"]}_{row["seat"]}.json.gz'
        receipt.parent.mkdir(exist_ok=True)
        identity={"protocol_sha256":sha(HERE/"protocol.json"),"reader_sha256":sha(HERE/"opportunities.py")}
        if receipt.exists():
            prior=read(receipt);assert prior["identity"]==identity and prior["raw_sha256"]==row["raw_sha256"]
        else:
            save(receipt,{"identity":identity,"raw_sha256":row["raw_sha256"],"panel":one(row)})
        r=read(receipt)["panel"];groups.append(r)
        print(row["split"],row["seed"],row["seat"],len(r["cards"]),flush=True)
    save(HERE/"all_opportunities.json.gz",{"protocol_sha256":sha(HERE/"protocol.json"),"groups":groups})

if __name__=="__main__":main()
