"""Verify policy disagreement, history compatibility and real Sapling influence at predeclared roots."""
import importlib.util
import sys
from collections import Counter
from pathlib import Path
import numpy as np
from register import HERE,ROOT,ACTOR,PARTNER,read,seal,sha

spec=importlib.util.spec_from_file_location("partner_modules",PARTNER/"mechanism.py")
sys.path.insert(0,str(PARTNER))
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)

def main():
    protocol=read(HERE/"protocol.json");roots=read(HERE/"roots.json")["roots"]
    paths={"it50":ROOT/"data/weights/sapling_1200k/terminal_return_fit_v1.py","it20":PARTNER/"candidate.py"}
    for k,p in paths.items():assert sha(p)==protocol["partners"][k]
    mods={k:old.module(p,"phase_"+k,.25) for k,p in paths.items()}
    # Save the immutable network input from each actual it50 call, enabling a same-prefix partner swap.
    base=mods["it50"]
    original_decode=base._mixed_actions
    def capture(x,aux):
        base.last_network_input=(x,aux)
        return original_decode(x,aux)
    base._mixed_actions=capture
    stripped={k:old.module(p,"phase_without_"+k,0.) for k,p in paths.items()}
    totals=Counter();rows=[]
    for r in roots:
        raw=read(ROOT/r["raw"]);seat=r["seat"];root=r["t"]
        sim=old.FastCarbon(raw["game_configuration"]).reset(seed=r["seed"])
        active=[*mods.values(),*stripped.values()]
        recorded=[]
        for t,joint in enumerate(raw["actions"][:root+1]):
            obs=sim.obs(seat)
            if t==0 or t>=root-2:
                orders=[m.agent(obs,sim.config) for m in active]
                assert orders[0]==joint[seat]
                if t>=root-2:
                    # Independent compatibility test: replace only the resident partner net on the already-warm
                    # it50 object and evaluate its current feature state against the separately warmed it20.
                    a,b=mods["it50"],mods["it20"]
                    old_expert=a._NETS[0]; a._NETS[0]=b._NETS[0]
                    action_change=orders[0]!=orders[1]
                    switched_labels=original_decode(*a.last_network_input)
                    switched=a._FEAT.to_actions(switched_labels)
                    assert switched==orders[1], "same-prefix partner swap disagreed with full it20 agent"
                    # The resident feature tracker contains identical per-seat public state under both packages.
                    for field in ("last","last_step"):
                        assert np.array_equal(getattr(a._FEAT,field),getattr(b._FEAT,field))
                    assert a._FEAT.hist==b._FEAT.hist
                    assert a._FEAT.prev_w==b._FEAT.prev_w and a._FEAT.prev_trees==b._FEAT.prev_trees
                    a._NETS[0]=old_expert
                    recorded.append({"t":t,"commands":dict(zip(("it50","it20","it50_without_sapling","it20_without_sapling"),orders)),
                        "root_partner_different":action_change,"it50_student_different":orders[0]!=orders[2],
                        "it20_student_different":orders[1]!=orders[3]})
            else:
                for m in active:m._FEAT.observe(obs)
            if t!=root:sim.step(*joint)
        row={"source":r["id"],"regime":r["regime"],"split":r["split"],"map":r["map"],"seat":seat,"audit":recorded}
        rows.append(row)
        if any(q["root_partner_different"] for q in recorded):totals[r["split"]+"_maps_with_disagreement"]+=1
        for partner in mods:
            if any(q[partner+"_student_different"] for q in recorded):totals[partner+"_maps_with_sapling_difference"]+=1
    gate=totals["train_maps_with_disagreement"]>=2 and totals["held_maps_with_disagreement"]>=1 and all(totals[arm+"_maps_with_sapling_difference"] for arm in mods)
    seal(HERE/"mechanism.json.gz",{"roots_sha256":sha(HERE/"roots.json"),"protocol_sha256":sha(HERE/"protocol.json"),"source_sha256":sha(HERE/"mechanism.py"),
        "partners":protocol["partners"],"exact_it50_command_matching":True,"same_prefix_resident_it20_swap_matches_real_it20_entry":True,"same_feature_history_including_prior_worker_and_tree_states":True,
        "counts":dict(totals),"gate":"PASS" if gate else "STOP","rows":rows})
    print(dict(totals),gate)
if __name__=="__main__":main()
