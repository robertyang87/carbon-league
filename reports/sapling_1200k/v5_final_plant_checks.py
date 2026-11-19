"""Exercise boundary, safety brakes, seizure preservation and an actual positive trajectory."""
import copy
import gzip
from pathlib import Path
import json
import sys

import numpy as np
from v5_final_plant_audit import ROOT,HERE,FastCarbon,judge
from build_sapling_cash_dagger import module
from eval_v53 import CFG
from cnet2_feat import Feat


def fixture(pack,pol,seat,reason):
    sim=FastCarbon(CFG).reset(seed=620198245)
    st=sim.get_state()
    st.update(step=297 if reason=="outside_final_step" else 298)
    st["cash"]=[80.,80.]
    st["workers"]={f"player-{seat}-worker-0":dict(type="PLANTER",pos=112,carbon=0,player=seat)}
    if reason=="original_cell_contested":
        st["workers"][f"player-{1-seat}-worker-0"]=dict(type="COLLECTOR",pos=113,carbon=0,player=1-seat)
    if reason=="insufficient_confirmed_cash":
        st["cash"][seat]=0
    if reason=="preserve_seizure":
        tid=f"player-{1-seat}-tree-0"
        st["trees"]={tid:dict(pos=112,player=1-seat,age=25)}
        st["trees_dict"]={tid:[None,0]}
    sim.set_state(st)
    obs=sim.obs(seat)
    feature=Feat(seat)
    x=feature.observe(obs)
    # A real V5 re-decode with the fixture's planter held stationary. Subsequent
    # legal decisions still come from the network.
    def select_stay(s,logits,prefix,picked):
        return 0 if s==0 else None
    y,logits,_=pol.act(x,feature.aux,teacher=2,hook=select_stay)
    z,actual=pack._final_choice(pol,feature,x,y,logits,obs,sim.config)
    assert actual==reason and np.array_equal(y,z),(seat,reason,actual)
    return dict(seat=seat,case=reason,unchanged=True)


def main():
    directory=ROOT/"data/weights/sapling_1200k"
    bare=module(directory/"soft_it50_v5_agent.py","bare_reference")
    arm=module(directory/"soft_it50_v5_final_plant_agent.py","candidate_actual_pack")
    assert (arm._W64,arm._QMETA,arm._WMETA)==(bare._W64,bare._QMETA,bare._WMETA)
    track=HERE/"tree_exam_soft_it50_v5_guard/raw_620106812_0.json.gz"
    raw=json.loads(gzip.decompress(track.read_bytes())); seat=raw["candidate_seat"]
    sim=FastCarbon(raw["game_configuration"]).reset(seed=raw["seed"])
    agreed=0
    for t,cmd in enumerate(raw["actions"]):
        obs=sim.obs(seat)
        base=bare.agent(obs,sim.config)
        actual=arm.agent(obs,sim.config)
        if t<298:
            assert base==actual and arm.agent.last_stats["final_plant_reason"]=="outside_final_step"
            agreed+=1
        else:
            assert base==cmd[seat] and actual!=base and arm.agent.last_stats["changed_slots"]==1
            full=sim.full_observation();full["full_carbon"]=full["carbon"]
            old=judge(full,sim.config,cmd)
            changed=list(cmd);changed[seat]=actual
            new=judge(full,sim.config,changed)
            assert old==raw["official_score_by_seat"] and round(new[seat]-old[seat],2)==29.65 and new[1-seat]==old[1-seat]
        sim.step(*cmd)
    checks=[fixture(arm,arm._POLICY,s,kind) for s in (0,1) for kind in
            ("outside_final_step","preserve_seizure","insufficient_confirmed_cash","original_cell_contested")]
    result=dict(identical_encoded_v5_weights_and_metadata=True,unmodified_calls_before_final=agreed,
                positive_root="tree_exam_soft_it50_v5_guard/raw_620106812_0.json.gz @298",
                exact_scores_bare=old,exact_scores_with_guard=new,fixtures=checks)
    (HERE/"v5_final_plant_checks.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
