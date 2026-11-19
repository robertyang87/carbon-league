"""Verify the registered multi-step endpoints, all three decoder weights, and the admission decision."""
from statistics import mean
from model import BASELINES, CONTRACT, HERE, load, save, sha

def main():
    held=load(HERE/"held.json")
    closed=load(HERE/"closed_summary.json")
    trace=load(HERE/"closed.json.gz")
    assert closed["complete"] and (closed["tapes"],closed["roots"],closed["branches"],closed["archived_transitions_verified"])==(20,40,120,5980)
    assert sha(HERE/"closed.json.gz")==closed["closed_sha256"]
    assert sha(CONTRACT)==held["protocol_sha256"]
    assert held["weight_sha256"]=={k:sha(p)for k,p in {**BASELINES,"new":HERE/"model.npz"}.items()}
    rows=trace["roots"]
    n=0;checks={}
    for split in ("test","exact"):
        chosen=[r for r in rows if r["split"]==split]
        for horizon in (5,10):
            h=str(horizon)
            for arm in ("new","published","calibrated"):
                for metric in ("expert_normal_deposit","expert_tree_count"):
                    bymap={}
                    for r in chosen:
                        branch=r["arms"][arm]
                        end=branch["horizons"][h]
                        assert end["effective_transitions"]==horizon
                        bymap.setdefault(r["seed"],[]).append(abs(end["error"][metric]))
                    expected=mean(mean(g)for g in bymap.values())
                    actual=closed["evaluation"][split]["both"]["arm"][arm][h]["map_equal_mae"][metric]
                    assert abs(expected-actual)<1e-8
                    n+=1
                if split=="exact":
                    z=closed["evaluation"][split]["both"]["arm"]
                    checks[h]={m:(z["new"][h]["map_equal_mae"][m],z["published"][h]["map_equal_mae"][m])for m in
                               ("expert_normal_deposit","expert_tree_count")}
    criteria={"H5_expert_normal_deposit":checks["5"]["expert_normal_deposit"][0]<=checks["5"]["expert_normal_deposit"][1],
              "H5_expert_tree_count":checks["5"]["expert_tree_count"][0]<=checks["5"]["expert_tree_count"][1],
              "H10_expert_normal_deposit":checks["10"]["expert_normal_deposit"][0]<checks["10"]["expert_normal_deposit"][1],
              "H10_expert_tree_count":checks["10"]["expert_tree_count"][0]<checks["10"]["expert_tree_count"][1],
              "free_diagnostic_action_nll":held["diagnostic"]["new"]["map_equal"]["mean_nll"]<
                                           held["diagnostic"]["published"]["map_equal"]["mean_nll"]}
    assert all(v for k,v in criteria.items() if k!="free_diagnostic_action_nll")==closed["closed_gate"]["passed_closed_conditions"]
    paths=["protocol.json","sources.json.gz","training_roots.json.gz","fit.json","model.npz","entry.json","held.json","held_orders.json.gz","closed.json.gz","closed_summary.json"]
    save(HERE/"verdict.json",{"protocol_conditions":criteria,"passed":all(criteria.values()),
                              "checks_from_branch_error_records":n,"normal_form":"Use archived outcomes only as the reference. No original-agent game or ladder deployment.",
                              "file_sha256":{k:sha(HERE/k)for k in paths}})
    print(criteria,all(criteria.values()))
if __name__=="__main__":main()
