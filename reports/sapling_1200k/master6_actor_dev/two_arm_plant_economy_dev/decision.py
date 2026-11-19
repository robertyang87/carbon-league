"""Apply the keys and cutoffs fixed in plan.json without mixing archive sensitivity maps."""
import numpy as np
from plan import HERE,read,save,sha

def main():
    facts=read(HERE/"factual_holdout.json")
    matched=read(HERE/"ladder_eval.json")
    native=read(HERE/"native_eval.json")
    model=sha(HERE/"model.json")
    assert facts["model_sha256"]==matched["identity"]["weights_sha256"]==native["model_sha256"]==model
    native_maps=[r for r in native["per_map"] if r["historical_use"]!="outside"]
    assert len(native_maps)==16 and sum(r["fixed_n"] for r in native_maps)==64
    delta=np.array([r["C_minus_parent"]["delta_mean_margin"] for r in native_maps])
    ci=np.quantile(delta[np.random.default_rng(89115).integers(0,len(delta),(10000,len(delta)))].mean(axis=1),[.025,.975])
    arms={group:all(row["all"][a]<row["all"][b] for a,b in (("death_brier","baseline_brier"),("net_mae","baseline_mae")))
          for group,row in facts["by_split"].items()}
    c=matched["splits"]["new_opponent"]["map_equal_survival_difference_vs_A"]
    positive=int((delta>0).sum())
    gate={**arms,"unknown_opponent_survival_at_least_A":c>=0,"native_positive_overall":bool(delta.mean()>0),
          "at_least_half_physical_native_maps_strictly_positive":positive*2>=len(delta)}
    assert np.isclose(delta.mean(),native["original_64_only"]["C_minus_parent"]["delta_mean_margin"])
    record={"model_sha256":model,"plan_sha256":sha(HERE/"plan.json"),"factual_sha256":sha(HERE/"factual_holdout.json"),
        "ladder_sha256":sha(HERE/"ladder_eval.json"),"terminal_sha256":sha(HERE/"native_eval.json"),
        "replayed_maps":len(delta),"strictly_positive":positive,"strictly_negative":int((delta<0).sum()),
        "flat":int((delta==0).sum()),"mean_official_margin_delta":float(delta.mean()),
        "map_resample95_descriptive":ci.tolist(),"conditions":gate,
        "disposition":"eligible_for_consideration_only" if all(gate.values()) else "STOP_before_new_original_games",
        "whole_games_started":0,"formal_games":0}
    save(HERE/"decision.json",record)
    print(record["disposition"],record["strictly_positive"],record["mean_official_margin_delta"])
if __name__=="__main__":main()
