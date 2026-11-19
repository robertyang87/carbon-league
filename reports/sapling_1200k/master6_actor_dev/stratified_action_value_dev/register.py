"""Record the acceptance rule, disjoint development panels and cost limits before new outcomes."""
from core import HERE, OLD, TRAIN_SEEDS,HOLD_SEEDS, read,sha,save

def main():
    inv=read(HERE/"source/inventory.json")
    groups=inv["games"]
    assert all(len([r for r in groups if r["seed"]==s])==2 for s in TRAIN_SEEDS+HOLD_SEEDS)
    assert {r["seed"] for r in groups if r["split"]=="train"}==set(TRAIN_SEEDS)
    assert {r["seed"] for r in groups if r["split"]=="held"}==set(HOLD_SEEDS)
    assert len({r["map"] for r in groups})==6
    assert not {r["map"] for r in groups}&{r["map"] for r in read(OLD/"protocol.json")["source_groups"]}
    files=["actions.py","cards.py","policy.py","core.py"]
    payload={"schema":1,"train_maps":list(TRAIN_SEEDS),"held_maps":list(HOLD_SEEDS),"inventory_sha256":sha(HERE/"source/inventory.json"),
             "frozen_method":{k:sha(HERE/k) for k in files},"parent_sha256":inv["accepted_raw_pack_sha256"],
             "units":"Physical map is the sampling unit. These six are development maps with existing observed parent-original games, never formal boards. Exclude all previously treated turns identified in source inventory. Historical suffixes are not responses.",
             "training":"24 primary pair opportunities over 4 maps x 2 seats x 3 bands [255,270],[271,285],[286,295], one per stratum. Kind quota cycles 4 kinds over map/seat/band; fixed hash assigns natural turn independent of branch result. At each root highest publicly prioritized prefix-valid proposal per kind only. Also repeat first train map/seat0 bands 0/2 using second independent common RNG; diagnostics excluded from fitting and effective N.",
             "support":"Stop before native computation unless at least 18 primary pairs, four maps, >=4 per band, >=3 action kinds, >=8 different physical turns and coverage of >=2 cargo buckets (0–49.99,50–99.99,>=100), both near-light-enemy <=2 and farther/no-light-enemy.",
             "fit":"Map-equal population; ridge regularization 0.5 all coefficients including intercept; 14 named action/public features plus 4 kind indicators; 4 targets alt-base official margin, focal old principal direct delivery, own and enemy real deposits; clip training targets at +/-400 for fitting, actual raw cash for all verdicts. At least18 4 maps. Same-root duplicate streams never enter fitted weights.",
             "eligibility":"Allow only predicted margin>15, principal>=-5, own delivery>=-5, enemy delivery<=10. At most one edit per game. Each turn from 255 through 295, retain best prefix-valid candidate per kind from frozen public priority; choose eligible maximum predicted margin breaking ties by original deterministic order. Zero outcomes for unchanged seats.",
             "train_gate":"Before exposing held responses do map-level leave-one-out fitting (weights are refit only without that map). Score the ONE actually sampled positive response at each primary train root against fallback zero. At least 3 eligible rows across >=2 maps, all-map mean official margin >0, mean principal >=0, mean own delivery >=0, mean enemy delivery <=+5. Honor support gate and complete 24 expected primary groups when eligible. Failed gate stops run with diagnostic frozen trained weights, no tuning and no held native.",
             "held_gate":"Replay every archived parent action and full public history t255–295 in two other maps, both seats. Study real deployed policy's FIRST natural intervention per seat, retaining 4 allocated seats incl unchanged zero. Advance only if at least two changed maps, all four seat/map mean official margin>0, each seat mean>=0, mean original principal>=0, mean team delivery>0, mean enemy delivery<=0. Verify deployed package selected exactly the same command as prospective card on all held prefixes through first dispatch. No hit can count as success. Four seats are a screen only.",
             "replicates":"On train minimum-numbered map seat0, earliest and latest time band require second common random stream. Report margin/principal/own-delivery sign changes separately; do not select or retune on this outcome and do not inflate N.",
             "cost":"At most 26 train paired original continuations, up to 4 held paired continuations after true train gain. Budget 2100 active seconds for native in total, persistent wall stopwatch; allow up to 4 hours for higher-priority queues including interruptions. Shared original single CPU1 pair lock, release after each pair, verify completed prior attack/defence repeatability. CPU2 nice19 sources and statistical checks. Full pilot conditional with its own registered limits and distinct new board clearance only.",
             "followup":"After held PASS, only the same frozen build may enter two new, identity-cleared, swapped-seat development maps (parent/candidate both versus full original). All map-seat-arm games and terminal official score with development provenance. Formal 100 maps remain reserved."}
    save(HERE/"protocol.json",payload)
    print(sha(HERE/"protocol.json"))

if __name__=="__main__":
    main()
