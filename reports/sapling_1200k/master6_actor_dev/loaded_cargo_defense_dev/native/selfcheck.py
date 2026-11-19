"""Exercise direct enemy-worker attribution and cached-pair handling on one already signed game."""
from pathlib import Path
import tempfile
import run,labels

def main():
    idx=run.archived()
    usable=run.read(run.PREVIOUS.parent/"cards.json")["cases"]
    card=next(c for c in usable if c["id"]=="621476120-0-255-mine_hold")
    tape=run.prior.source(card)
    sim=run.prior.account.settled.FastCarbon(tape["game_configuration"]).reset(seed=card["seed"])
    for j in tape["actions"][:card["t"]]:sim.step(*j)
    match=idx[run.signature(card["raw_sha256"],card["seat"],card["t"],card["base"],card["alt"],card["root_enemy"])][0]
    results={}
    with tempfile.TemporaryDirectory(dir=run.HERE) as tmp:
        old_hash=run.frozen_hash;run.frozen_hash=lambda:"offline-archive-fixture"
        try:
            for a in ("base","alt"):
                run.materialize(card,a,Path(tmp),match);z=run.read(Path(tmp)/f"{a}.json.gz")
                assert z["new_original_calls"]==0 and z["completed_original_calls"]==43
                corr=labels.augment(run.prior.account,sim.get_state(),tape["game_configuration"],[z["joint0"]]+[r["joint"] for r in z["history"]],
                                    card["seat"],[card["proposal"]["uid"]])
                assert corr["official_scores_physical"]==z["official_scores"]
                results[a]={"official":z["official_scores"],"amounts":corr["defense"]["measures"],"new_original_calls":0}
        finally:run.frozen_hash=old_hash
    assert results["base"]["amounts"]["principal_home_direct"]==32.0298 and results["base"]["amounts"]["principal_to_enemy_worker_at_root"]==0
    assert results["alt"]["amounts"]["principal_home_direct"]==0 and results["alt"]["amounts"]["principal_to_enemy_worker_at_root"]==32.0298
    run.freeze(run.HERE/"offline_checks.json",{"test":"previous independent positive and negative actual cargo transfer, cache replay using no new searches",
        "card":card["id"],"raw_sha256":card["raw_sha256"],"old_receipt":match,"cases":results,
        "source_sha256":{n:run.sha(run.HERE/n) for n in ("run.py","labels.py","selfcheck.py")}})
if __name__=="__main__":main()
