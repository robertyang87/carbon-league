"""Read only signed decisions; stop before a native intervention when the triggering rule misses every robbery."""
from collections import defaultdict
from statistics import mean
from shared import HERE,read,save,sha

def avg(rows,metric):
    by_map=defaultdict(list)
    for r in rows:by_map[r["map"]].append(r["delta"][metric])
    return mean(mean(v) for v in by_map.values()) if by_map else None

def main():
    c=read(HERE/"coverage.json");p=read(HERE/"package.json"); protocol=sha(HERE/"protocol.json")
    assert p["sha256"]==sha(HERE/"candidate.py") and p["actions_sha256"]==sha(HERE/"actions.py")
    native=HERE/"native"
    result={"protocol_sha256":protocol,"candidate_sha256":p["sha256"],"coverage_sha256":sha(HERE/"coverage.json")}
    if not c["native_diagnostic_possible"]:
        assert not list(native.glob("*/pair.json")) and not (native/"train_labels.json").exists()
        save(HERE/"verdict.json",dict(result,decision="HOLD_NO_VERIFIED_ROBBERY_COVERED",training_root_games=0,held_root_games=0,
                                    whole_games=0,formal_games=0,original_native_active_seconds=0,
                                    measured_official_gain=None))
        print("HOLD_NO_VERIFIED_ROBBERY_COVERED")
        return
    train=read(native/"train_labels.json");rows=train["cases"]
    primary=[r for r in rows if not r["diagnostic"]]
    rescue=[r for r in primary if r["stratum"]=="rescue"];controls=[r for r in primary if r["stratum"]=="control"]
    repeated=[r for r in rows if r["stratum"]=="rescue" and r["diagnostic"]]
    gates={"adequate_roots":c["train_gate_minimum_reachable"],"protected_principal":(avg(rescue,"principal_to_enemy_worker") or 0)<0,
           "rescued_principal_banked":(avg(rescue,"principal_home_direct") or 0)>0,
           "rescue_margin":(avg(rescue,"margin") or 0)>0,
           "controls_safe":bool(controls) and avg(controls,"margin")>=-10 and avg(controls,"own_delivery")>=-10,
           "repeat_robbery_reduced":len(repeated)==1 and repeated[0]["delta"]["principal_to_enemy_worker"]<0}
    train_ok=all(gates.values())
    save(HERE/"train_gate.json",{"checks":gates,"pass":train_ok,"labels_sha256":sha(native/"train_labels.json")})
    if not train_ok:
        save(HERE/"verdict.json",dict(result,decision="HOLD_TRAIN",checks=gates,whole_games=0,formal_games=0))
        return
    panel=c["all_held_seats"]
    if any(r["excluded_first"] for r in panel):
        save(HERE/"verdict.json",dict(result,decision="HOLD_HELD_FIRST_ALREADY_USED",whole_games=0,formal_games=0))
        return
    selected=[r["first_changed_card"] for r in panel if r["first_changed_card"]]
    save(HERE/"held_selection.json",{"candidate_sha256":p["sha256"],"cards_sha256":sha(HERE/"cards.json"),
                                      "all_registered":panel,"active_ids":selected})
    if not (native/"held_labels.json").exists():return
    actual=read(native/"held_labels.json")["cases"];by_id={r["id"]:r for r in actual}
    empty={k:0. for k in ("margin","principal_to_enemy_worker","principal_home_direct","own_delivery")}
    full=[by_id[r["first_changed_card"]] if r["first_changed_card"] else {**r,"delta":empty} for r in panel]
    assert len(full)==8 and set(by_id)==set(selected)
    hg={"two_maps":len({r["map"] for r in actual})>=2,"both_seats_changed":{r["seat"] for r in actual}=={0,1},
        "margin_up":avg(full,"margin")>0,"neither_seat_worse":min(avg([r for r in full if r["seat"]==seat],"margin") for seat in (0,1))>=0,
        "robbery_down":avg(full,"principal_to_enemy_worker")<0,"principal_deposit_up":avg(full,"principal_home_direct")>0,
        "team_deposit_safe":avg(full,"own_delivery")>=0}
    save(HERE/"verdict.json",dict(result,decision="HELD_ROOT_PASS" if all(hg.values()) else "HOLD_HELD",checks=hg,whole_games=0,formal_games=0))

if __name__=="__main__": main()
