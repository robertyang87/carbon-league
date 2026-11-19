"""Verify callable physical commands and branch history against both seats' held free-decoder receipts."""
import time

from model import HERE, SOURCES, bridge, load, save, sha, ticks_for, validate
from opponent import OriginalOpponentRollout

def main():
    validate();sources=load(SOURCES)["test"]
    tapes=[r for r in sources if r["seed"]==min(c["seed"] for c in sources)]
    kept=load(HERE/"held_orders.json.gz")["test"]
    ledger={(r["raw_sha256"],r["t"]):r for r in kept};timings=[];hits=0;forks=0
    for card in tapes:
        raw=load(bridge.ROOT/card["path"]);s=card["expert_seat"];agent=OriginalOpponentRollout(s)
        f=bridge.F.Feat(s);ticks=ticks_for([card])[card["raw_sha256"]]
        sim=bridge.FastCarbon(raw["game_configuration"]).reset(seed=raw["seed"])
        for t,joint in enumerate(raw["actions"]):
            obs=sim.obs(s);x=f.observe(obs)
            if t in ticks:
                if t==ticks[0]:
                    clone=agent.fork();fork_action=clone.action(obs);assert clone.next_step==agent.next_step+1
                    forks+=1
                before=time.perf_counter();actual=agent.action(obs);timings.append(time.perf_counter()-before)
                if t==ticks[0]:assert actual==fork_action
                assert set(actual)<=set(obs["players"][s][1])|set(obs["players"][s][2])
                y=f.labels(actual);record=ledger[card["raw_sha256"],t]
                for uid,role,cargo,truth,arms in record["slots"]:
                    slot=f.own_uid.index(uid)
                    assert int(y[slot])==arms["new"][0]
                    hits+=1
            else:agent.warm(obs)
            sim.step(*joint)
    n=len(timings);total=sum(timings)
    save(HERE/"entry.json",dict(compared_turn_commands=n,compared_units_or_bases=hits,independent_branch_checks=forks,
         seeds=[tapes[0]["seed"]],expert_seats=sorted(c["expert_seat"]for c in tapes),total_observe_to_actions_wall_seconds=total,
         commands_per_second=n/total,mean_ms=1000*total/n,weights_sha256=sha(HERE/"model.npz"),
         source_sha256={name:sha(HERE/name) for name in ("opponent.py","check_entry.py","model.py")}))
    print("callable held order checks",n,hits,n/total)
if __name__=="__main__":main()
