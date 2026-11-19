"""Export every shadow-teacher opportunity from the two fitting maps, with all rejections retained."""
from collections import Counter
from pathlib import Path
import sys

from run import HERE, PLAN, START, END, R, archive


def main():
    samples=[]
    groups=[]
    for case in PLAN["cases"]:
        raw=archive(case)
        s=case["seat"]
        for arm in ("parent","teacher"):
            archive_path=HERE/f"{case['seed']}_{s}_{arm}.json.gz"
            logged=R.load(archive_path)
            sim=R.FastCarbon(raw["game_configuration"]).reset(seed=case["seed"])
            for joint in raw["actions"][:START]:
                sim.step(*joint)
            count=Counter()
            for i,decision in enumerate(logged["decisions"]):
                tape=logged["tape"][i]
                view=sim.obs(s)
                assert sim.step_count==decision["t"] and tape[s]==decision["actual"]
                states=view["players"][s][2]
                for uid,(pos,cargo,kind) in states.items():
                    if kind!="PLANTER":
                        continue
                    parent=decision["parent"].get(uid,"NONE")
                    teacher=decision["teacher_planter"][uid]
                    eligible=decision["eligible"].get(uid,"NONE")
                    actual=tape[s].get(uid,"NONE")
                    count["opportunities"]+=1
                    count["teacher_disagreed"]+=teacher!=parent
                    count["eligible_disagreed"]+=eligible!=parent
                    count["actual_changed"]+=actual!=parent
                    count["actual_direction_to_direction"]+=actual!=parent and actual!="NONE" and parent!="NONE"
                    immediate_payments=[list(e) for e in logged["cash"]["steps"][i]["events"]
                                        if e[0] in ("plant","seize") and e[3]==uid]
                    count["actual_changed_with_payment"]+=bool(actual!=parent and immediate_payments)
                    samples.append(dict(seed=case["seed"],terrain=case["terrain"],seat=s,arm=arm,t=decision["t"],uid=uid,pos=pos,
                        public=view,reference=parent,teacher=teacher,accepted=eligible,executed=actual,
                        team_fallback=decision["rejected"],physical_orders=tape[s],poststep_payments=immediate_payments))
                sim.step(*tape)
            assert len(logged["decisions"])==END-START
            groups.append(dict(seed=case["seed"],seat=s,arm=arm,trajectory_sha256=R.sha(archive_path),counts=dict(count)))
    R.write(HERE/"planter_labels.json.gz",dict(method=R.sha(Path(__file__)),scope="Fitting maps only. Use public observations and preceding recorded own-seat history as inputs. Teacher labels include unchanged routes; realized payments are outcomes only. No per-slot alternative value labels.",groups=groups,rows=samples))
    print(groups)


if __name__=="__main__":
    main()
