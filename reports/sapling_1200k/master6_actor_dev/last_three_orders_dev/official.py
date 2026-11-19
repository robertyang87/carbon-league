"""Use the original Carbon interpreter on all enumerated final orders. State is cloned before every alternative."""
import copy
from collections import Counter
import numpy as np
import check as c


def main():
    totals=[];gains=[]
    for row in c.P["games"]:
        r=c.read(c.HERE/"per_game"/f"{row['seed']}_{row['seat']}.json.gz");q=r["roots"][-1]
        if "actions" not in q:continue
        g=c.read(c.ROOT/row["source"]);s=row["seat"];env=c.live.begin(g)
        for joint in g["actions"][:-1]:env.step(joint)
        assert not env.done and env.states[0].observation.step==298
        base_cash=[x[0] for x in env.states[0].observation.players];scores=[]
        for a in q["actions"]:
            branch=copy.copy(env);branch.states=copy.deepcopy(env.states);branch.steps=list(env.steps);branch.logs=list(env.logs);branch.error=dict(env.error) if isinstance(env.error,dict) else list(env.error)
            pair=list(g["actions"][-1]);pair[s]=a["command"];branch.step(pair)
            official=[x.reward for x in branch.states];assert branch.done and all(x.status=="DONE" for x in branch.states)
            assert abs((official[s]-official[1-s])-a["actual_one_step_cash_margin"])<=.00001
            scores.append(official)
        assert not env.done and base_cash==[x[0] for x in env.states[0].observation.players] and list(scores[0])==g["official_score_by_seat"]
        gains.append(q["actual_one_step_cash_gain"])
        totals.append(dict(seed=row["seed"],seat=s,options=len(scores),sha256=row["sha256"],actual_simultaneous_opponent=g["actions"][-1][1-s],
            official_scores_by_option=scores,selected=q["chosen"],selected_final_gain=q["actual_one_step_cash_gain"],hindsight_ceiling=q["oracle_max_one_step_gain"]))
    c.write(c.HERE/"official_final.json.gz",dict(protocol_sha256=c.sha(c.HERE/"protocol.json"),rows=totals))
    print(len(totals),sum(r["options"] for r in totals),max(gains),flush=True)


if __name__=="__main__":main()
