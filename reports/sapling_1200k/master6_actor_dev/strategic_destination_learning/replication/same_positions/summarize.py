"""Extract before-the-turn decisions and cash actually realized by named actors from the two tapes."""
import gzip
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def read(path):
    data=path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix==".gz" else data)


def main():
    result=read(HERE/"result.json")
    keys=read(HERE/"key_events.json")
    focused=[]
    for game in keys["games"]:
        s=game["seat"]
        k=game["decision"]
        rows=read(HERE/f"seat{s}.json.gz")
        rows_by_step={r["step"]:r for r in rows}
        e=rows_by_step[k]
        officials=game["focal_workers"]
        records=[]
        for who in officials:
            owner=int(who.split("-")[1])
            pre=e["worker_states"][owner][who]
            chosen=e["official_actions"][owner].get(who,"NONE")
            source=e["orders"]
            record=dict(uid=who,owner=owner,role=pre[2],position=pre[0],cargo=pre[1],actual_action=chosen,
                realized_cash=[a for a in e["actual_ledger"] if len(a)>=4 and a[3]==who],lookback=[])
            if owner==s:
                record["policies_at_pivot"]={n:source[n].get(who,"NONE") for n in source}
            for t in range(k-10,k+1):
                r=rows_by_step[t]
                state=r["worker_states"][owner].get(who)
                if state:
                    a=r["official_actions"][owner].get(who,"NONE")
                    detail=dict(t=t,position=state[0],cargo=state[1],actual_action=a)
                    if owner==s:detail["own_policy_commands"]={n:r["orders"][n].get(who,"NONE") for n in source}
                    record["lookback"].append(detail)
            if owner==s:
                refs={}
                for name,i in (("parent",1),("normalized_zero",2),("literal_zero",3)):
                    refs[name]=[r["step"] for r in rows if r["step"]<=k and any(
                        d["uid"]==who and d["a"][0]!=d["a"][i] for d in r["different"])]
                record["observed_history_decision_disagreements_vs_bc_before_or_at_pivot"]=refs
            records.append(record)
        focused.append(dict(seat=s,step=k,actors=records,pivot_changed_ours=e["different"],
                            actual_margin_before=game["focal_margin_before"],actual_margin_after=game["focal_margin_after"],
                            actual_margin_final=game["official_margin"]))
    report=dict(source_result_sha256=hashlib.sha256((HERE/"result.json").read_bytes()).hexdigest(),
                source_events_sha256=result["key_events_sha256"],foci=focused)
    (HERE/"pivots.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
