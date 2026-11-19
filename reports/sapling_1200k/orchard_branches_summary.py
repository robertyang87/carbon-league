"""Summarize every prospectively locked map and separate direct tree income from terminal gain."""
import gzip
import hashlib
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent/"orchard_branches"


def main():
    plan=json.loads((OUT/"plan.json").read_bytes())
    rows=[]
    for card in plan["maps"]:
        tape=OUT/(card["id"]+".json.gz")
        data=tape.read_bytes()
        game=json.loads(gzip.decompress(data))
        if card!=game["card"] or game["pack_sha256"]!=plan["pack"]:
            raise ValueError("map or policy identity mismatch")
        row={"id":card["id"],"seed":card["seed"],"seat":card["seat"],"group":card["group"],
             "eligible_slots":game["eligible_slots"],"sha256":hashlib.sha256(data).hexdigest()}
        if "root" in game:
            seat=card["seat"]
            buy,move=game["buy"],game["move"]
            if any(g["terminal_status"]!=["DONE","DONE"] for g in (buy,move)):
                raise ValueError("unfinished arm")
            difference=lambda k,p: buy["by_seat"][p].get(k,0.)-move["by_seat"][p].get(k,0.)
            tree=lambda p: sum(difference(k,p) for k in ("plant","seize","tree_absorption","tree_collision_cash"))
            row.update(step=game["root"]["step"],prediction=game["root"]["predicted_net"],
                       public_price=game["root"]["public_price"],actual_solo_net=(buy["root_tree"] or {}).get("net"),
                       margin_if_buy=buy["margin"], margin_if_move=move["margin"],
                       buy_minus_move=game["buy_minus_move"], own_delivery_change=difference("own_delivery",seat),
                       enemy_delivery_change=difference("own_delivery",1-seat),
                       both_sides_tree_margin_change=tree(seat)-tree(1-seat),
                       changed_other_friendly_orders=len({k for k in set(buy["actions"][0][seat])|set(move["actions"][0][seat])
                            if k!=game["root"]["unit"] and buy["actions"][0][seat].get(k)!=move["actions"][0][seat].get(k)}))
        rows.append(row)
    summary={}
    for group in ("design","reserved","all"):
        sample=[r for r in rows if "buy_minus_move" in r and (group=="all" or r["group"]==group)]
        high=[r for r in sample if r["prediction"]>50.]
        def summarize(s):
            if not s:return {"n":0}
            return {"n":len(s),"mean_buy_minus_move":sum(r["buy_minus_move"] for r in s)/len(s),
                    "num_buy_better":sum(r["buy_minus_move"]>0 for r in s),
                    "buy_wins":sum(r["margin_if_buy"]>0 for r in s),
                    "move_wins":sum(r["margin_if_move"]>0 for r in s)}
        summary[group]={"used_opportunities":summarize(sample),"historical_high_net_criterion":summarize(high)}
    (OUT/"summary.json").write_text(json.dumps({"summary":summary,"rows":rows},indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
