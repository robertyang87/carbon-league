"""Apply the frozen rent predictor to both independently collected training examples when defined."""
import gzip
import json
import zipfile

from study import D, HERE, check, dump, freight, read, sha, tree


def run():
    labels=D/"tree_value_labels"
    panel=read(labels/"settlement.json")
    check(sha((labels/"protocol.json").read_bytes())==panel["protocol_sha256"],"Wrong action pair panel")
    wp=HERE/"rent/weights.json"
    check(sha(wp.read_bytes())=="0a9fb5f2bcafae18cf1a25fafc72e9fe9d6b84ac7e75c27155a93e3282ea42ab","Wrong frozen weights")
    weights=read(wp);output=[]
    for game in panel["games"]:
        ticket=game["ticket"]
        t,s=ticket["step"],ticket["seat"]
        row=dict(seed=ticket["seed"],seat=s,step=t,kind=ticket["kind"],cell=ticket["cell"],worker=ticket["uid"],
                 official_plant_or_take_minus_leave=game["delta_stay_minus_leave"])
        output.append(row)
        if ticket["kind"]!="plant":
            row.update(conditional_rent_prediction=None,reason="seizure opportunity on an existing enemy tree outside training target")
            continue
        folder=D/"cold_start/development/hosted"
        package=folder/(ticket["archive"]+".zip")
        check(sha(package.read_bytes())==ticket["archive_sha256"],"Wrong history source")
        with zipfile.ZipFile(package) as ar:
            raw=ar.read("cold_repair/observations.jsonl.gz")
        check(sha(raw)==ticket["public_stream_sha256"],"Changed historical stream")
        frames=gzip.decompress(raw).splitlines()
        def at(i):return json.loads(frames[i])["observations"][s]
        obs=at(t);cfg=freight.econ.config()
        X=tree.features(obs,ticket["uid"],cfg,{k:at(t-k)["players"] for k in (3,5)})
        row["conditional_rent_prediction"]=max(0.,weights["intercept"]+sum(b*(v-m)/z for b,v,m,z in zip(weights["beta"],X,weights["center"],weights["scale"],strict=True)))
        tape=read(labels/f"{ticket['seed']}_stay.json.gz")
        check(len(tape["rows"][0]["tree_on_root_cell"])==1,"Ambiguous focal planting outcome")
        tid=tape["rows"][0]["tree_on_root_cell"][0]["tree"]
        got=[0.,0.];charge=[]
        for k,turn in enumerate(tape["rows"]):
            after=turn["tree_on_root_cell"]
            still=any(q["tree"]==tid and q["seat"]==s for q in after)
            for typ,seat,val,*detail in turn["flows"]:
                if seat!=s:continue
                if k==0 and typ=="plant" and detail==[ticket["uid"],ticket["cell"]]:charge.append(-val)
                if still and typ=="tree_absorption" and detail[0]==tid:got[int(k>=8)]+=val
                if typ=="tree_collision_cash" and len(detail)>2 and detail[2]==tid:got[int(k>=8)]+=val
            if not still:
                break
        check(len(charge)==1,"Focal tree root payment missing")
        row.update(own_focal_first8=got[0],own_focal_tail=got[1],plant_fee_paid=charge[0],
                   actual_first_tenure_net=sum(got)-charge[0],model_zero_slopes_intercept=weights["intercept"],
                   parent_native_search_leaf_residual=None)
    dump("factual_example.json",dict(source_settlement_sha256=sha((labels/"settlement.json").read_bytes()),rows=output,
         interpretation="Two independent physically specified action pairs. Only the new-plant arm has a defined conditional rent prediction. Diagnostic readback only; neither action-pair protocol supplies both native H8 finalist values on a shared search menu."))

if __name__=="__main__":run()
