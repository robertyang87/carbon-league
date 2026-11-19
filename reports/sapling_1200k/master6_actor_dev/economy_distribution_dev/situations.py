"""Audit contrasting retrospective alert and conditional-action cases without future actor inputs."""
import gzip
import json
from pathlib import Path
import numpy as np
from score import HERE, FRESH_CACHE, data_build, base, old, read, paired


def state(frame, seat):
    players = frame[0]["observation"]["players"]
    out = []
    for player in (seat,1-seat):
        cash, base_pos, workers, trees = players[player]
        out.append({"seat":player,"cash":cash,"bases":base_pos,"trees":trees,"workers":workers})
    return out


def cases(panel):
    scores = read(HERE / (panel + "_result.json"))
    cards = {c["gid"]: c for c in scores["cards"]}
    z = np.load(HERE / (panel+"_predictions.npz"))
    actual = z["cash"][:,6]
    oldp, newp = z["log_candidate_cash"][:,6], z["dist_candidate_cash"][:,6]
    valid = z["full_horizon"][:,6]
    truth, was, now = actual>=500, oldp>=500, newp>=500
    match = z["action"]
    changed = (z["dist_candidate_action"] != z["dist_control_action"]) & z["legal_choice"]
    chosen = []
    for key, cond, rank in (
        ("largest_old_cash_error",valid,np.abs(oldp-actual)),
        ("reserve_alert_corrected",valid & (was!=truth) & (now==truth),abs(oldp-newp)),
        ("reserve_alert_regression",valid & (was==truth) & (now!=truth),abs(oldp-newp)),
        ("changed_choice_matches_logged_expert",valid & (changed & (z["dist_candidate_action"]==match)).any(1),np.ones(len(valid))),
        ("changed_choice_disagrees_with_logged_expert",valid & (changed & (z["dist_control_action"]==match)).any(1),np.ones(len(valid)))):
        rows=np.flatnonzero(cond)
        if len(rows):
            chosen.append((key, int(rows[np.argmax(rank[rows])])))
    out = []
    for key,i in chosen:
        card = cards[int(z["gid"][i])];t=int(z["step"][i]);seat=card["seat"]
        raw=(old.CACHE/(card["id"]+".json.gz")).read_bytes()
        if base.digest(raw)!=scores["source_replays"][card["id"]]:
            raise ValueError("source trace mismatch")
        game=json.loads(gzip.decompress(raw));frames=game["steps"]
        before = {n:state(frames[t-n],seat) for n in (0,3,5,10) if t>=n}
        recent = [{"t":u,"absolute_seats": [frames[u+1][s]["action"] for s in (0,1)]} for u in range(max(0,t-10),t)]
        rec={"selection":key,"game":card,"step":t,"preaction_public_assets_at_lags":before,
             "last_ten_executed_actions":recent,"archived_realized_rival_freight_next30":float(actual[i]),
             "rival30_cash_estimates":{m:float(z[m+"_cash"][i,6]) for m in ("log_control","log_candidate","dist_control","dist_candidate")}}
        if key.startswith("changed"):
            fe=data_build.F.Feat(seat)
            for u in range(t+1):
                frame=frames[u];common=frame[0]["observation"]
                obs={k:common[k] for k in data_build.PUBLIC}
                obs.update(player=seat,carbon=frame[seat]["observation"]["carbon"])
                fe.observe(obs)
            accepted = z["dist_candidate_action"] if key.endswith("matches_logged_expert") else z["dist_control_action"]
            ss=np.flatnonzero(changed[i] & (accepted[i]==match[i]))
            if not len(ss):
                raise ValueError("selected action is unchanged")
            slot = int(ss[0])
            def label(a):
                singles = np.full_like(match[i],-1)
                singles[slot]=int(a[slot])
                return fe.to_actions(singles).get(fe.own_uid[slot],"NONE")
            rec["conditional_action"]={"slot":slot,"public_unit_id":fe.own_uid[slot],
                    "joint":label(z["dist_candidate_action"][i]),"detach":label(z["dist_control_action"][i]),
                    "recorded":label(match[i]),"own_logged_prefix_labels":match[i,:slot].astype(int).tolist()}
        out.append(rec)
    p=HERE/(panel+"_cases.json.gz")
    p.write_bytes(gzip.compress(json.dumps(out,ensure_ascii=False,allow_nan=False,separators=(",",":")).encode(),mtime=0))
    return out


def fresh_pair():
    scores=read(HERE/"fresh_result.json")
    cards=read(HERE/"fresh_panel.json")["cards"]
    allrows=scores["by_stratum"]["all"]["scores"]
    rows={}
    for label,group in (("unseen_exact_opponent",[c for c in cards if not c["known_opponent"]]),
                        ("every_map",cards)):
        rows[label]={"games":[(v["id"],v["opponent"],v["seat"]) for v in group],
                     "paired_diff_dist_joint_minus_baseline":{}}
        for name in ("log_candidate","dist_control"):
            summaries={}
            for k in ("mae","bias","cost_per_root"):
                vals=[allrows["dist_candidate"]["by_map"][str(c["gid"])][k]-
                      allrows[name]["by_map"][str(c["gid"])][k] for c in group]
                summaries[k]={"per_map":vals,"descriptive_bootstrap":paired(vals)}
            rows[label]["paired_diff_dist_joint_minus_baseline"][name]=summaries
    base.write_json(HERE/"fresh_paired.json",rows)


if __name__=="__main__":
    for p in ("holdout","transfer","fresh"):
        cases(p)
    fresh_pair()
