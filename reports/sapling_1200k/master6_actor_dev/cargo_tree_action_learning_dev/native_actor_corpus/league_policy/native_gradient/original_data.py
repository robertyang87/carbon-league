"""Build matched historical original-opponent policy-gradient roots from complete signed action tapes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
LEAGUE = HERE.parent
CORPUS = LEAGUE.parent
FIT = CORPUS.parent
HISTORY = FIT / "onpolicy_native"
sys.path.insert(0, str(LEAGUE))
import collect
import optimize as opt
sys.path.insert(0, str(CORPUS/"expansion"))
import clock_audit
import numpy as np

PROTOCOL = "7f25ea66b27db3d70264d2b6f728e3a88f931c4ca300c7bb3ebb80f3d5ab41e9"
SCORES = "0487c9467bffef920fab883c86ff704d7df88b2cd046f20805c0a5cca543ba2e"
ORIGINAL = "65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"
SAMPLER = "2df318842cddfaf830cf69e5b630e9f19fd1e564926be82702af415d26df05e8"
STEPS = (12,48,84,120,156,192,228,264)
SEEDS = (621758345,621758346,621758348,621758349)


def bound_game(path, receipt_sha, tape_sha, seed, seat, terrain, scores, policy_sha):
    raw = opt.read(path)
    receipt_path = path.with_name(f"receipt_{seed}_{seat}.json")
    receipt = opt.read(receipt_path)
    opt.require(opt.sha(path)==tape_sha==receipt["game"]["raw_sha256"] and
                opt.sha(receipt_path)==receipt_sha and receipt["binding"]["candidate"]==policy_sha
                and receipt["binding"]["original"]==ORIGINAL, "Historical actor or tape identity differs")
    opt.require((raw["seed"],raw["candidate_seat"],raw["initial_terrain_u8_sha256"],raw["official_score_by_seat"],
                 raw["pack_sha256"]) == (seed,seat,terrain,scores,policy_sha), "Archived historical game mismatch")
    clock=clock_audit.clock(raw,path,opt.ROOT)
    opt.require(not clock["issues"] and clock["configuration"]["natural_t1_058"] and
                clock["search"]["every_action_logged"] and
                clock["search"]["call_count"]==len(raw["actions"])==raw["steps"] and
                not raw["errors"] and not raw["audit_messages"], "Native search clock or archive incomplete")
    result=dict(seed=seed,seat=seat,terrain=terrain,scores=scores,tape_sha256=tape_sha,
                receipt_sha256=receipt_sha,opponent_sha256=ORIGINAL,actor_sha256=policy_sha,
                original_clock=dict(configuration=clock["configuration"],evidence=clock["evidence"],
                call_count=clock["search"]["call_count"],return_counts=clock["search"]["return_counts"],
                completed_depth=clock["search"]["searched_completed_depth_histogram"],
                public_wall=clock["search"]["all_public_wall_seconds"],public_cpu=clock["search"]["all_public_cpu_seconds"]))
    return raw,result


def replay(raw, model=None, teacher=None, event_refs=None, advantage=0.):
    seed,seat=raw["seed"],raw["candidate_seat"]
    env=collect.begin(seed,raw["game_configuration"])
    terrain=hashlib.sha256(bytes(map(int,env.states[0].observation.full_carbon))).hexdigest()
    opt.require(terrain==raw["initial_terrain_u8_sha256"], "Official historical map differs")
    events={}
    recheck=None
    if event_refs is not None:
        opt.require([x["t"] for x in event_refs]==[t for t in STEPS if t<raw["steps"]],"Scheduled opportunity missing")
        for row in event_refs:
            t=row["t"]; path=HISTORY / "traces" / f"{seed}_{seat}" / f"{t:03d}.json"
            event=opt.read(path)
            opt.require(opt.sha(path)==row["sha256"] and event["supported"]==row["supported"] and
                        (event["step"],event["seat"])==(t,seat),"Historical behavior trace identity differs")
            events[t]=event
        os.environ["SAPLING_NATIVE_STREAM"]=str(seed)
        os.environ.pop("SAPLING_NATIVE_TRACE", None)
        feat=sys.modules["cnet2_feat"].Feat(seat)
        recheck=collect.Player(HISTORY/"training_policy.py")
        opt.require(tuple(recheck.mod._SAMPLE_STEPS)==STEPS,"Incorrect eight-opportunity historical behavior")
    captured=[];diagnostics=[]
    for t, cmds in enumerate(raw["actions"]):
        opt.require(not env.done,"Historical tape continues after official terminal")
        if recheck is not None:
            obs=collect.public(env,seat)
            x=feat.observe(obs)
            if t==0 or t in events:
                issued=recheck.act(obs,raw["game_configuration"])
                opt.require(issued==cmds[seat],f"Archive diverges from conditional public policy at {seed}/{seat}/{t}")
            else:
                with collect.attached(recheck.modules):
                    recheck.mod._FEAT.observe(obs)
            if t in events:
                event=events[t]
                observed=recheck.mod.last_training_event
                for name in ("issued","supported","choice","actions","slot","uid","prefix","route","options","legal"):
                    opt.require(observed.get(name)==event.get(name),f"Original menu mismatch: {name} at {seed}/{seat}/{t}")
                h=hashlib.sha256(collect.canonical_bytes(obs)).hexdigest()
                record=dict(t=t,supported=event["supported"],public_sha256=h)
                if event["supported"]:
                    q_error=float(np.max(np.abs(np.subtract(observed["probabilities"],event["probabilities"]))))
                    hidden_error=float(np.max(np.abs(np.subtract(observed["hidden"],event["hidden"]))))
                    opt.require(q_error <= opt.PROB_TOL and hidden_error<=2e-4 and
                                observed["uniform"]==event["uniform"],"Fresh deployment trace differs from logged sample")
                    meta=dict(terrain=terrain,seed=seed,seat=seat,opponent="original_master3s2",advantage=advantage)
                    captured.append(opt.capture(model,teacher,feat,x,event,cmds[seat],meta))
                    record.update(choice=event["choice"],changed=cmds[seat]!=event["options"][0],
                                  probability_error=q_error,hidden_error=hidden_error)
                diagnostics.append(record)
        env.step(cmds)
    opt.require(env.done and [a.status for a in env.states]==["DONE","DONE"] and
                [a.reward for a in env.states]==raw["official_score_by_seat"] and
                env.states[0].observation.step==len(raw["actions"]), "Historical official terminal differs")
    return captured,diagnostics


def original_roots(*, first_only=False):
    """Return (optimizer-compatible (features,row,public_root) triples, independent receipt)."""
    protocol_path=HISTORY/"protocol.json"; scores_path=HISTORY/"scores.json"
    opt.require(opt.sha(protocol_path)==PROTOCOL and opt.sha(scores_path)==SCORES and
                opt.sha(HISTORY/"training_policy.py")==SAMPLER and opt.sha(opt.PACK)==opt.PARENT_SHA,
                "Historical protocol or actor changed")
    plan=opt.read(protocol_path); schedules=opt.read(scores_path)
    opt.require(tuple(plan["steps"])==STEPS and tuple(plan["seeds"])==SEEDS and
                plan["original"]==ORIGINAL and plan["parent"]==opt.PARENT_SHA and plan["training_pack"]==SAMPLER,
                "Historical schedule, original or candidate differs")
    pair={(s,i) for s in SEEDS for i in (0,1)}
    controls={(c["seed"],c["seat"]):c for c in plan["historical_controls"]}
    sampled={(g["seed"],g["seat"]):g for g in schedules}
    opt.require(len(schedules)==len(plan["historical_controls"])==8 and set(sampled)==set(controls)==pair,
                "Incorrect historical ticket census")
    model=opt.package();teacher=opt.teacher_reader(model)
    roots=[];games=[]
    for seed,seat in sorted(pair)[:1 if first_only else None]:
        c,g=controls[seed,seat],sampled[seed,seat]
        opt.require(c["candidate"]==opt.PARENT_SHA,"Archived control has different frozen policy")
        raw_c,c_receipt=bound_game(opt.ROOT/c["raw"],c["receipt_sha256"],c["sha256"],seed,seat,
                                   c["terrain"],c["scores"],opt.PARENT_SHA)
        replay(raw_c)
        sample_path=HISTORY/"games"/"candidate"/f"raw_{seed}_{seat}.json.gz"
        rpath=sample_path.with_name(f"receipt_{seed}_{seat}.json")
        opt.require(opt.read(HISTORY/f"complete_{seed}_{seat}.json")==g and
                    opt.read(rpath)["binding"]["protocol"]==PROTOCOL,"Completion record differs")
        raw_s,s_receipt=bound_game(sample_path,opt.sha(rpath),g["raw"],seed,seat,c["terrain"],g["score"],SAMPLER)
        _,u_s=opt.payoff(raw_s["official_score_by_seat"],seat)
        _,u_c=opt.payoff(raw_c["official_score_by_seat"],seat)
        rs,checks=replay(raw_s,model,teacher,g["events"],u_s-u_c)
        roots.extend(rs)
        games.append(dict(seed=seed,seat=seat,terrain=c["terrain"],sampled=s_receipt,baseline=c_receipt,
                          sampled_u=u_s,baseline_u=u_c,advantage=u_s-u_c,opportunities=checks,
                          supported=len(rs),alternate=sum(r.get("changed",False) for r in checks)))
    return roots,dict(schema="historical-original-conditional-root-audit/1",protocol_sha256=PROTOCOL,
                      scores_sha256=SCORES,source_sha256=opt.sha(Path(__file__)),sampled_package=SAMPLER,
                      parent_package=opt.PARENT_SHA,games=games,scheduled=sum(len(g["opportunities"]) for g in games),
                      supported=len(roots),alternate=sum(g["alternate"] for g in games),
                      limitation="Eight sampled physical map/seats on four exposed maps. Greedy baselines are independent historical runs of the original adaptive search; their actual opponent draws and available CPU differ. Matched terminal utility is an episode-level advantage; a single selected worker move does not receive a causal return label.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--first-only",action="store_true")
    parser.add_argument("--out",type=Path,required=True); args=parser.parse_args()
    rows,receipt=original_roots(first_only=args.first_only)
    opt.save(args.out,receipt)
    print(len(rows),"supported,",receipt["scheduled"],"scheduled",flush=True)
