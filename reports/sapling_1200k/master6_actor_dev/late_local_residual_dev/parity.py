"""Check the packaged public agent on both seats. Fixtures and learned-package results are separate."""
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import json
from pathlib import Path
import sys
from types import ModuleType

from pydantic_core import from_json

import build
import runtime

HERE = build.HERE
REPLAYS = Path("/workspace/sapling-all-top5-20261119/replays")
PANEL = HERE.parent / "late_advantage_actor_dev/panel.json"
PUBLIC = {"step", "player", "carbon", "players", "trees"}
CHECKPOINTS = {0, 150, 269, 285, *range(286, 299)}
POSITIVE = {"features":list(runtime.FEATURES), "enabled": True,
            "heads":{k:[11.0 if k=="margin" else 0.0] + [0.0] * 7 for k in ("margin","principal","own_delivery")}}


def module(raw, name):
    mod = ModuleType(name)
    exec(compile(raw, name, "exec"), mod.__dict__)
    return mod


def checked(test, text):
    if not test:
        raise AssertionError(text)


def synthetic(seat):
    own = [170, {"base":16}, {"collector-a":[17,70,"COLLECTOR"],"collector-b":[18,10,"COLLECTOR"],
                             "planter":[66,0,"PLANTER"]}, {}]
    foe = [200, {"other-base":209}, {"other":[153,2,"COLLECTOR"]}, {}]
    return {"step":290, "player":seat, "players":[own,foe] if seat==0 else [foe,own],
            "carbon":[38.0] * 225, "trees":{}}


def fixture(pack):
    f = sys.modules["cnet2_feat"]
    result = []
    for seat in (0,1):
        obs = synthetic(seat)
        cfg = dict(pack._LOCAL_DEFAULT_CFG)
        featured = f.Feat(seat)
        x = featured.observe(obs)
        cases = [
            ("worker_collision", {}, {"collector-a":"RIGHT"}, "new_friendly_or_recruit_collision"),
            ("recruit_collision", {"base":"RECCOLLECTOR"}, {"collector-a":"LEFT"}, "new_friendly_or_recruit_collision"),
            ("worker_move", {"collector-b":"DOWN"}, {"collector-a":"UP"}, "accepted"),
            ("explicit_stop", {"collector-a":"UP","collector-b":"DOWN"}, {"collector-a":"NONE"}, "accepted"),
            ("two_orders", {}, {"collector-a":"UP","collector-b":"UP"}, "multiple_edits"),
            ("base_veto", {}, {"base":"RECCOLLECTOR"}, "inactive_or_protected_target"),
            ("planter_veto", {}, {"planter":"UP"}, "inactive_or_protected_target"),
            ("foreign_veto", {}, {"other":"UP"}, "inactive_or_protected_target"),
            ("invalid_move", {}, {"collector-a":"RECCOLLECTOR"}, "invalid_move"),
            ("malformed_proposal", {}, [], "malformed_proposal"),
            ("identical", {}, {"collector-a":"NONE"}, "already_selected"),
            ("empty", {}, {}, "no_proposal"),
            ("unaffordable_recruit", {"base":"RECPLANTER"}, {"collector-a":"UP"}, "unfunded_or_over_capacity_recruit"),
            ("capacity_recruit", {"base":"RECCOLLECTOR"}, {"collector-a":"UP"}, "unfunded_or_over_capacity_recruit"),
        ]
        for name, parent, proposal, expected in cases:
            conf = dict(cfg)
            if name == "unaffordable_recruit":
                conf["recPlanterIncreaseCost"] = 1_000
            if name == "capacity_recruit":
                conf["workerLimit"] = 3
            out, reason = pack._local_edit(obs,conf,parent,proposal,x,featured.aux,featured)
            checked(reason==expected, (seat,name,expected,reason))
            if reason=="accepted":
                who, move = next(iter(proposal.items()))
                checked({u for u in set(out)|set(parent) if out.get(u)!=parent.get(u)}=={who}
                        and out.get(who,"NONE")==move,(seat,name,"wrong executed action"))
            else:
                checked(out is parent,(seat,name,"changed parent on fallback"))
            result.append([seat,name,reason])
        # Real proposer, diagnostic always-positive outputs, controlled physical joint orders.
        positive_parent = {"collector-a":"LEFT", "collector-b":"DOWN"}
        proposed = pack._LOCAL_RUNTIME.residual_action(obs,cfg,positive_parent,POSITIVE)
        changed, reason = pack._local_edit(obs,cfg,positive_parent,proposed,x,featured.aux,featured)
        checked(reason=="accepted" and changed!=positive_parent, ("positive",seat,proposed,reason))
        result.append([seat,"constant_positive_heads",proposed])
    for s,n,expected in ((285,300,False),(286,300,True),(296,300,True),(297,300,False),
                         (298,300,False),(286,288,False),(286,290,True),(306,320,False)):
        checked(pack._local_window(s,n) is expected, ("window",s,n))
    return {"guard_cases":result,"window_cases":8}


def views(frame, seat):
    shared = frame[0]["observation"]
    return {**{k:shared[k] for k in ("step","trees","players")}, "player":seat,
            "carbon": frame[seat]["observation"]["carbon"]}


def histories(pack, parent, positive_mode):
    cards = json.loads(PANEL.read_bytes())
    chosen = []
    for seat in (0,1):
        eligible = [c for c in cards if c["split"]=="held" and c["seat"]==seat and c["turns"]>=299]
        chosen.extend(sorted(eligible,key=lambda c:c["id"])[:2])
    checked(len({c["terrain_sha256"] for c in chosen})==4, "four different observation-only development terrains")
    cfg_default = dict(pack._LOCAL_DEFAULT_CFG)
    calls = []
    original = pack._LOCAL_RUNTIME.residual_action

    def inspected(view, cfg, orders, weights):
        checked(set(view)==PUBLIC and "randomSeed" not in cfg and "full_carbon" not in view, "private value leak")
        answer = original(view,cfg,orders,weights)
        calls.append((view["step"],view["player"]))
        return answer

    pack._LOCAL_RUNTIME.residual_action = inspected
    rows = []
    for card in chosen:
        seat = card["seat"]
        packed = (REPLAYS/(card["id"]+".json.gz")).read_bytes()
        checked(build.sha(packed)==card["sha256"],"replay fingerprint")
        game = from_json(gzip.decompress(packed))
        config = game.get("configuration") or cfg_default
        config = {**config,"randomSeed":333,"probe_only_private":7529}
        group = []
        for t,frame in enumerate(game["steps"][:299]):
            obs = views(frame,seat)
            if t not in CHECKPOINTS:
                parent._FEAT.observe(obs)
                pack._FEAT.observe(obs)
                continue
            expect = parent.agent(obs,config)
            pre = deepcopy(pack._FEAT) if t in (286,296) else None
            acc = deepcopy(pack._COUNTS) if t in (286,296) else None
            input_obs = {**obs,"full_carbon":[9999]*225,"opponent_policy_private":"ignored"}
            before = len(calls)
            output = pack.agent(input_obs,config)
            trace = pack._LOCAL_TRACE
            checked(trace["parent"]==expect,("parent mismatch",card["id"],t))
            checked(len(calls)-before==int(286<=t<=296),("queried outside window",t))
            diff = sorted(u for u in set(expect)|set(output) if expect.get(u)!=output.get(u))
            checked(len(diff)<=1,(t,"changed multiple commands"))
            for u in diff:
                checked(286<=t<=296 and obs["players"][seat][2][u][2]=="COLLECTOR","changed protected command")
            if t in (286,296):
                after,counts = pack._FEAT,pack._COUNTS
                pack._FEAT,pack._COUNTS = pre,acc
                pack._LOCAL_ENABLED = False
                disabled = pack.agent(obs,config)
                reason = pack._LOCAL_TRACE["reason"]
                checked(disabled==expect and reason=="disabled" and len(calls)-before==1,("disabled identity",t))
                pack._FEAT,pack._COUNTS = after,counts
                pack._LOCAL_ENABLED = True
            group.append({"step":t,"reason":trace["reason"],"changed":diff,"parent":expect,"actual":output})
        checked(parent._COUNTS==pack._COUNTS,"different real-parent prefix counts")
        checked(len(pack._NETS)==2,"both parent networks required")
        rows.append({"replay":card["id"],"sha256":card["sha256"],"terrain":card["terrain_sha256"],
                     "seat":seat,"turns":group,"parent_prefix_counts":deepcopy(parent._COUNTS)})
    counts = Counter(turn["reason"] for game in rows for turn in game["turns"])
    if positive_mode:
        checked(counts["accepted"]>0, "positive control never traversed public agent")
    return {"maps":rows,"total_roots":sum(counts.values()),"decision_reasons":dict(counts),
            "disabled_public_agent_comparisons":2*len(rows),"per_seat_maps":[2,2],
            "selected_sapling_slots_on_parent_decode":sum(d["selected_student_in_disagreement"]
                     for row in rows for d in row["parent_prefix_counts"].values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-only",action="store_true")
    args=parser.parse_args()
    if args.fixture_only:
        source, receipt = build.compose((HERE/"runtime.py").read_bytes(),POSITIVE)
        target = HERE/"mechanism_fixture.json.gz"
    else:
        receipt=json.loads((HERE/"package.json").read_bytes())
        source=(HERE/"candidate.py").read_bytes()
        checked(build.sha(source)==receipt["candidate_sha256"],"candidate fingerprint")
        checked(build.sha((HERE/"weights.json").read_bytes())==receipt["weights_sha256"],"weight fingerprint")
        target=HERE/"package_parity.json.gz"
    pack=module(source,"residual_for_parity")
    parent=module(build.PARENT.read_bytes(),"frozen_for_parity")
    result={"source":receipt,"mode":"constant_positive_fixture" if args.fixture_only else "learned_weights",
            "selection":"two smallest replay ids per acting seat among previously exposed held-panel games with all 299 turns",
            "mechanism":fixture(pack),"serving":histories(pack,parent,args.fixture_only)}
    build.atomic(target,gzip.compress((json.dumps(result,ensure_ascii=False,allow_nan=False)+"\n").encode(),mtime=0))
    print(result["mode"],result["serving"]["decision_reasons"],result["serving"]["selected_sapling_slots_on_parent_decode"])


if __name__=="__main__":
    main()
