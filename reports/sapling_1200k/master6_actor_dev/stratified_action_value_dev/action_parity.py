"""Check actual guarded actions, true-prefix validation, causal visibility and opportunity coverage."""
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType

from pydantic_core import from_json

import actions

HERE = Path(__file__).resolve().parent
ACTOR = HERE.parent
ROOT = ACTOR.parents[2]
FROZEN = ACTOR / "late_local_residual_dev"
REPLAYS = Path("/workspace/sapling-all-top5-20261119/replays")
PARENT = ROOT / "data/weights/sapling_1200k/terminal_return_fit_v1.py"
PANEL = ACTOR / "late_advantage_actor_dev/panel.json"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def check(cond, details):
    if not cond:
        raise ValueError(details)


def get_actor():
    parent = ModuleType("stratified_guarded_parent")
    content = PARENT.read_bytes()
    check(sha(content)=="9c54df17eba7cfecf6ac2698eccbbadc2607665e16ab613925188131ce5514e0","parent fingerprint")
    exec(compile(content,str(PARENT),"exec"),parent.__dict__)
    ctx = []
    decode = parent._mixed_actions
    def capture(x, aux):
        ctx[:] = [x,aux]
        return decode(x,aux)
    parent._mixed_actions = capture
    scope = dict(vars(parent))
    validator = (FROZEN/"deploy.py").read_bytes()
    check(sha(validator)=="db27b1832f1b1140484987411afcaff687c0f8b1c112df55c659416aff387ba8","validator fingerprint")
    exec(compile(validator,"deploy.py","exec"),scope)
    cfg = json.loads((ROOT/"code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.json").read_bytes())["configuration"]
    defaults = {k:v["default"] for k,v in cfg.items() if isinstance(v,dict) and "default" in v and k!="randomSeed"}
    defaults.setdefault("episodeSteps",300)
    return parent, ctx, scope["_local_edit"], defaults


def construct(cfg):
    # Known coordinates in the physical frame; both players can occupy the synthetic acting seat.
    results = []
    for seat in (0,1):
        fixtures = [
            ("mine_hold", [32,0,"COLLECTOR"], {"focus":"RIGHT"}, [125,10,"COLLECTOR"], 32),
            ("bank_route", [31,95,"COLLECTOR"], {}, [125,10,"COLLECTOR"], 31),
            ("evade", [31,95,"COLLECTOR"], {"focus":"RIGHT"}, [33,0,"COLLECTOR"], 31),
            ("ore_route", [32,0,"COLLECTOR"], {}, [125,5,"COLLECTOR"], 33),
        ]
        for kind,focus,parent,foe,ore_cell in fixtures:
            own = [250,{"home":18 if kind=="ore_route" else 16},{"focus":focus,"scout":[48,0,"PLANTER"]},{}]
            enemy = [90,{"enemy_home":201},{"enemy":foe},{}]
            public = {"step":280,"player":seat,"players":[own,enemy] if seat==0 else [enemy,own],
                      "trees":{},"carbon":[0.]*225}
            public["carbon"][ore_cell] = 100.
            rows = actions.proposals(public,cfg,parent)
            check(any(p["kind"]==kind for p in rows),(seat,kind,rows))
            for t, expected in ((254,False),(255,True),(270,True),(271,True),(285,True),
                                (286,True),(295,True),(296,False)):
                out = actions.proposals({**public,"step":t},cfg,parent)
                check(bool(out)==expected,(t,kind,seat))
            results.append([seat,kind,[p["action"] for p in rows if p["kind"]==kind]])
    return results


def main():
    parent, ctx, validate, defaults = get_actor()
    controlled = construct(defaults)
    source = json.loads(PANEL.read_bytes())
    cards = []
    for seat in (0,1):
        cards.extend(sorted((c for c in source if c["seat"]==seat and c["split"]=="held" and c["turns"]>=299),
                            key=lambda c:c["id"])[:2])
    check(len({c["terrain_sha256"] for c in cards})==4,"selection")
    detail = []
    kinds = Counter()
    byband = Counter()
    protected, raw, validated, hidden, guards = 0,0,0,0,0
    for card in cards:
        blob = (REPLAYS/(card["id"]+".json.gz")).read_bytes()
        check(sha(blob)==card["sha256"],"archive sha")
        game = from_json(gzip.decompress(blob))
        cfg = {**defaults,**(game.get("configuration") or {})}
        seat = card["seat"]
        roots = []
        for t,frame in enumerate(game["steps"][:297]):
            public = {k:frame[0]["observation"][k] for k in ("step","trees","players")}
            public.update(player=seat,carbon=frame[seat]["observation"]["carbon"])
            if t not in (0,254,296) and not 255<=t<=295:
                parent._FEAT.observe(public)
                continue
            orders = parent.agent(public,cfg)
            before = dict(orders)
            mine = public["players"][seat]
            first = actions.proposals(public,cfg,orders)
            alternatives = actions.validated_proposals(public,cfg,orders,*ctx,parent._FEAT,validate)
            raw += len(first)
            validated += len(alternatives)
            check(before==orders,"mutated parent")
            if t not in range(255,296):
                check(not first and not alternatives,"outside bands")
                protected += 1
                continue
            band = next(i for i,(lo,hi) in enumerate(actions.BANDS) if lo<=t<=hi)
            visible = actions.visible_cells(mine[2],15)
            injected = {**public,"carbon":[v if p in visible else 90000+p
                                           for p,v in enumerate(public["carbon"])],
                        "full_carbon":[-872]*225,"reveal":{"14":9999}}
            check(actions.proposals(injected,cfg,orders)==first,(t,"hidden mineral or diagnostic data leak"))
            hidden += 1
            check(len({(p["uid"],p["action"]) for p in alternatives})==len(alternatives),"duplicate")
            for p in alternatives:
                out,reason=validate(public,cfg,orders,{p["uid"]:p["action"]},*ctx,parent._FEAT)
                different = [u for u in out.keys()|orders.keys() if out.get(u)!=orders.get(u)]
                check(reason=="accepted" and different==[p["uid"]] and mine[2][different[0]][2]=="COLLECTOR"
                      and orders.get(p["uid"],"NONE")==p["original"] and out.get(p["uid"],"NONE")==p["action"],
                      (t,seat,p["uid"],"actual order scope"))
                checked_cells = {actions.step(w[0],orders.get(u,"NONE"),15) for u,w in mine[2].items() if u!=p["uid"]}
                checked_cells.update(cell for u,cell in mine[1].items() if orders.get(u) in ("RECCOLLECTOR","RECPLANTER"))
                check(actions.step(mine[2][p["uid"]][0],p["action"],15) not in checked_cells,(t,"joint collision"))
                kinds[p["kind"]] += 1
                guards += 1
            if alternatives:
                byband[str(band)] += 1
            roots.append({"t":t,"parent":orders,"raw":len(first),
                          "valid":[[p["uid"],p["original"],p["action"],p["kind"]] for p in alternatives]})
        detail.append({"replay":card["id"],"seat":seat,"terrain":card["terrain_sha256"],"source_sha256":card["sha256"],
                       "roots":roots,"parent_prefix":deepcopy(parent._COUNTS)})
    check(set(kinds)==set(actions.KINDS) and len(byband)==3,"incomplete observational opportunity coverage")
    record={"action_sha256":sha((HERE/"actions.py").read_bytes()),"validator_sha256":sha((FROZEN/"deploy.py").read_bytes()),
            "parity_sha256":sha(Path(__file__).read_bytes()),"parent_sha256":sha(PARENT.read_bytes()),
            "panel_sha256":sha(PANEL.read_bytes()),"selection":"Two lexicographically smallest old development replay IDs per seat with 299 decisions; outcomes unused.",
            "controlled":controlled,"inspected_roots":hidden,"unsupported_roots_rejected":protected,"unchanged_after_hidden_poison":hidden,
            "raw_actions":raw,"validated":validated,"independent_collision_and_slot_checks":guards,
            "eligible_roots_by_band":dict(byband),"legal_actions_by_kind":dict(kinds),"maps":detail}
    final = HERE / "action_parity.json.gz"
    tmp = HERE / ".action_parity.tmp"
    tmp.write_bytes(gzip.compress((json.dumps(record,separators=(",",":"),allow_nan=False)+"\n").encode(),mtime=0))
    tmp.replace(final)
    print({k:v for k,v in record.items() if k not in ("maps","controlled")})


if __name__=="__main__":
    main()
