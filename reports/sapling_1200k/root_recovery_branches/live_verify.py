"""Check both live-arm action tapes independently and calculate immediate versus terminal effects."""
import gzip
import json

from branches import DIRS,OUT,begin,digest,utility
from verify import replay


def main():
    ledgers={k:json.loads((OUT/f"live_{k}.json").read_bytes()) for k in ("parent","capture")}
    base=ledgers["parent"];seat=base["seat"];op=1-seat;seed=base["seed"];root=base["root"]
    if any(v["root"]!=root or v["shared_init_sha256"]!=base["shared_init_sha256"] or
           v["root_opponent_observation_sha256"]!=base["root_opponent_observation_sha256"] or
           v["root_actions"][op]!=base["root_actions"][op] for v in ledgers.values()):
        raise ValueError("arms do not share the original joint root")
    src=DIRS[seed];b=(src/f"raw_{seed}_{seat}.json.gz").read_bytes();raw=json.loads(gzip.decompress(b))
    cert=json.loads((src/"official_replay.json").read_bytes())["runs"][src.name]["games"]
    raw["initial_terrain_u8_sha256"]=next(c["map_sha256"] for c in cert if (c["seed"],c["seat"])==(seed,seat))
    actual={}
    for k,r in ledgers.items():
        blob=(OUT/r["action_tape"]).read_bytes()
        if digest(b)!=r["original_trajectory_sha256"] or digest(blob)!=r["action_sha256"]:raise ValueError("transcript changed")
        tail=json.loads(gzip.decompress(blob));run=replay(raw,seat,root,tail)
        if run["score"]!=r["score"] or run["end_t"]!=r["terminal_t"]:raise ValueError("independent terminal replay mismatch")
        env=begin(raw)
        for a in raw["actions"][:root["t"]]:env.step(a)
        trajectory=[]
        for j,a in enumerate(tail[:15],1):
            env.step(a);obs=env.states[0].observation
            trajectory.append(dict(delta_t=j,cash=[p[0] for p in obs.players],target_owner=next((s for s in (0,1) if root["tree"] in obs.players[s][3]),None)))
        actual[k]=dict(**run,early_steps=trajectory)
    dmargin=round(ledgers["capture"]["margin"]-base["margin"],4);du=ledgers["capture"]["u"]-base["u"]
    first=[]
    for j in range(15):
        p,c=[actual[k]["early_steps"][j]["cash"] for k in ("parent","capture")]
        first.append(round((c[seat]-c[op])-(p[seat]-p[op]),2))
    cash=lambda k:actual[k]["income"]
    result=dict(seed=seed,seat=seat,root_t=root["t"],live_delta_u=du,live_delta_margin=dmargin,
                own_target_tree_receipt_difference=round(cash("capture")[seat]-cash("parent")[seat],4),
                rival_target_tree_receipt_difference=round(cash("capture")[op]-cash("parent")[op],4),
                immediate_to_15step_margin_differences=first,arms=actual)
    (OUT/"live_verification.json").write_text(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2)+"\n")
    print(dmargin,du,first,flush=True)


if __name__=="__main__":main()
