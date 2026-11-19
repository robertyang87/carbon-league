"""Read-only census of additional completed live-format responses at t<=128.

Each reported branch is replayed with the original action tape. This is a prospective
source inventory outside the registered 108 pairs; it is never loaded by a fitting job.
"""
from pathlib import Path
import numpy as np
from support import F, HERE, ROOT, read, save, sha, option, _carbon_view, clean
from common import FastCarbon

BRANCH_ROOT = ROOT / "reports/sapling_1200k"
PROVEN_ORIGINAL = "65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"

def replay(archive, seed):
    return FastCarbon(archive["game_configuration"]).reset(seed=seed,full_carbon=archive["initial_full_carbon"])

def measure(protocol, params):
    p = read(protocol)
    root = p["root_t"]
    seed,seat = p["seed"],p["seat"]
    source = ROOT/p["record"]["source"]
    raw = read(source)
    pa = read(protocol.parent/"live_parent.json")
    candidates = [read(q) for q in sorted(protocol.parent.glob("live_*.json"))
                  if q.name not in ("live_protocol.json","live_shared_init.json","live_parent.json") and "arm" in read(q)]
    assert len(candidates) == 1
    aa, = candidates
    assert pa["root_actions"][1-seat] == aa["root_actions"][1-seat]
    assert pa["root_opponent_observation_sha256"] == aa["root_opponent_observation_sha256"]
    assert pa["shared_init_sha256"] == aa["shared_init_sha256"] == sha((protocol.parent/"live_shared_init.json").read_bytes())
    assert pa["protocol_sha256"] == aa["protocol_sha256"] == sha(protocol.read_bytes())
    assert pa["original_trajectory_sha256"] == aa["original_trajectory_sha256"] == sha(source.read_bytes()) == p["record"]["source_sha256"]
    assert pa["source_sha256"] == aa["source_sha256"] == PROVEN_ORIGINAL
    assert pa["pack_sha256"] == aa["pack_sha256"] and pa["statuses"] == aa["statuses"] == ["DONE","DONE"]
    arms = [pa,aa]
    for a in arms:
        tape_file=protocol.parent/a["action_tape"]
        tape=read(tape_file)
        assert tape[0] == a["root_actions"] and len(tape)-1 == a["search_calls"]
        assert sha(tape_file.read_bytes()) == a["action_sha256"]
        sim=replay(raw,seed)
        for joint in raw["actions"][:root] + tape:
            sim.step(*joint)
        np.testing.assert_allclose(sim.cash,a["score"],atol=.011,rtol=0)
    sim=replay(raw,seed)
    feat=F.Feat(seat)
    for t in range(root+1):
        obs=sim.obs(seat)
        x=feat.observe(obs)
        if t<root:
            sim.step(*raw["actions"][t])
    order=F.CANON if seat else F.GRID
    boards=[z["maps"][0].astype(np.float64)*(100/255) for z in (x,_carbon_view(x,feat,F,params))]
    actions=[clean(a["root_actions"][seat]) for a in arms]
    assert clean(raw["actions"][root][1-seat])==clean(pa["root_actions"][1-seat])
    changed=[u for u in sorted(actions[0].keys()|actions[1].keys()) if actions[0].get(u,"NONE") != actions[1].get(u,"NONE")]
    assert changed
    us=obs["players"][seat]
    records=[]
    for uid in changed:
        vectors=[]
        for b in boards:
            vectors.append(np.stack([option(b,us[1],us[2],uid,action.get(uid,"NONE"),order)[0] for action in actions]))
        old,new=vectors
        dd=(new[1]-new[0])-(old[1]-old[0])
        records.append(dict(uid=uid,commands=[a.get(uid,"NONE") for a in actions],contrast_delta=dd.tolist()))
    scores = [a['score'] for a in arms]
    follow_files = sorted([*protocol.parent.glob("follow*.json"),*protocol.parent.glob("second_turn.json")])
    follows = []
    for f in follow_files:
        compare = read(f)
        assert "executed" in compare and "parent" in compare
        follows.append({"file":f.name, "sha256":sha(f.read_bytes()),
                        "secondary_override":clean(compare["executed"])!=clean(compare["parent"])})
    return dict(path=str(protocol.parent.relative_to(ROOT)),seed=seed,seat=seat,t=root,source=str(source.relative_to(ROOT)),
                source_sha256=sha(source.read_bytes()),terrain=raw["initial_terrain_u8_sha256"],protocol_sha256=sha(protocol.read_bytes()),
                parent_pack=pa["pack_sha256"],family=p["family"],alt_arm=aa["arm"], changed=records,
                scores=scores, root_commands=actions, max_contrast=max(abs(z) for r in records for z in r["contrast_delta"]),
                board_changed=int(np.sum(boards[0]!=boards[1])),
                secondary_decision_receipts=follows, observed_root_parent_equal=clean(raw["actions"][root][seat])==actions[0],
                delta_margin=round((scores[1][seat]-scores[1][1-seat])-(scores[0][seat]-scores[0][1-seat]),2),
                receipts=[a["action_sha256"] for a in arms])

def main():
    signed=read(HERE.parents[2]/"joint_response_critic_dev/signed_roots.json.gz")["rows"]
    # Repeat samples of a source, seat and turn already present in the 108 add no new state.
    old={(r["source_sha256"],r["seat"],r["t"]) for r in signed}
    weights=np.load(HERE.parent/"weights.npz")
    params=[weights[k] for k in "wbva"]
    rows=[]
    for path in sorted(BRANCH_ROOT.rglob("live_protocol.json")):
        p=read(path)
        t=p.get("root_t",999)
        if t>128 or "record" not in p:continue
        if (p["record"].get("source_sha256"),p["seat"],t) in old:continue
        rows.append(measure(path,params))
    def counts(group):
        return dict(pairs=len(group), source_seat_time_roots=len({(r["source_sha256"],r["seat"],r["t"]) for r in group}),
                    terrains=len({r["terrain"] for r in group}),continuation_packages=len({r["parent_pack"] for r in group}))
    save(HERE/"extra_sources.json",dict(all_cases=counts(rows), nonzero_contrast=counts([r for r in rows if r["max_contrast"]>1e-6]),
         contrast_at_least_one=counts([r for r in rows if r["max_contrast"]>=1]),
         caveat="These are heterogeneous historical responder experiments. Secondary decision receipts are retained when available. Actual additional interventions make a terminal score a multi-step return. No unmeasured team action has a score.",
         criteria="All existing reports/sapling_1200k/**/live_protocol.json at root t<=128 with a source record, excluding source_sha256/seat/t triples already in the 108 registered rows. This inventory intentionally accepts heterogeneous continuation packages and separately records duplicate measurements. Non-live-schema archives omitted. No new policy calls or fit.",rows=rows))
    for r in rows:print(r["path"].split("master6_actor_dev/")[-1],r["seed"],r["seat"],r["t"],r["board_changed"],round(r["max_contrast"],4),r["delta_margin"],flush=True)
if __name__=="__main__":main()
