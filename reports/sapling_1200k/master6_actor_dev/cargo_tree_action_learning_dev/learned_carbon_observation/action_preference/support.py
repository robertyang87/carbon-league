"""Reconstruct both historical public decision inputs and measure carbon exposure before any fit."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import F, ROOT, read, save, sha
from runtime import _carbon_view
sys.path.insert(0, str(ROOT / "reports/sapling_1200k/root_recovery_branches"))
from branches import begin, clean, public


def digest(obj):
    return sha(json.dumps(obj,ensure_ascii=False,sort_keys=True,allow_nan=False,separators=(",",":")).encode())


def option(map_values, own_bases, own_workers, uid, cmd, order):
    if uid in own_bases:
        pos = order[own_bases[uid]]
        neighborhood = F.DEST[pos]
        # A pass has no spawned role; recruiting two different roles gives separate channels.
        out = np.zeros((2,3))
        if cmd != "NONE":
            role = ("RECCOLLECTOR","RECPLANTER").index(cmd)
            out[role] = [map_values[pos],map_values[neighborhood[1:]].mean(),map_values[neighborhood].mean()]
        return out.flatten(), set(map(int,neighborhood))
    physical,_,role = own_workers[uid]
    dest = order[F.DEST[physical,F.MOVES.index(cmd)]]
    neighbors = F.DEST[dest,1:]
    all_bases = order[list(own_bases.values())]
    distance = F.DIST[dest,all_bases].min()
    shortest = (F.DIST[dest,:,None] + F.DIST[:,all_bases] == distance).any(axis=1)
    assert shortest[dest]
    values = [map_values[dest], map_values[neighbors].mean(), map_values[shortest].mean()]
    return np.array(values), set(map(int,neighbors)) | set(map(int,np.flatnonzero(shortest)))


def summarize(rows):
    return dict(roots=len(rows),terrains=len({r["terrain"] for r in rows}),
                whole_board_changed=sum(r["changed_map_cells"]>0 for r in rows),
                either_footprint_changed=sum(r["changed_footprint_cells"]>0 for r in rows),
                any_arm_changed=sum(r["max_any_arm_carbon_change"]>1e-6 for r in rows),
                option_contrast_changed=sum(r["max_contrast_change"]>1e-6 for r in rows),
                contrast_ge_one_carbon=sum(r["max_contrast_change"]>=1 for r in rows),
                signed_alt_gained=sum(r["alt_minus_parent"]>0 for r in rows),signed_alt_lost=sum(r["alt_minus_parent"]<0 for r in rows))


def main():
    registry = HERE.parents[2] / "joint_response_critic_dev/signed_roots.json.gz"
    # HERE.parents[2] is the historical actor experiment root.
    audit_file = HERE.parents[1] / "paired_response_prefix/signed.json.gz"
    book, audit = read(registry), {r["case"]:r for r in read(audit_file)["rows"]}
    assert len(book["rows"]) == len(audit) == 108
    weights_file = HERE.parent/"weights.npz"
    parts = np.load(weights_file)
    network = [parts[k] for k in "wbva"]
    train_terrains = {c["terrain"] for c in read(HERE.parent/"protocol.json")["cards"] if c["split"]=="train"}
    by_source = defaultdict(lambda: defaultdict(list))
    for r in book["rows"]:
        by_source[r["source"]][r["t"]].append(r)
    rows = []
    for source, times in sorted(by_source.items()):
        path = ROOT/source
        archived = read(path)
        seat = archived["candidate_seat"]
        env = begin(archived)
        f = F.Feat(seat)
        order = F.CANON if seat else F.GRID
        for t in range(max(times)+1):
            obs = public(env,seat)
            assert set(obs) == {"step","player","players","trees","carbon","remainingOverageTime"}
            x = f.observe(obs)
            if t in times:
                own = obs["players"][seat]
                old = x["maps"][0].astype(np.float64)*(100/255)
                new = _carbon_view(x,f,F,network)["maps"][0].astype(np.float64)*(100/255)
                changed = old != new
                for r in times[t]:
                    a = audit[r["case"]]
                    assert sha(path.read_bytes())==a["source_sha256"]==r["source_sha256"]
                    assert r["parent_sha256"]==archived["pack_sha256"]==book["frozen_parent_sha256"]
                    assert a["terrain"]==r["terrain"]==archived["initial_terrain_u8_sha256"]
                    assert digest({k:v for k,v in obs.items() if k!="remainingOverageTime"})==r["public_root_sha256"]
                    assert a["prefix_verdict"]=="matched_prefix_and_root_enemy" and a["free_from_preexisting_divergence_and_same_root_enemy"]
                    assert a["official_scores_physical"] == [r["parent_scores"],r["alternative_scores"]]
                    assert clean(archived["actions"][t][seat])==r["parent"] and clean(archived["actions"][t][1-seat])==r["root_opponent"]
                    pair = {k:r[k] for k in ("parent","alternative")}
                    assert digest([r["terrain"],seat,t,r["public_root_sha256"],pair,r["root_opponent"]]) == r["identity"]
                    unit_rows, footprint = [], set()
                    for uid in r["changed_uids"]:
                        vectors = []
                        for b in (old,new):
                            arms = []
                            for side in ("parent","alternative"):
                                cmd = r[side].get(uid,"NONE")
                                values, cells = option(b,own[1],own[2],uid,cmd,order)
                                arms.append(values)
                                footprint.update(cells)
                            vectors.append(np.stack(arms))
                        first, second = vectors
                        unit_rows.append(dict(uid=uid,parent=r["parent"].get(uid,"NONE"),alternative=r["alternative"].get(uid,"NONE"),
                            max_either_arm=float(abs(second-first).max()),
                            max_contrast=float(abs((second[1]-second[0])-(first[1]-first[0])).max()),
                            old_features=first.tolist(),learned_features=second.tolist()))
                    delta = round((r["alternative_scores"][seat]-r["alternative_scores"][1-seat]) -
                                  (r["parent_scores"][seat]-r["parent_scores"][1-seat]),2)
                    assert delta==a["delta_official_margin"]
                    rows.append(dict(case=r["case"],identity=r["identity"],seed=r["seed"],terrain=r["terrain"],seat=seat,t=t,
                         role=r["role"],family=r["family"],source=source,carbon_estimator_train=r["terrain"] in train_terrains,
                         official_margins=a["official_margins"],alt_minus_parent=delta,changed_map_cells=int(changed.sum()),
                         changed_footprint_cells=int(changed[sorted(footprint)].sum()),footprint_cells=len(footprint),
                         max_any_arm_carbon_change=max(u["max_either_arm"] for u in unit_rows),
                         max_contrast_change=max(u["max_contrast"] for u in unit_rows),changed_units=unit_rows))
            if t<max(times):
                env.step(archived["actions"][t])
    assert len(rows)==108 and len({r["terrain"] for r in rows})==18
    supported=[r for r in rows if r["max_contrast_change"]>1e-6]
    g=dict(total_support=len(supported),support_maps=len({r["terrain"] for r in supported}),
           positive_support_maps=len({r["terrain"] for r in supported if r["alt_minus_parent"]>0}),
           negative_support_maps=len({r["terrain"] for r in supported if r["alt_minus_parent"]<0}))
    success = g["total_support"]>=24 and g["support_maps"]>=8 and min(g["positive_support_maps"],g["negative_support_maps"])>=3
    save(HERE/"support_rows.json.gz",rows)
    save(HERE/"support.json",dict(protocol_sha256=sha((HERE/"support_protocol.json").read_bytes()),source_sha256=sha(registry.read_bytes()),
         official_audit_sha256=sha(audit_file.read_bytes()),weights_sha256=sha(weights_file.read_bytes()),row_sha256=sha((HERE/"support_rows.json.gz").read_bytes()),
         overall=summarize(rows),support_only=summarize(supported),thresholds_passed=success,counts=g,
         outside_estimator_train=summarize([r for r in rows if not r["carbon_estimator_train"]]),
         by_seed={s:summarize([r for r in rows if r["seed"]==s]) for s in sorted({r["seed"] for r in rows})},
         by_phase={str(s):summarize([r for r in rows if s<=r["t"]<s+50]) for s in range(0,300,50)},
         decision="eligible_to_preregister" if success else "stop_before_fitting"))
    print(summarize(rows),g,success,flush=True)


if __name__=="__main__":
    main()
