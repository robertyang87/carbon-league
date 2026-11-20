"""Register identity and split by physical map; no terminal margin or action used for selection."""
import hashlib
import json
import zipfile
from pathlib import Path

H = Path(__file__).resolve().parent
D = H.parent
ROOT = D.parents[1]
PARENT = D / "advantage_policy_update/v2_bc.py"
PARENT_SHA = "cd230d40a08f7dea07e261202b865f951b88b3b030febc9828627aa5f47c7d2a"
ORIGINAL = "65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def get(path):
    return json.loads(path.read_bytes())


def save(path, value):
    data = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    if path.exists() and path.read_bytes() != data:
        raise ValueError(f"Refusing to overwrite differing registered output {path}")
    path.write_bytes(data)
    return sha(data)


def main():
    assert sha(PARENT.read_bytes()) == PARENT_SHA
    blocked_seeds = {r[0] for r in get(D / "root_rotation_policy/terminal/protocol.json")["keys"]}
    ticket_sources = [get(D / n)["tickets"] for n in ("action_rank_data/protocol.json",
                      "tree_action_outcomes/protocol.json", "parallel_responses/protocol.json")]
    ticket_sources.append([r["ticket"] for r in get(D / "return_response_learning/close_games/menus.json")["rows"]])
    blocked = {r["terrain"] for group in ticket_sources for r in group if r["seed"] in blocked_seeds}
    assert len(blocked) == len(blocked_seeds)
    cards = []
    for src in get(D / "planter_policy_update/protocol.json")["tickets"]:
        assert src["terrain"] not in blocked and src["replay_seed"] not in blocked_seeds
        cards.append(dict(domain="ladder", source=src["path"], sha256=src["sha256"], repeat=src["match"],
                          terrain=src["terrain"], seed=src["replay_seed"], seats=[src["seat"]],
                          own=src["source_package_sha256"], split=src["split"], name=src["opponent"]))
    core = []
    for series in ("dev1","cold_start/development","identity_replication_dev","paired_improvement_dev",
                   "calibration_improvement_dev","nn_policy_development","v3_release/development"):
        folder = D / series / "hosted"
        ledger = get(folder / "settlement.json")
        archives = {(a["seed"],a["seat"]):a for a in ledger["archives"]}
        for g in ledger["games"]:
            if g["seed"] in blocked_seeds or g["terrain"] in blocked or g["status"] != "complete":
                continue
            key = (g["seed"],g["seat"])
            a = archives[key]
            path = folder / f"{a['id']}.zip"
            if not path.exists():
                continue
            assert sha(path.read_bytes()) == a["sha256"]
            with zipfile.ZipFile(path) as z:
                reg = json.loads(z.read(g["arm"] + "/registration.json"))
            if reg["files"]["opponent"]["sha256"] != ORIGINAL:
                continue
            core.append(dict(domain="h2h", source=str(path.relative_to(ROOT)), sha256=a["sha256"],
                             repeat=g["arm"], terrain=g["terrain"], seed=g["seed"], seats=[0,1],
                             own=reg["files"]["candidate"]["sha256"], candidate=g["seat"], name=series))
    terrains = sorted({t["terrain"] for t in core},key=lambda s:sha(("native-critic-map-v1/"+s).encode()))
    held = set(terrains[:max(3,round(len(terrains)*.25))])
    for c in core:
        c["split"] = "held" if c["terrain"] in held else "fit"
    cards.extend(core)
    assert not ({r["terrain"] for r in cards if r["split"]=="fit"} & {r["terrain"] for r in cards if r["split"]!="fit"})
    assert {r["split"] for r in cards}=={"fit","held"}
    doc = dict(parent=PARENT_SHA, excluded_seeds=sorted(blocked_seeds), excluded_terrain=sorted(blocked),
      tickets=cards, steps=list(range(6,299,15)), target="Native _terminal: clip(own official margin/3000,-2.5,2.5)+.5*sign(margin).",
      inputs="Observed own seat belief from prior actual own commands and public carbon, actual public players and previous players; set known=all ones exactly like search look, quantize 54 planes and use the native five scalars. No full-carbon, next actions or opposing actual commands. This is a proxy for a simulated state's estimated map; paired true H8 search leaf evaluation is independent.",
      objective="On the frozen policy's native 96x225 h, update six float32 v1/v2/v3 tensors from v4. Each terrain equal, then each game, seat and registered available time. Full-batch Huber delta=0.75 to official terminal +0.5 MSE to initial native critic. Adam lr=.0003, no decay, 40 steps, no early stopping/no hyperparameter search. Store float16 as parent and score that model.",
      test="Held maps reported separately for exact original H2H and official ladder. Primary all-source map-equal MAE strictly below original and no H2H MAE regression. Show MSE/signed bias/sign accuracy, mean outcome per fixed predicted value bin, 3 phases, map-cluster intervals and each map. Evaluate actual commands with known afterstate pairs separately. Actor unchanged, both backends. No new matches.")
    print(save(H / "protocol.json", doc),len(core),len(terrains),len(held),len(cards))

if __name__=="__main__":
    main()
