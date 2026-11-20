"""Extend the existing two-map registration to the new ladder opening without changing its receipt."""
import argparse
import gzip
import importlib.util
import io
import json
from pathlib import Path
import tarfile

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("two_map_registration",HERE/"clearance.py")
prior=importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
sha=prior.sha
git=prior.git
BASE_RECEIPT="7c5215c091fda4d2500e28303a923595c7f8b32e4c3e9ed166feeb46cc1d85a8"
BASE_VERIFIER="9aaf1f58ef69649ade54265f1ad1c7687e8b8125d7dc0d96b524374f8144b9ca"
ADD="reports/sapling_1200k/all_top5/appendix_ddf7e2ec"
GAME="fcc8c3d54ef85d7c13f1083632c57ef2-1"
SEED=1125564511
LAYOUT="b103e3031444df568b61270ed0799701b1e8f81a74ec7a41d39b80f2c0b0ce7a"


def single_ledger(commit, boards):
    def read(path):
        return git("show",commit+":"+path)
    record=read(ADD+"/validation_maps.jsonl.gz")
    rows=[json.loads(v) for v in gzip.decompress(record).splitlines()]
    expected=dict(id=GAME,seed=SEED,initial_terrain_u8_sha256=LAYOUT)
    if rows!=[expected]:
        raise ValueError("Wrong supplemental map ledger")
    manifest=read(ADD+"/manifest.json");cards=json.loads(manifest)["games"]
    if len(cards)!=1 or cards[0]["id"]!=GAME:
        raise ValueError("Wrong official scorecard")
    c=cards[0]
    path=c["archive_location"]["archive"]
    member=c["archive_location"]["member"]
    packed=read(path)
    with tarfile.open(fileobj=io.BytesIO(packed)) as tar:
        body=tar.extractfile(member).read()
    game=json.loads(gzip.decompress(body))
    obs=game["steps"][0][0]["observation"]
    actual=prior.old.engine.map_identity.terrain_sha256(obs["full_carbon"])
    if actual!=LAYOUT or sha(body)!=c["sha256"] or c["seed"]!=game["configuration"]["randomSeed"] or c["seed"]!=SEED:
        raise ValueError("Physical map, source or seed disagreement")
    if (game["id"]!=c["environment_uuid"] or game["rewards"]!=c["score_by_seat"] or
            game["statuses"]!=["DONE","DONE"] or len(game["steps"])-1!=c["transitions"]):
        raise ValueError("Different match or invalid end")
    prior.admit([dict(seed=SEED,terrain=actual)],boards)
    return dict(**expected,archive=path,member=member,tar_sha256=sha(packed),replay_sha256=sha(body),manifest_sha256=sha(manifest),
                official_reward_by_seat=game["rewards"],seats=c["seat_names"],physical_layout_recomputed_from="steps[0][0].observation.full_carbon",
                ledger_path=ADD+"/validation_maps.jsonl.gz",ledger_sha256=sha(record),official_transitions=c["transitions"])


def record(endpoint=None):
    baseline=(HERE/"clearance.json").read_bytes()
    if sha(baseline)!=BASE_RECEIPT or sha((HERE/"clearance.py").read_bytes())!=BASE_VERIFIER:
        raise ValueError("Previously frozen clearance changed")
    base=json.loads(baseline)
    boards=base["by_seed"]
    if boards!=prior.old.boards():
        raise ValueError("Changed physical development openings")
    start=base["through"]
    result=dict(by_seed=boards,remote_extension=[])
    end=endpoint or prior.checked_remote()
    initial=start
    while True:
        prior.extend(result,start,end)
        start=end
        observed=endpoint or prior.checked_remote()
        if end==observed:
            break
        end=observed
    src=single_ledger(end,boards)
    added=prior.verify_added_boards(initial,end,boards)
    if not any(r["terrain"]==LAYOUT and r["seed"]==SEED for r in added.values()):
        raise ValueError("Newly archived game missing from independent opening census")
    ref=result["remote_extension"][-1]
    if ref["top5_unique_openings"]<1650 or src["ledger_path"] not in ref["top5_ledgers"] or ref["top5_ledgers"][src["ledger_path"]]!=src["ledger_sha256"]:
        raise ValueError("Latest canonical ledger population omitted the addition")
    return dict(schema="sapling-midgame-economic-development/clearance-increment/1",after=initial,through=end,
                through_tree=git("rev-parse",end+"^{tree}").decode().strip(),baseline_sha256=BASE_RECEIPT,base_verifier_sha256=BASE_VERIFIER,
                verifier_sha256=sha(Path(__file__).read_bytes()),physical_boards=boards,seed_intersections=[],physical_intersections=[],
                new_ladder_opening=src,top5_unique_openings=ref["top5_unique_openings"],top5_scored=ref["top5_unique_openings"]-len(ref["top5_faults"]),
                top5_faults=ref["top5_faults"],new_archive_openings=added,deltas=result["remote_extension"])


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through",help="Pin a specific GitHub endpoint for replay")
    args=parser.parse_args()
    res=record(args.through)
    dest=HERE/"clearance_increment.json"
    data=(json.dumps(res,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode()
    if dest.exists() and dest.read_bytes()!=data:
        raise ValueError("A different incremental receipt already exists")
    tmp=dest.with_suffix(".partial")
    tmp.write_bytes(data);tmp.replace(dest)
    print(res["through"],res["through_tree"],res["top5_unique_openings"],sha(data),flush=True)
