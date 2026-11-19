"""Build the untouched remainder of the original Top Five validation split."""
import argparse
import gzip
import json
from pathlib import Path
from courier_train_labels import ELITE_SHA, ROOT, game_data, sha


def main(a):
    raw=a.elite.read_bytes()
    first=json.loads(gzip.decompress(a.previous.read_bytes()))
    if sha(raw)!=ELITE_SHA or first["parent_corpus_sha256"]!=ELITE_SHA:
        raise ValueError("unexpected feature set")
    used={g["seed"] for g in first["games"]}
    games=[g for g in json.loads(raw)["games"] if g["split"]=="val" and g["seed"] not in used]
    if len(games)!=11:
        raise ValueError("held-out pool changed")
    archive=json.loads((ROOT/"reports/master_family/tactical_rules/top5_inventory.json").read_bytes())["replays"]
    rows=[game_data(g,a.elite,a.replays,archive) for g in games]
    result=dict(first_label_sha256=sha(a.previous.read_bytes()),builder_sha256=sha(Path(__file__).read_bytes()),games=rows)
    with a.out.open("xb") as f:
        f.write(gzip.compress(json.dumps(result,separators=(",",":"),allow_nan=False).encode(),mtime=0))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("elite","previous","replays","out"):
        p.add_argument("--"+name,type=Path,required=True)
    main(p.parse_args())
