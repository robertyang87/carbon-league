"""Add the 15 new numeric games to the previously attested 1,488-board exclusion set."""
from pathlib import Path
import gzip
import hashlib
import json

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]


def load(p):
    return json.loads(p.read_bytes())


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    old=load(HERE.parent/"appendix_97de1ebc/exposure.json")
    p=HERE/"validation_maps.jsonl.gz"
    raw=gzip.decompress(p.read_bytes())
    new=[json.loads(line) for line in raw.splitlines()]
    entry=[str(p.relative_to(BASE)),len(new),sha(raw),sha("\n".join(sorted(x["id"] for x in new)).encode())]
    sources=old["identity_sources"]+[entry]
    terrains=set();seeds=set();ids=set()
    for filename,count,raw_sha,id_sha in sources:
        raw=gzip.decompress((BASE/filename).read_bytes())
        rows=[json.loads(line) for line in raw.splitlines()]
        if len(rows)!=count or sha(raw)!=raw_sha or sha("\n".join(sorted(x["id"] for x in rows)).encode())!=id_sha:
            raise ValueError(filename)
        for x in rows:
            if x["id"] in ids or x["seed"] in seeds or x["initial_terrain_u8_sha256"] in terrains:
                raise ValueError("duplicate game, seed or terrain")
            ids.add(x["id"]);seeds.add(x["seed"]);terrains.add(x["initial_terrain_u8_sha256"])
    if len(ids)!=1503:
        raise ValueError("incomplete ledger")
    A=BASE/"master6_actor_dev"
    paths=[A/"symmetry_dev/clearance.json",A/"economy_aux_dev/native_dev/clearance.json",
           A/"tree_response_training/clearance.json",A/"tree_response_training/next_intake.json",
           A/"tree_response_training/supplement_01/registration.json"]
    cross={}
    for path in paths:
        d=load(path)
        sets=[(b["name"],b["terrain_at_queue_time"]) for b in d["batches"]] if "batches" in d else [(str(path.relative_to(BASE)),d["boards"])]
        for label,boards in sets:
            if set(map(int,boards)) & seeds or set(boards.values()) & terrains:
                raise ValueError(f"overlap: {label}")
            cross[label]=list(map(int,boards))
    result={"revision":"2cc7c3588279454823a2fd816e6d8ad593166125","complete_two_scores":1502,
            "official_fault_prefix":1,"distinct_ids_seeds_and_initial_terrains":len(ids),
            "new_source":entry,"all_source_sha256":sha(json.dumps(sources,separators=(",",":")).encode()),
            "previous_receipt_sha256":sha((HERE.parent/"appendix_97de1ebc/exposure.json").read_bytes()),
            "checked_development_and_training_zero_overlap":cross}
    (HERE/"exposure.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
