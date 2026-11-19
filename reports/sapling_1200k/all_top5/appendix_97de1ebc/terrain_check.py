"""Include known error-game terrain in the same exclusion ledger as the numeric cohort."""
from pathlib import Path
import gzip
import hashlib
import json

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
A=BASE/"master6_actor_dev"
T=A/"tree_response_training"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(p):
    return json.loads(p.read_bytes())


def main():
    files=[BASE/r["source"] for r in read(A/"symmetry_dev/ladder_overlap_336e8b8c.json")["sources"]]
    files.extend([HERE/"validation_maps.jsonl.gz",HERE/"unscored/validation_maps.jsonl.gz"])
    games={}; ids=set(); seeds=set(); terrains=set(); sources=[]
    for path in files:
        raw=gzip.decompress(path.read_bytes())
        rows=[json.loads(line) for line in raw.splitlines()]
        for r in rows:
            if r["id"] in ids or r["seed"] in seeds or r["initial_terrain_u8_sha256"] in terrains:
                raise ValueError("duplicate game, seed or actual board")
            ids.add(r["id"]);seeds.add(r["seed"]);terrains.add(r["initial_terrain_u8_sha256"])
        sources.append([str(path.relative_to(BASE)),len(rows),sha(raw),sha("\n".join(sorted(r["id"] for r in rows)).encode())])
    if len(ids)!=1488:
        raise ValueError("incomplete public ledger")
    sets={"symmetry_development":read(A/"symmetry_dev/clearance.json")["boards"]}
    for src in (T/"clearance.json",T/"next_intake.json",T/"supplement_01/registration.json"):
        doc=read(src)
        if "boards" in doc:
            sets[src.parent.name]=doc["boards"]
        else:
            sets.update({b["name"]:b["terrain_at_queue_time"] for b in doc["batches"]})
    result={}
    for tag,boards in sets.items():
        seed_overlap=seeds & set(map(int,boards))
        terrain_overlap=terrains & set(boards.values())
        result[tag]={"seeds":list(map(int,boards)),"seed_hits":sorted(seed_overlap),"terrain_hits":sorted(terrain_overlap)}
    if any(v["seed_hits"] or v["terrain_hits"] for v in result.values()):
        raise ValueError("development or training exposure overlapped new ladder maps")
    out={"revision":"97de1ebc1834785a6648f5876ce803311354a63b","verified_scored":1487,"fault_prefix":1,
         "distinct_terrain_for_exclusion":len(terrains),"nonoverlap_with_registered_training_and_development":result,
         "identity_sources":sources}
    (HERE/"exposure.json").write_text(json.dumps(out,indent=2)+"\n")


if __name__=="__main__":
    main()
