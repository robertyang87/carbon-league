"""Attach the final-plant comparison to the frozen single-network V5 payload."""
from pathlib import Path
from hashlib import sha256
import tempfile

HERE=Path(__file__).resolve().parent
WEIGHTS=HERE.parents[1]/"data/weights/sapling_1200k"


def main():
    source=(WEIGHTS/"soft_it50_v5_agent.py").read_bytes()
    if sha256(source).hexdigest()!="dceed7891e9ff25cf9e211e1c9b61d979d27a253b4c0935f2439804372b31389":
        raise ValueError("unexpected parent pack")
    header,delim,tail=source.partition(b"\n_AG = None\n")
    if not delim or b"def agent(" not in tail:
        raise ValueError("unexpected packed entry")
    result=header+b"\n"+(HERE/"v5_final_plant_runtime.py").read_bytes()
    if len(result)>10_000_000:
        raise ValueError("size limit")
    with tempfile.NamedTemporaryFile(dir=WEIGHTS, delete=False) as f:
        f.write(result)
    Path(f.name).replace(WEIGHTS/"soft_it50_v5_final_plant_agent.py")
    print(sha256(result).hexdigest(),len(result))


if __name__=="__main__":
    main()
