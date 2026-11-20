"""Restore each public history, execute the rebuilt single file, and verify all 64 frozen commands."""
import gzip
import json
from pathlib import Path
import tempfile
import time
import types

import numpy as np
import corpus
from export import rebuild, digest

HERE = Path(__file__).resolve().parent


def verify():
    blob = rebuild()
    pack = types.ModuleType("exported_8da4_training_probe")
    pack.__file__ = str(HERE/"research_agent.py")
    exec(compile(blob, pack.__file__, "exec"), vars(pack))
    rowfile = HERE/"orders.json.gz"
    result = json.loads((HERE/"result.json").read_bytes())
    if digest(rowfile.read_bytes()) != result["orders_sha256"]:
        raise ValueError("changed expected physical orders")
    rows = json.loads(gzip.decompress(rowfile.read_bytes()))
    lengths, timing = [], []
    for card in corpus.sources()["cards"]:
        raw = corpus.json_file(corpus.ROOT/card["source"],card["source_sha256"])
        reply = corpus.json_file(corpus.ROOT/card["response"],card["response_sha256"])
        first, seat = card["first_step"], card["seat"]
        tape = raw["actions"][:first-1] + reply["tape"]
        sim = corpus.FastCarbon(raw["game_configuration"]).reset(seed=card["seed"])
        expected = [r for r in rows if r["gid"] == card["gid"]]
        for t, pair in enumerate(tape[:first+len(expected)]):
            obs = sim.obs(seat)
            begin = time.perf_counter()
            physical = pack.agent(obs,raw["game_configuration"])
            if t >= first:
                timing.append((time.perf_counter()-begin)*1000)
                row = expected[t-first]
                if row["step"]!=t or row["updated_int8"]!=physical:
                    raise ValueError(("single-file public agent disagreement",card["seed"],seat,t))
            sim.step(*pair)
        lengths.append(dict(seed=card["seed"],seat=seat,public_calls=first+len(expected),checked=len(expected)))
    record = dict(package_bytes=len(blob),package_sha256=digest(blob),inputs=result["orders_sha256"],cards=lengths,
                  checked=len(timing),warm_local_public_ms={"mean":float(np.mean(timing)),"p95":float(np.percentile(timing,95)),"max":max(timing)},
                  timing_scope="local single-thread Python entry without platform sandbox or transport")
    with tempfile.NamedTemporaryFile(dir=HERE,delete=False) as f:
        f.write((json.dumps(record,indent=2)+"\n").encode())
    Path(f.name).replace(HERE/"pack_check.json")
    print(record,flush=True)


if __name__=="__main__":
    verify()
