"""Run the existing full-repository seed and terrain census at the pre-registration tree."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import types

HERE = Path(__file__).resolve().parent
CARGO = HERE.parents[1] / "cargo_tree_action_learning_dev"
TEMPLATE = CARGO / "native_actor_corpus/league_policy/clearance.py"
SOURCE = "504ee6832ae4b59f8a02916000e6d5f401cc5430"
SEEDS = list(range(621758510, 621758514))


def census():
    spec = importlib.util.spec_from_file_location("league_census", TEMPLATE)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    seen, faults, members = {}, [], {}
    for path in base.git("ls-tree","-r","--name-only",SOURCE).decode().splitlines():
        if "/all_top5/" in path and path.endswith("/validation_maps.jsonl.gz"):
            data = base.git("show",SOURCE+":"+path)
            members[path] = hashlib.sha256(data).hexdigest()
            for line in gzip.decompress(data).splitlines():
                r = json.loads(line)
                if r["id"] in seen:
                    raise ValueError("Duplicate canonical top-five map")
                seen[r["id"]] = r["seed"], r["initial_terrain_u8_sha256"]
                if "/unscored/" in path:
                    faults.append(r["id"])
    if (len(faults)!=1 or len(seen)<1647 or
        len({s for s,h in seen.values()})!=len(seen) or len({h for s,h in seen.values()})!=len(seen)):
        raise ValueError("Incomplete canonical top-five openings")
    source = base.git("show",SOURCE+":"+TEMPLATE.relative_to(base.ROOT).as_posix())
    adapted = source.decode()
    for old,new in (("len(top5)==1637",f"len(top5)=={len(seen)}"),
                    ("if b'6217584' in b and CANDIDATE.search(b)","if CANDIDATE.search(b)")):
        if adapted.count(old)!=1:
            raise ValueError("Historical scanner changed")
        adapted=adapted.replace(old,new)
    mod = types.ModuleType("pinned_bc_map_census")
    mod.__file__=str(TEMPLATE)
    exec(compile(adapted,str(TEMPLATE),"exec"),mod.__dict__)
    mod.SNAPSHOT, mod.SEEDS, mod.GROUPS = SOURCE, SEEDS, {"fresh_development":SEEDS}
    mod.CANDIDATE = re.compile(rb"(?<![.\d])62175851[0-3](?![.\d])")
    out=mod.scan()
    out.update(schema="sapling-bc-four-map-replication/clearance/1",
               purpose="Fresh normal-opening development comparison of frozen immediate-action Sapling and parent. All four maps are excluded from training and the formal examination.",
               checked_current_canonical_ledgers=members,census_implementation={
                    "repo_file":TEMPLATE.relative_to(base.ROOT).as_posix(),
                    "sha256":hashlib.sha256(source).hexdigest(),
                    "adaptation":"Use the unique count read directly from every current canonical top-five ledger; scan all structured records using the exact four-seed expression, including the changed numeric prefix."})
    return out


if __name__=="__main__":
    output=HERE/"clearance.json"
    encoded=(json.dumps(census(),sort_keys=True,indent=2)+"\n").encode()
    if output.exists() and output.read_bytes()!=encoded:
        raise ValueError("Refusing to replace a different map inventory")
    output.write_bytes(encoded)
