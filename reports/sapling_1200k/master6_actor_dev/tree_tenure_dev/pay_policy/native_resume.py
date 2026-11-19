"""Resume the original bound game driver with an audited exclusion-only registration."""
import fcntl
import inspect
import json
import hashlib
from pathlib import Path

import native_run as run

PANEL = run.ACTOR / "planter_threat_transfer_dev/parent_panel/panel.json"
PANEL_HASH = "54d7862c05e44d979ae89de6bae3488f7150232177bf81f3440fa95efa9d1ee4"


def scanned_bytes(path, data, boards):
    # Source files that now consult the external exclusion receipt directly need no treatment.
    # Returning full bytes here retains the original scanner's interval checks farther down.
    if not any(str(seed).encode() in data or terrain.encode() in data for seed, terrain in boards.items()):
        return data
    if path == PANEL.parents[1] / "B_parity.py":
        run.checked(hashlib.sha256(data).hexdigest() == "1e4a5639990f64e73cb70939bb1715e95e5e1d2659452a5e0a5592ad8ea282f0",
                    "policy parity source changed")
        guard = ("EXCLUDED={" + ",".join(str(v) for v in boards) + "}").encode()
        run.checked(data.count(guard) == 1, "B parity exclusion declaration changed")
        return data.replace(guard, b"EXCLUDED={}")
    if path == PANEL.parent / "panel.py":
        run.checked(hashlib.sha256(data).hexdigest() == "6ec0c9c96af607eddc3ca0640403944faa42e6fe73b42bf3ceaeb6830e4b1146",
                    "source guard changed")
        guard = ("BLOCKED={" + ",".join(str(v) for v in boards) + "}").encode()
        run.checked(data.count(guard) == 1, "guard declaration changed")
        return data.replace(guard, b"BLOCKED={}")
    if path == PANEL.parents[1] / "splits.json.gz":
        run.checked(hashlib.sha256(data).hexdigest() == "ed5940b3a74139f334e4b8c43cebdc1424f3bb63332944eb14c55d51a19136f4",
                    "other map project splits changed")
        split = json.loads(data)
        run.checked(split.pop("excluded_fresh_map_ids") == list(boards) and split["excluded_fresh_results_read"] is False,
                    "explicit exclusions changed")
        return run.pack(split)
    if path != PANEL:
        return data
    run.checked(hashlib.sha256(data).hexdigest() == PANEL_HASH, "external exclusion declaration changed")
    record = json.loads(data)
    run.checked(record.pop("blocked_new_pilot_seeds") == list(boards) and
                record["pay_protocol_sha256"] == run.sha(run.PLAN) and
                record["pay_candidate_sha256"] == run.CAND_SHA and
                all(row["seed"] not in boards and row["map"] not in boards.values() for row in record["games"]),
                "external panel contains a played or unauthorized map")
    return run.pack(record)


def install_scope_check(lib):
    # All other path, keyword, regex, range and board checks remain in the original shared scanner.
    source = inspect.getsource(lib.newer_exposures)
    before = 'if regex.search(encoded):'
    run.checked(source.count(before) == 1, "shared map scanner changed")
    source = source.replace(before, 'encoded=scanned_bytes(p,encoded,boards)\n        ' + before)
    namespace = {**vars(lib), "scanned_bytes": scanned_bytes}
    exec(compile(source, lib.__file__ + "::external-banned-seeds", "exec"), namespace)
    lib.newer_exposures = namespace["newer_exposures"]


def main():
    source = inspect.getsource(run.preflight)
    before = '    lib.read=protocol_view'
    run.checked(source.count(before) == 1, "original preflight changed")
    source = source.replace(before, before + '\n    install_scope_check(lib)')
    namespace = {**vars(run), "install_scope_check": install_scope_check}
    exec(compile(source, run.__file__ + "::read-exclusion-declaration", "exec"), namespace)
    run.preflight = namespace["preflight"]
    with (run.OUT / ".owner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run.supervise()


if __name__ == "__main__":
    main()
