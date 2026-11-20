"""Register and run the two-map temperature-calibration test against the unmodified original."""
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
FAMILY=HERE.parent
TEMPLATE=FAMILY/"paired_improvement_dev/matrix.py"
spec=importlib.util.spec_from_file_location("common_paired_improvement",TEMPLATE)
common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
SEEDS=(621758600,621758601)
WORKFLOW="master3s2-calibration-improvement.yml"
ARM_SHA="2cf0c9ac5066b19d8375e82140a991ecc85d0a660d1e5a361ed2a184993cea58"
PARENT_SHA="7cf5c6c91fc59b46b2a38924c4c5ab56143fb1242d082f9426303af7336ac3ed"

def contract():
    p=common.paired
    prior=p.read(common.SOURCE/"protocol.json")
    if p.sha(common.SOURCE/"protocol.json")!=common.REGISTERED:
        raise ValueError("Referee source or qualification changed")
    keys=("opponent","official_helpers","rules","environment","ladder_runner","worker","container","base_registration",
          "original_game_configuration","current_guide","match_runner","replayer","native_timing_audit","previous_exposure_guard",
          "registered_runner","registered_settler","registered_sparse","observation_replay","paired_settler")
    files={k:prior["files"][k] for k in keys}
    files.update(p.read(common.RELEASE)["files"])
    for k,path in {"clearance":HERE/"clearance.json","clearance_code":HERE/"clearance.py","matrix":HERE/"matrix.py",
                   "prior_reference":common.SOURCE/"protocol.json","paired_adapter":common.SOURCE/"matrix.py",
                   "common_adapter":TEMPLATE,"release_pair":common.RELEASE,
                   "workflow":common.ROOT/".github/workflows"/WORKFLOW}.items():
        files[k]=p.runner.record(path)
    if (files["challenger"]["sha256"],files["parent"]["sha256"],files["opponent"]["sha256"])!=(ARM_SHA,PARENT_SHA,common.ORIGINAL):
        raise ValueError("Wrong calibration, parent or reference bytes")
    if not all(p.runner.record(common.ROOT/f["path"])==f for f in files.values()):
        raise ValueError("Registration input fingerprint changed")
    cleared=p.read(HERE/"clearance.json")
    if not (cleared["passed"] and cleared["verifier_sha256"]==files["clearance_code"]["sha256"] and not cleared["physical_intersections"]
            and not cleared["seed_intersections"] and set(map(int,cleared["by_seed"]))==set(SEEDS) and len(set(cleared["by_seed"].values()))==2):
        raise ValueError("Incomplete independent board clearance")
    seats=[dict(seed=seed,seat=s,order=list(common.ARMS if (i+s)%2==0 else reversed(common.ARMS))) for i,seed in enumerate(SEEDS) for s in (0,1)]
    return dict(schema="master3s2-calibrated-rollout-development/1",registration_class="development",arms=list(common.ARMS),seeds=list(SEEDS),
                physical_boards=cleared["by_seed"],schedule=seats,files=files,outcomes=prior["outcomes"],scope=prior["scope"],
                platform=prior["platform"],examiner=prior["examiner"])
if __name__=="__main__":
    common.HERE=HERE;common.RELEASE=HERE/"release.json";common.PLAN=HERE/"protocol.json"
    common.SEEDS=SEEDS;common.WORKFLOW=WORKFLOW;common.contract=contract
    raise SystemExit(common.main())
