"""Round 3 ECS mock checks; performs no network or resource operations.
Reuses the previous committed mock harness against current tools/ecs_release.py.
CARBON_REPO selects the checkout; otherwise find it above this script.
"""
import contextlib, io, json, os, pathlib, runpy
def root():
    if os.environ.get("CARBON_REPO"):
        return pathlib.Path(os.environ["CARBON_REPO"]).resolve()
    for p in pathlib.Path(__file__).resolve().parents:
        if (p / "code/fastsim.py").is_file():
            return p
    raise RuntimeError("Set CARBON_REPO")
repo = root()
os.environ["CARBON_REPO"] = str(repo)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ns = runpy.run_path(str(repo / "reports/review_evidence/20261112_fixes/ecs_checks.py"))
cases = [
    ns["run"]("200_instance_missing_id", [(200, {"Instances": {"Instance": [{}]}, "TotalCount": 1})]),
    ns["run"]("200_instance_null_id", [(200, {"Instances": {"Instance": [{"InstanceId": None}]}, "TotalCount": 1})])
]
print(json.dumps({"previous_cases": json.loads(buf.getvalue()), "extra_structure_cases": cases}, indent=2))

