"""Dispatch-only helper: s30echo3-dispatch.py <script> <instance_id> <marker>
Prints the UTC dispatch instant so the caller can claim in a tight window."""
import sys, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api
from datetime import datetime, timezone

sh_file, inst, MARK = sys.argv[1], sys.argv[2], sys.argv[3]
script = open(sh_file).read()
assert MARK in script, "marker not in script"
gh_api.dispatch("robertyang87", "carbon-league", "cloud-assistant.yml", ref="main",
                inputs={"instance_id": inst, "region": "cn-shanghai",
                        "command_b64": base64.b64encode(script.encode()).decode()})
print("DISPATCHED", inst, MARK, datetime.now(timezone.utc).isoformat())
