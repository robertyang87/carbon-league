"""Run the archived S01 terminal counterexamples against this checkout.

The archived probe also asserted that guided_match was unchanged since
7b025ae5. That unrelated file identity assertion became obsolete before
this fix (the v2 wiring changed it). Keep every simulator/terminal/guard
assertion and report the actual dependency identity instead.
"""
import os
from pathlib import Path

repo = Path(__file__).resolve().parents[3]
source = repo / "reports/review_evidence/20261113_stage1_r3/terminal/teacher_terminal_probe.py"
text = source.read_text()
identity_assert = "    assert current == old, name\n"
identity_label = '"unchanged_since_7b025ae5": True'
assert text.count(identity_assert) == text.count(identity_label) == 1
text = text.replace(identity_assert, "")
text = text.replace(identity_label, '"unchanged_since_7b025ae5": current == old')
os.environ["CARBON_REPO"] = str(repo)
os.environ.setdefault("CARBON_REVIEW_OUT", str(Path(__file__).resolve().parent))
exec(compile(text, str(source), "exec"), {"__name__": "__main__", "__file__": str(source)})
