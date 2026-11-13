"""Ladder audit stub: _submit_state semantics + _recover_post no-resend.
No real network: ladder.request is replaced wholesale."""
import os, sys
CODE = os.path.expanduser("~/workspace/final-accept/code")
sys.path.insert(0, CODE)
import ladder as L

# 1) _submit_state: name present with an eval-failed status -> APPLIED
L.request = lambda path, *a, **k: {"strategies": [{"name": "s1", "status": "eval_failed", "rating": None}]}
assert L._submit_state("s1") == L.APPLIED, L._submit_state("s1")
print("PASS created-but-eval-failed counts APPLIED (name in /api/mine only)")
# absent -> UNKNOWN ; request error -> UNKNOWN (never REJECTED)
L.request = lambda path, *a, **k: {"strategies": []}
assert L._submit_state("s1") == L.UNKNOWN
def boom(path, *a, **k): raise TimeoutError("lag")
L.request = boom
assert L._submit_state("s1") == L.UNKNOWN
print("PASS absent/error -> UNKNOWN, never REJECTED")

# 2) ambiguous POST (possibly sent) + not found -> stop, exactly 1 POST
posts = {"n": 0}
def do_post():
    posts["n"] += 1
    raise L.AmbiguousRequestError("connection dropped mid-submit", possibly_sent=True)
try:
    L._recover_post("submit 's1'", do_post, lambda: L.UNKNOWN, manual="check mine first")
    print("FAIL expected AmbiguousRequestError")
    sys.exit(1)
except L.AmbiguousRequestError:
    pass
assert posts["n"] == 1, posts
print("PASS UNKNOWN after ambiguous POST -> stopped, exactly 1 POST, no auto-resend")

# 3) ambiguous POST + state APPLIED -> confirmed-applied, still 1 POST
posts["n"] = 0
r = L._recover_post("submit 's1'", do_post, lambda: L.APPLIED, manual="")
assert r == "confirmed-applied" and posts["n"] == 1, (r, posts)
print("PASS APPLIED on state check -> confirmed-applied without resend")
print("== ladder stub: all passed ==")
