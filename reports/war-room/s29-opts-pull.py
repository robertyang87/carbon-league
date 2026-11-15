"""One-off read-only pull (war-room 01:45): fetch _OPTS + make_agent source
from packs/n1r_int8.py on spot-29 via the L20->DGX relay. Read-only:
no file on the target is written; the backfill loops are untouched.
Reuses the fable loop's dispatch/claim machinery (import is safe:
its main() is __main__-guarded)."""
import importlib.util, sys, time

BASE = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files"
spec = importlib.util.spec_from_file_location("snap", BASE + "/s29-snaploop-fable.py")
snap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snap)

PAYLOAD = r"""
echo "PAYLOAD-BEGIN"
echo "--- LS /root/carbon/code ---"
ls /root/carbon/code 2>/dev/null
F=/root/carbon/code/cnet2_pack.py
echo "FILE=$F LINES=$(wc -l < "$F")"
echo "--- ALL _OPTS / make_agent MENTIONS in cnet2_pack.py ---"
grep -n '_OPTS\|def make_agent' "$F"
A=$(grep -n '_OPTS *=' "$F" | head -n 1 | cut -d: -f1)
echo "OPTS-ASSIGN-LINE=$A"
if [ -n "$A" ]; then echo "--- OPTS WINDOW ---"; sed -n "${A},$((A+39))p" "$F"; fi
M=$(grep -n 'def make_agent' "$F" | head -n 1 | cut -d: -f1)
echo "MAKE-AGENT-LINE=$M"
if [ -n "$M" ]; then echo "--- MAKE-AGENT WINDOW ---"; sed -n "${M},$((M+14))p" "$F"; fi
echo "PAYLOAD-END"
"""

TEMPLATE = """MARK=__MARKER__
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 300 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
MARK=__MARKER__
TGT=47.117.170.99
KEY=/home/arm/.ssh/id_ed25519_carbon_relay
echo "$MARK DGX-BEGIN"
timeout -k 30 180 ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=20 root@"$TGT" 'bash -s' <<'TGTEOF'
echo "TGT-BEGIN"
date '+%F %T %Z'
__PAYLOAD__
echo "TGT-END"
TGTEOF
echo "TGT-RC=$?"
echo "$MARK DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
"""

marker = "S29OPTS" + time.strftime("%H%M%S")
script = TEMPLATE.replace("__MARKER__", marker).replace("__PAYLOAD__", PAYLOAD)
t0 = time.time()
snap.dispatch(script)
print(f"dispatched marker={marker}", flush=True)
log = snap.claim(marker, timeout_s=600, not_before=t0)
if log is None:
    print("NOT-CLAIMED")
    sys.exit(1)
i = log.rfind("TGT-BEGIN")
j = log.rfind("TGT-END")
print(log[i:j + 7] if i >= 0 and j > i else log[-20000:])
