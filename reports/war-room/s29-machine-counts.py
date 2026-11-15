"""One-off: wc -l all eight h29 bigeval_live files on spot-29.

C1 escalation from the 03:42 reporter run: four tags stalled ~42 min with
complete seed prefixes and empty gap lists. Store evidence says the stall
is loop-side (seed-indexed window over a completion-ordered file), but the
machine-side row counts confirm the source files really hold 200 rows each
(and the fix can therefore unblock them). Reuses the fableg1 loop module's
dispatch/claim machinery; marker is unique to this probe.
"""
import base64, sys, time
sys.path.insert(0, "/home/hatch/workspace")
import gh_api
import importlib.util

LOOP = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/s29-snaploop-fableg1.py"
spec = importlib.util.spec_from_file_location("snap_fg1", LOOP)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

MARKER = "S29MCOUNT" + time.strftime("%H%M%S")
TAGS = ["fabler1_m2", "fabler1_echo2", "fableg1_m2", "fableg1_echo2",
        "m7bsv2r1_m2", "m7bsv2r1_echo2", "m7bsv2g1_m2", "m7bsv2g1_echo2"]
FILES = " ".join(f"/root/carbon/data/bigeval_live/{t}.jsonl" for t in TAGS)

script = """MARK=__MARKER__
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 300 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
MARK=__MARKER__
TGT=47.117.170.99
KEY=/home/arm/.ssh/id_ed25519_carbon_relay
echo "$MARK DGX-BEGIN"
timeout -k 30 180 ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=20 root@"$TGT" 'bash -s' <<'TGTEOF'
echo "TGT-BEGIN"
date '+%F %T %Z'
echo "--- COUNTS ---"
wc -l __FILES__
echo "TGT-END"
TGTEOF
echo "TGT-RC=$?"
echo "$MARK DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
""".replace("__MARKER__", MARKER).replace("__FILES__", FILES)

t0 = time.time()
mod.dispatch(script)
print(f"dispatched marker={MARKER}", flush=True)
log = mod.claim(MARKER, not_before=t0)
if log is None:
    print("NOT-CLAIMED")
else:
    i = log.find("--- COUNTS ---")
    print(log[i:i + 1500] if i >= 0 else log[-1500:])
