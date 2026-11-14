#!/usr/bin/env python3
"""Capture each arm's anchor state: state_last.pt written at the END of
warmup iteration 3 (vwarm=3), before the first formal round. Trigger = a
train.jsonl line with it==3 and warm==true. Then verify actor vs init
equality and record md5 + the trainer-recorded state sha256."""
import json, subprocess, sys, time

SSH = ["ssh", "-i", "/home/hatch/.ssh/id_ed25519", "-o", "IdentitiesOnly=yes",
       "-o", "UserKnownHostsFile=/home/hatch/.ssh/known_hosts",
       "-o", "ProxyCommand=nc -X connect -x hatch-egress-proxy:3130 %h %p",
       "root@100.79.251.87"]
def run(args):
    try:
        r = subprocess.run(SSH + args, capture_output=True, text=True, timeout=45)
        if r.returncode != 0:
            return f"ERR rc={r.returncode} {r.stderr.strip()[:200]}"
        return r.stdout.strip()
    except Exception as e:
        return f"ERR {e}"

for i in range(1, 241):
    out = run(["python3", "/root/carbon/remote_anchor_check.py"])
    print(f"poll {i} {time.strftime('%H:%M:%S')}: {out}", flush=True)
    if "WARM3" in out:
        print(run(["python3", "/root/carbon/remote_anchor_check.py", "capture"]), flush=True)
        out2 = run(["python3", "/root/carbon/remote_anchor_check.py"])
        print(f"after capture: {out2}", flush=True)
        # c is the live arm; its anchor capture ends this watch (m's anchor
        # is moot until a relaunch decision lands).
        c_line = [l for l in out2.splitlines() if l.startswith("c ")]
        if c_line and "anchor_done" in c_line[0]:
            print("ANCHOR_DONE_C", flush=True)
            break
    time.sleep(30)
