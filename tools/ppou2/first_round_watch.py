#!/usr/bin/env python3
"""Wait for the first FORMAL train.jsonl line (warm == false) per arm and
print it (achieved KL of round 1). Warmup lines (warm == true) do not count."""
import json, subprocess, time

SSH = ["ssh", "-i", "/home/hatch/.ssh/id_ed25519", "-o", "IdentitiesOnly=yes",
       "-o", "UserKnownHostsFile=/home/hatch/.ssh/known_hosts",
       "-o", "ProxyCommand=nc -X connect -x hatch-egress-proxy:3130 %h %p",
       "root@100.79.251.87"]
def run():
    try:
        r = subprocess.run(SSH + ["python3", "/root/carbon/remote_round_check.py"], capture_output=True, text=True, timeout=45)
        if r.returncode != 0:
            return f"ERR rc={r.returncode} {r.stderr.strip()[:200]}"
        return r.stdout.strip()
    except Exception as e:
        return f"ERR {e}"

for i in range(1, 241):
    out = run()
    print(f"poll {i} {time.strftime('%H:%M:%S')}: {out}", flush=True)
    import re
    if re.search(r"formal=[1-9]", out) and "ERR" not in out:
        print("FIRST_FORMAL_ROUND_SEEN", flush=True)
        break
    time.sleep(60)
