#!/usr/bin/env python3
"""Wait for MAIN arm's first FORMAL train.jsonl line (warm == false) and print
it (achieved KL of round 1). Control arm already has formal=1, so the m line
is checked on its own; the shared first_round_watch.py would fire at once.
"""
import re, subprocess, time

SSH = ["ssh", "-i", "/home/hatch/.ssh/id_ed25519", "-o", "IdentitiesOnly=yes",
       "-o", "UserKnownHostsFile=/home/hatch/.ssh/known_hosts",
       "-o", "ProxyCommand=nc -X connect -x hatch-egress-proxy:3130 %h %p",
       "root@100.79.251.87"]


def run():
    try:
        r = subprocess.run(SSH + ["python3", "/root/carbon/remote_round_check.py"],
                           capture_output=True, text=True, timeout=45)
        if r.returncode != 0:
            return f"ERR rc={r.returncode} {r.stderr.strip()[:200]}"
        return r.stdout.strip()
    except Exception as e:
        return f"ERR {e}"


for i in range(1, 241):
    out = run()
    print(f"poll {i} {time.strftime('%H:%M:%S')}: {out}", flush=True)
    m_line = [l for l in out.splitlines() if l.startswith("m ")]
    if m_line and re.search(r"formal=[1-9]", m_line[0]) and "ERR" not in out:
        print("FIRST_FORMAL_ROUND_M_SEEN", flush=True)
        break
    time.sleep(60)
