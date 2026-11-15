import sys, base64, os
# usage: m320_wrap.py <inner_dgx_sh> <marker> <ssh_timeout_s> <out_wrapper>
inner, mark, tmo, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
body = open(inner).read()
b64 = base64.b64encode(body.encode()).decode()
wrap = f"""#!/bin/bash
MARK={mark}
echo "${{MARK}}_BEGIN $(date '+%F %T %Z') host=$(hostname)"
echo '{b64}' | base64 -d > /tmp/m320_inner_{mark}.sh
wc -c /tmp/m320_inner_{mark}.sh
timeout -k 30 {tmo} ssh -i ~/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' < /tmp/m320_inner_{mark}.sh 2>&1
echo "ssh-rc=$?"
echo "${{MARK}}_END"
"""
assert mark in wrap
open(out, "w").write(wrap)
print("wrapper written", out, len(wrap), "bytes")
