#!/bin/bash
# spot25 step F5: find the missing file - read sources + in-process repro with traceback.
MARK="SPOT25F1K5"
echo "MARKER $MARK START"
date '+%F %T %Z'
echo "--- BIGEVAL FULL ---"
cat /root/carbon/code/bigeval.py
echo "--- SEARCH OPEN/LOAD REFS ---"
grep -nE "open\(|np\.load|load\(|os\.path\.join" /root/carbon/code/cnet2_search.py | head -30
echo "--- IN-PROCESS REPRO ---"
cd /root/carbon/code && timeout 120 ../.venv/bin/python - <<'PYEOF' 2>&1 | tail -30
import traceback, sys
sys.path.insert(0, '/root/carbon/code')
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location('m3s2r1', '/root/carbon/code/exp/m3s2r1.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    obs = {"step": 0}
    print("agent-callable:", callable(m.agent))
    # try building the underlying search agent directly
    import cnet2_search as S
    ag = S.make_search_agent(m.CONF, weights=m._W)
    print("SEARCH-AGENT-BUILT", type(ag))
except Exception:
    traceback.print_exc()
PYEOF
echo "MARKER $MARK END"
