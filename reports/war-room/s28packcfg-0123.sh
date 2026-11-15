#!/bin/bash
# READ-ONLY probe on spot-28: retrieve pack config files verbatim.
# Marker: S28PACKCFG0123
echo "S28PACKCFG0123_BEGIN"
date "+%F %T %Z"
hostname || true
echo "--- FIND-SCOPE ---"
echo "find root: /root/carbon"
echo "--- LS PACKS DIR ---"
ls -l /root/carbon/data/packs/ 2>&1 || echo "NO-PACKS-DIR /root/carbon/data/packs/"
echo "--- FIND n1r_int8.py ---"
find /root/carbon -name 'n1r_int8.py' 2>/dev/null || echo "FIND-N1R-FAILED"
echo "--- FIND cnet2_pack.py (preferred path first) ---"
ls -l /root/carbon/code/cnet2_pack.py 2>&1 || echo "NO /root/carbon/code/cnet2_pack.py"
find /root/carbon -name 'cnet2_pack.py' 2>/dev/null || echo "FIND-PACK-FAILED"
echo "--- FILE1 n1r_int8.py FULL-OR-SECTIONS ---"
/root/carbon/.venv/bin/python - <<'PYEOF' 2>&1 || python3 - <<'PYEOF2'
import os, re
cands = []
for root, dirs, files in os.walk("/root/carbon"):
    if "n1r_int8.py" in files:
        cands.append(os.path.join(root, "n1r_int8.py"))
print("N1R_CANDIDATES:", cands)
if not cands:
    print("N1R_NOT_FOUND under /root/carbon")
else:
    p = cands[0]
    data = open(p, "rb").read()
    txt = data.decode("utf-8", errors="replace")
    lines = txt.splitlines()
    print(f"N1R_PATH: {p}")
    print(f"N1R_BYTES: {len(data)}")
    print(f"N1R_LINES: {len(lines)}")
    print("--- N1R _OPTS GREP -n ---")
    for i, ln in enumerate(lines, 1):
        if "_OPTS" in ln:
            print(f"{i}:{ln}")
    print("--- N1R make_agent GREP -n ---")
    for i, ln in enumerate(lines, 1):
        if "make_agent" in ln or "def main" in ln:
            print(f"{i}:{ln}")
    if len(data) < 30000:
        print("--- N1R_FULL_BEGIN ---")
        print(txt)
        print("--- N1R_FULL_END ---")
    else:
        print("--- N1R_SECTIONS (file >=30KB, key lines + context) ---")
        keys = ("CONF","TEACHER","GUARD","nocollide","endgame","SEARCH","MODE","_OPTS","make_agent","shield","pshield","rguard")
        shown = set()
        for i, ln in enumerate(lines):
            if any(k in ln for k in keys):
                for j in range(max(0,i-3), min(len(lines), i+8)):
                    if j not in shown:
                        shown.add(j)
                        print(f"{j+1}:{lines[j]}")
        print("--- N1R_PYTHON _OPTS EXTRACT ---")
        m = re.search(r"_OPTS\s*=\s*\{", txt)
        if m:
            start = m.start()
            depth = 0; end = None
            for k in range(m.end()-1, len(txt)):
                if txt[k] == "{": depth += 1
                elif txt[k] == "}":
                    depth -= 1
                    if depth == 0:
                        end = k+1; break
            print(txt[start:end] if end else txt[start:start+3000])
        else:
            print("NO _OPTS assignment found")
PYEOF
PYEOF2
echo "--- FILE2 cnet2_pack.py SECTIONS ---"
/root/carbon/.venv/bin/python - <<'PYEOF' 2>&1 || python3 - <<'PYEOF2'
import os, re
p = "/root/carbon/code/cnet2_pack.py"
cands = []
for root, dirs, files in os.walk("/root/carbon"):
    if "cnet2_pack.py" in files:
        cands.append(os.path.join(root, "cnet2_pack.py"))
print("PACK_CANDIDATES:", cands)
if os.path.exists(p):
    chosen = p
elif cands:
    chosen = cands[0]
else:
    chosen = None
    print("PACK_NOT_FOUND under /root/carbon")
if chosen:
    data = open(chosen, "rb").read()
    txt = data.decode("utf-8", errors="replace")
    lines = txt.splitlines()
    print(f"PACK_PATH: {chosen}")
    print(f"PACK_BYTES: {len(data)}")
    print(f"PACK_LINES: {len(lines)}")
    print("--- PACK SIGNATURES grep def main / make_agent ---")
    for i, ln in enumerate(lines, 1):
        if re.search(r"def (main|make_agent)\b", ln):
            # print def line plus following continuation lines until closing paren
            print(f"{i}:{ln}")
            j = i
            while (":" not in lines[j-1] or lines[j-1].count("(") > lines[j-1].count(")")) and j < len(lines):
                print(f"{j+1}:{lines[j]}")
                j += 1
                if lines[j-1].rstrip().endswith(":"):
                    break
    print("--- PACK add_argument ALL ---")
    for i, ln in enumerate(lines, 1):
        if "add_argument" in ln:
            print(f"{i}:{ln}")
    print("--- PACK DEFAULT/CONF/_OPTS/teacher/shield/nocollide/endgame/pshield/rguard LINES ---")
    keys = ("DEFAULT","CONF","_OPTS","teacher","shield","nocollide","endgame","pshield","rguard","TEACHER","GUARD","SEARCH","MODE")
    for i, ln in enumerate(lines, 1):
        if any(k in ln for k in keys):
            print(f"{i}:{ln}")
    if len(data) < 30000:
        print("--- PACK_FULL_BEGIN ---")
        print(txt)
        print("--- PACK_FULL_END ---")
    else:
        print("--- PACK_SECTIONS_NOTE file >=30KB, sections above only ---")
PYEOF
PYEOF2
echo "S28PACKCFG0123_END"
