#!/bin/bash
# Hakimi v2 pro replica dual-arm launch on Studio, TRIP 2 (v2 script).
# v1 failure (claimed run 43269357082, 01:26:42 CST): FAIL no-repo-clone --
# Studio $HOME/junyang/carbon is a synced workdir, NOT a git clone (no .git),
# and $HOME/junyang/carbon-repo does not exist either. v1 assumed a clone.
# v2: look for a Studio clone in several places; if none, pull on DGX over LAN
# and rsync data/hakimi2_matches.json + data/hakimi2_corpus/ to Studio, then
# verify/build against the synced copies. Chain + ship logic unchanged from v1.
# Channel: cloud-assistant.yml -> L20 -> ssh Studio. Marker: HAKIMI2LCH2
MARK=HAKIMI2LCH2
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
set -u
MARK=HAKIMI2LCH2
D=$HOME/junyang/carbon/data
DGX=arm@10.188.0.23
VENV=$HOME/junyang/carbon/.venv/bin/python
echo "$MARK STUDIO-BEGIN $(date '+%F %T %Z')"
ls -l "$VENV" || { echo "$MARK FAIL no-venv-python"; echo "$MARK STUDIO-END"; exit 0; }
echo '--- STUDIO-LAYOUT ---'
ls $HOME/junyang/ 2>/dev/null
# --- 0) find the data: Studio clone first, else sync from DGX ---
DATA=""
REPO=""
for c in $HOME/junyang/carbon $HOME/junyang/carbon-repo $HOME/junyang/carbon-league $HOME/carbon-league $HOME/carbon-repo; do
  if [ -d "$c/.git" ] && [ -f "$c/data/hakimi2_matches.json" ]; then REPO="$c"; DATA="$c/data"; echo "CLONE-FOUND $c"; break; fi
done
if [ -n "$REPO" ]; then
  echo "HEAD_BEFORE=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo UNKNOWN)"
  timeout 120 git -C "$REPO" pull --ff-only > $D/hakimi2_pull.log 2>&1; PULL_RC=$?
  tail -3 $D/hakimi2_pull.log
  echo "PULL_RC=$PULL_RC HEAD_AFTER=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo UNKNOWN)"
  if [ "$PULL_RC" -ne 0 ]; then
    if git -C "$REPO" merge-base --is-ancestor be909e6c HEAD 2>/dev/null; then
      echo "PULL failed but local HEAD already contains be909e6c; proceeding on local files"
    else
      echo "$MARK FAIL git-pull rc=$PULL_RC and HEAD lacks be909e6c"; echo "$MARK STUDIO-END"; exit 0
    fi
  fi
else
  echo "NO-STUDIO-CLONE, syncing data from DGX over LAN"
  timeout 120 ssh -o BatchMode=yes -o ConnectTimeout=20 $DGX 'git -C /home/arm/junyang/carbon-repo pull --ff-only 2>&1 | tail -2; git -C /home/arm/junyang/carbon-repo rev-parse --short HEAD' || { echo "$MARK FAIL dgx-pull-ssh"; echo "$MARK STUDIO-END"; exit 0; }
  timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=20 $DGX "python3 -c 'import json; print(\"DGX-MATCHES\", len(json.load(open(\"/home/arm/junyang/carbon-repo/data/hakimi2_matches.json\"))))'; ls /home/arm/junyang/carbon-repo/data/hakimi2_corpus/*.json.gz | wc -l" || { echo "$MARK FAIL dgx-counts (matches/corpus missing on DGX clone)"; echo "$MARK STUDIO-END"; exit 0; }
  mkdir -p $D/hakimi2_src
  timeout 300 rsync -avz --partial -e "ssh -o BatchMode=yes -o ConnectTimeout=20" $DGX:/home/arm/junyang/carbon-repo/data/hakimi2_matches.json $D/hakimi2_src/hakimi2_matches.json || { echo "$MARK FAIL rsync-matches"; echo "$MARK STUDIO-END"; exit 0; }
  timeout 300 rsync -avz --partial -e "ssh -o BatchMode=yes -o ConnectTimeout=20" $DGX:/home/arm/junyang/carbon-repo/data/hakimi2_corpus/ $D/hakimi2_src/hakimi2_corpus/ || { echo "$MARK FAIL rsync-corpus"; echo "$MARK STUDIO-END"; exit 0; }
  DATA="$D/hakimi2_src"
  echo "SYNC-DONE DATA=$DATA"
fi
# --- 1) write verify/build python (takes a DATA dir holding hakimi2_matches.json + hakimi2_corpus/) ---
cat > $D/hakimi2_index.py <<'PYEOF'
import gzip, json, os, shlex, shutil, sys
from collections import Counter

def load_matches(data):
    return json.load(open(os.path.join(data, "hakimi2_matches.json")))

def teacher_name(matches):
    c = Counter()
    for m in matches:
        c[m.get("name_a")] += 1
        c[m.get("name_b")] += 1
    n = len(matches)
    cands = [name for name, k in c.items() if k == n]
    if len(cands) == 1:
        return cands[0], c
    for name in cands:
        if "hakimi" in str(name).lower() or "哈基米" in str(name):
            return name, c
    return None, c

def verify(data):
    matches = load_matches(data)
    cdir = os.path.join(data, "hakimi2_corpus")
    files = [f for f in os.listdir(cdir) if os.path.isfile(os.path.join(cdir, f))]
    gz = [f for f in files if f.endswith(".json.gz")]
    teacher, counts = teacher_name(matches)
    print(f"VERIFY matches={len(matches)} corpus_files={len(files)} corpus_gz={len(gz)} teacher={teacher} top_names={counts.most_common(5)}")
    if len(files) != 102:
        print(f"VERIFY NOTE corpus_files={len(files)} != 102 (gz={len(gz)} is the operative count)")
    ok = len(matches) == 100 and len(gz) == 102 and teacher is not None
    if not ok:
        print(f"VERIFY FAIL need matches=100 corpus_gz=102 unique-teacher; got matches={len(matches)} corpus_gz={len(gz)} teacher={teacher}")
        return 1
    env = os.path.expanduser("~/junyang/carbon/data/hakimi2_env.sh")
    with open(env, "w") as f:
        f.write(f"DATA={shlex.quote(data)}\nTEACHER={shlex.quote(teacher)}\n")
    print(f"VERIFY GATE=PASS env={env}")
    return 0

def build(data, dest):
    matches = load_matches(data)
    teacher, _ = teacher_name(matches)
    assert teacher, "no teacher"
    cdir = os.path.join(data, "hakimi2_corpus")
    have = {}
    for f in os.listdir(cdir):
        if not f.endswith(".json.gz") or "-" not in f[:-len(".json.gz")]:
            continue
        base = f[:-len(".json.gz")]
        mid, g = base.rsplit("-", 1)
        try:
            have[(mid, int(g))] = f
        except ValueError:
            pass
    outdir = os.path.join(dest, "hakimi2_corpus")
    repdir = os.path.join(outdir, "replays")
    os.makedirs(repdir, exist_ok=True)
    entries, missing, corrupt, short = [], [], [], 0
    used = set()
    for m in matches:
        key = (m["id"], m["game"])
        fn = have.get(key)
        if fn is None:
            missing.append(f"{m['id']}-{m['game']}")
            continue
        used.add(key)
        src = os.path.join(cdir, fn)
        dst = os.path.join(repdir, fn)
        try:
            steps = len(json.load(gzip.open(src))["steps"]) - 1
        except Exception as e:
            corrupt.append(f"{fn}:{e!r}"[:160])
            continue
        if steps < 200:
            short += 1
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
        entries.append({"id": m["id"], "game": m["game"], "order": m["order"],
                        "scores": m["scores"], "winner": m.get("winner"), "steps": steps,
                        "file": f"replays/{fn}", "created": m.get("created"),
                        "strategies": {"a": {"id": "", "name": m["name_a"]},
                                       "b": {"id": "", "name": m["name_b"]}}})
    orphans = sorted(set(have) - used)
    print(f"THREE-COUNT matches={len(matches)} replays={len(entries)} index={len(entries)} orphans={len(orphans)} short_lt200={short} missing={missing[:5]} corrupt={corrupt[:3]}")
    if missing or corrupt:
        print("BUILD FAIL missing/corrupt non-empty")
        return 1
    json.dump(entries, open(os.path.join(outdir, "index.json"), "w"), ensure_ascii=False)
    print(f"BUILD OK index_entries={len(entries)} teacher={teacher} orphans={orphans[:5]}")
    return 0

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "verify":
        sys.exit(verify(sys.argv[2]))
    if mode == "build":
        sys.exit(build(sys.argv[2], sys.argv[3]))
    print("usage: verify DATA_DIR | build DATA_DIR DEST")
    sys.exit(2)
PYEOF
# --- 2) verify (hard gate) ---
$VENV $D/hakimi2_index.py verify "$DATA" > $D/hakimi2_verify.log 2>&1; VRC=$?
cat $D/hakimi2_verify.log
if [ "$VRC" -ne 0 ]; then echo "$MARK GATE FAIL verify-rc=$VRC"; echo "$MARK STUDIO-END"; exit 0; fi
# --- 3) write chain (build + train + ship, detached) ---
cat > $D/hakimi2_chain.sh <<'CHEOF'
#!/bin/bash
MARK=HAKIMI2CHN1
D=$HOME/junyang/carbon/data
C=$HOME/junyang/carbon/code
VENV=$HOME/junyang/carbon/.venv/bin/python
LOG=$D/hakimi2_chain.log
exec >> "$LOG" 2>&1
echo "$MARK CHAIN-BEGIN $(date '+%F %T %Z')"
source $D/hakimi2_env.sh
echo "DATA=$DATA TEACHER=$TEACHER"
for i in $(seq 1 12); do
  if ! pgrep -f "cnet2_train.py" >/dev/null; then echo "trainer-free at check $i"; break; fi
  echo "trainer still running, wait 10s ($i)"; sleep 10
done
if pgrep -f "cnet2_train.py" >/dev/null; then echo "$MARK FAIL trainer-busy"; pgrep -fl "cnet2_train.py"; exit 1; fi
if [ -d $D/ds_hakimi2 ]; then echo "$MARK FAIL ds_hakimi2-exists"; exit 1; fi
$VENV $D/hakimi2_index.py build "$DATA" "$D" || { echo "$MARK FAIL index-build"; exit 1; }
cd $C || { echo "$MARK FAIL cd-code"; exit 1; }
export CNET2_TOP_TEACHERS="$TEACHER@2"
$VENV -c "import build_top, cnet2_build as B; g=B.scan(['$D/hakimi2_corpus']); print('SCAN games=', len(g), 'val=', sum(x['val'] for x in g))" 2>&1 | tail -3
$VENV build_top.py $D/ds_hakimi2 $D/hakimi2_corpus > $D/hakimi2_build.log 2>&1; echo "build_top rc=$?"
tail -3 $D/hakimi2_build.log
$VENV set_uw.py $D/ds_hakimi2 1 1 1 >> $D/hakimi2_build.log 2>&1; echo "set_uw rc=$?"
tail -2 $D/hakimi2_build.log
$VENV -c "import json; d=json.load(open('$D/ds_hakimi2/meta.json')); print('DS-META games=', len(d['games']), 'total=', d['total'], 'teachers=', d['teachers'])"
for f in cnet2_arch_n1_last.npz cnet2_b128m2t3.npz; do
  if [ ! -f $D/weights/$f ]; then
    echo "MISSING $f, fetching from DGX"
    scp -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.188.0.23:/home/arm/junyang/carbon/data/weights/$f $D/weights/$f || { echo "$MARK FAIL fetch-init $f"; exit 1; }
  fi
done
ls -l $D/weights/cnet2_arch_n1_last.npz $D/weights/cnet2_b128m2t3.npz
export BC_DEVICE=mps BC_RAM=1 BC_THREADS=8 BC_USE_W=1 BC_USE_UW=1 BC_BS=256 BC_SAVE_EPOCHS=1 BC_PCT=0.1 BC_LR=3e-4
BC_INIT=$D/weights/cnet2_arch_n1_last.npz nohup $VENV cnet2_train.py $D/ds_hakimi2 $D/weights/cnet2_hakimi2r1.npz 20 '{"C":128,"NB":8,"D":128,"ZI":1,"TA":4,"TNH":4}' > $D/hakimi2r1.log 2>&1 < /dev/null &
echo $! > $D/hakimi2r1.pid; echo "R1_PID=$(cat $D/hakimi2r1.pid)"
BC_INIT=$D/weights/cnet2_b128m2t3.npz nohup $VENV cnet2_train.py $D/ds_hakimi2 $D/weights/cnet2_hakimi2g1.npz 20 '{"C":128,"NB":8,"D":128}' > $D/hakimi2g1.log 2>&1 < /dev/null &
echo $! > $D/hakimi2g1.pid; echo "G1_PID=$(cat $D/hakimi2g1.pid)"
cat > $D/ship_hakimi2.sh <<'SHIP_EOF'
#!/bin/bash
D=$HOME/junyang/carbon/data
LOG=$D/hakimi2_ship.log
for arm in r1 g1; do
  pid=$(cat $D/hakimi2${arm}.pid 2>/dev/null)
  if [ -n "$pid" ]; then while kill -0 "$pid" 2>/dev/null; do sleep 30; done; fi
  sleep 5
  f=$D/weights/cnet2_hakimi2${arm}_ep19.npz
  if [ -f "$f" ]; then
    LMD5=$(md5 -q "$f")
    scp -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new "$f" arm@10.188.0.23:/home/arm/junyang/carbon/data/weights/cnet2_hakimi2${arm}_fromstudio.npz
    RC=$?
    RMD5=$(ssh -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.188.0.23 "md5sum /home/arm/junyang/carbon/data/weights/cnet2_hakimi2${arm}_fromstudio.npz" | awk '{print $1}')
    if [ "$LMD5" = "$RMD5" ]; then M=YES; else M=NO; fi
    BEST=$(grep -h '^best' $D/hakimi2${arm}.log | tail -1)
    EP19=$(grep -h '"ep": 19' $D/hakimi2${arm}.log | tail -1)
    echo "$(date '+%F %T') shipped ${arm} scp_rc=$RC local_md5=$LMD5 dgx_md5=$RMD5 match=$M best=$BEST ep19=$EP19" >> "$LOG"
  else
    echo "$(date '+%F %T') EP19 MISSING ${arm} $f" >> "$LOG"
  fi
done
echo "$(date '+%F %T') HAKIMI2_SHIP_DONE" >> "$LOG"
SHIP_EOF
nohup bash $D/ship_hakimi2.sh > $D/hakimi2_ship.out 2>&1 < /dev/null &
echo "SHIP_WATCHER_PID=$!"
sleep 150
echo '--- CHAIN-ATTEST +150s ---'
pgrep -fl "cnet2_train.py" | cut -c1-200 || echo NO-TRAIN-PROC
tail -3 $D/hakimi2r1.log 2>/dev/null; tail -3 $D/hakimi2g1.log 2>/dev/null
ls -lt $D/weights/ | grep -i hakimi2 | head -6
echo "$MARK CHAIN-ATTESTED $(date '+%F %T %Z')"
CHEOF
chmod +x $D/hakimi2_chain.sh
nohup bash $D/hakimi2_chain.sh > $D/hakimi2_chain.log 2>&1 < /dev/null &
echo "CHAIN_PID=$!"
sleep 120
echo '--- CHAIN-ATTEST +120s ---'
ps -o pid,etime,command -p $! 2>/dev/null | cut -c1-200 || echo CHAIN-PROC-GONE
pgrep -fl "cnet2_train.py" | cut -c1-200 || echo NO-TRAIN-PROC-YET
tail -15 $D/hakimi2_chain.log 2>/dev/null || echo NO-CHAIN-LOG
ls -ld $D/ds_hakimi2 $D/hakimi2_corpus 2>/dev/null
echo "$MARK STUDIO-END $(date '+%F %T %Z')"
STEOF
echo "$MARK L20-END rc=$?"
