#!/bin/bash
# Hakimi v2 pro replica dual-arm launch on Studio, TRIP 5b (retry of
# trip-5 Stage 2 inside the same trip budget, dispatch 2 of 3).
# Trip-5 first run (run 43291431703) failed LOUDLY at rsync-matches
# rc=12: the prescribed rsync transport "ssh -n ..." is itself broken --
# ssh -n redirects ssh's stdin from /dev/null, but under rsync that
# stdin IS rsync's protocol pipe, so the transfer collapsed with
# "connection unexpectedly closed (0 bytes received)". Evidence the
# plain ssh -n fixes are correct: both DGX ssh -n calls ran fine and
# the script survived past them (die + STUDIO-END executed). rsync's
# ssh child always gets rsync's own pipes, never the script stdin, so
# rsync was never an eater -- in trip-4 the eaters were the two plain
# ssh calls only. 5b therefore reverts ONLY the two rsync -e strings
# to the proven trip-4 form; everything else is byte-identical to 5.
#
# Hakimi v2 pro replica dual-arm launch on Studio, TRIP 5.
# Trip-4 outcome: the macOS timeout fix worked and data sync completed
# (matches 100 + corpus 102 landed in Studio hakimi2_src, counts double-
# verified), but the Studio heredoc script was silently truncated right
# after SYNC-DONE and the run reported a FALSE success: training never
# started. Root cause: the Studio script is fed via stdin
# (ssh 'bash -s' <<heredoc); its INNER ssh/rsync commands default to
# reading stdin and swallowed the not-yet-executed remainder of the
# script (verify -> chain write -> launch -> STUDIO-END).
# Trip-5 changes vs trip-4 (ONLY these):
#   1) Every inner ssh in the Studio segment gets -n; rsync uses
#      -e "ssh -n -o BatchMode=yes -o ConnectTimeout=15" so its
#      transport cannot read stdin either. Chain/ship-side scp/ssh fix
#      choice = </dev/null: hakimi2_chain.sh, ship_hakimi2.sh and both
#      trainers are all launched with stdin redirected from /dev/null
#      (already in place in trip-4, kept and documented here), so no
#      script stream exists for them to eat.
#   2) The OUTER L20->Studio ssh 'bash -s' <<heredoc deliberately gets
#      NO -n -- that would starve the heredoc itself (reverse trap).
#   3) Driver-side canary (hakimi2-driver5.py): a run with conclusion
#      success but no STUDIO-END in the claimed log counts as FAILURE.
#   4) Markers renamed: HAKIMI2LCH5 / HAKIMI2CHN5 (probe HAKIMI2P5,
#      accept HAKIMI2AC5) to prevent trip-4 claim cross-talk.
# Data route + chain identical to trip-4 (Studio rsync workdir, no .git;
# DGX LAN pull source; replica_chain single-chain discipline).
# Channel: cloud-assistant.yml -> L20 -> ssh Studio. Marker: HAKIMI2LCH5
MARK=HAKIMI2LCH5
set -eu
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
SSHK="-i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112"
# --- precheck: repeat of the Stage 1 probe (channel may have changed) ---
PERR=/tmp/hakimi2lch5_pre.err
PRC=0
POUT=$(timeout -k 30 120 ssh $SSHK arm@10.211.43.68 'echo STUDIO-REACHABLE; hostname' 2>$PERR) || PRC=$?
if [ $PRC -ne 0 ] || ! echo "$POUT" | grep -q STUDIO-REACHABLE; then
  echo "$MARK FAIL precheck rc=$PRC"
  echo "--- precheck ssh stderr tail ---"
  tail -5 $PERR
  echo "$MARK L20-END"
  exit 1
fi
echo "PRECHECK OK: $POUT"
RC=0
timeout -k 30 420 ssh $SSHK arm@10.211.43.68 'bash -s' <<'STEOF' || RC=$?
set -eu
MARK=HAKIMI2LCH5
D=$HOME/junyang/carbon/data
DGX=arm@10.188.0.23
VENV=$HOME/junyang/carbon/.venv/bin/python
die() { echo "$MARK FAIL $1 rc=$2"; echo "$MARK STUDIO-END"; exit 1; }
echo "$MARK STUDIO-BEGIN $(date '+%F %T %Z')"
# --- self-sweep: leftover hakimi-line processes from earlier trips ---
# macOS has no /proc: verify each candidate PID's command line via ps
# before any kill. Kill ONLY stale hakimi2_chain.sh launchers; a live
# cnet2_train.py is never killed here (the chain step handles trainer-busy).
for pid in $(pgrep -f "hakimi2" 2>/dev/null || true); do
  if [ "$pid" = "$$" ]; then continue; fi
  cmd=$(ps -o command= -p "$pid" 2>/dev/null || true)
  echo "SWEEP pid=$pid cmd=$(echo "$cmd" | cut -c1-160)"
  case "$cmd" in
    *hakimi2_chain.sh*) echo "SWEEP-KILL stale chain pid=$pid"; kill "$pid" 2>/dev/null || true ;;
  esac
done
if [ ! -x "$VENV" ]; then die no-venv-python 1; fi
ls -l "$VENV"
echo '--- STUDIO-LAYOUT ---'
ls $HOME/junyang/ 2>/dev/null || true
# --- 0) find the data: Studio clone first, else sync from DGX ---
DATA=""
REPO=""
for c in $HOME/junyang/carbon $HOME/junyang/carbon-repo $HOME/junyang/carbon-league $HOME/carbon-league $HOME/carbon-repo; do
  if [ -d "$c/.git" ] && [ -f "$c/data/hakimi2_matches.json" ]; then REPO="$c"; DATA="$c/data"; echo "CLONE-FOUND $c"; break; fi
done
if [ -n "$REPO" ]; then
  echo "HEAD_BEFORE=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo UNKNOWN)"
  PULL_RC=0
  git -C "$REPO" pull --ff-only > $D/hakimi2_pull.log 2>&1 || PULL_RC=$?
  tail -3 $D/hakimi2_pull.log || true
  echo "PULL_RC=$PULL_RC HEAD_AFTER=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo UNKNOWN)"
  if [ "$PULL_RC" -ne 0 ]; then
    if git -C "$REPO" merge-base --is-ancestor be909e6c HEAD 2>/dev/null; then
      echo "PULL failed but local HEAD already contains be909e6c; proceeding on local files"
    else
      die git-pull $PULL_RC
    fi
  fi
else
  echo "NO-STUDIO-CLONE, syncing data from DGX over LAN"
  # TRIP-5 FIX: inner ssh MUST NOT read stdin (-n) -- in trip-4 these two
  # ssh calls and the rsync transports below ate the rest of this script.
  ssh -n -o BatchMode=yes -o ConnectTimeout=20 $DGX 'git -C /home/arm/junyang/carbon-repo pull --ff-only 2>&1 | tail -2; git -C /home/arm/junyang/carbon-repo rev-parse --short HEAD' || die dgx-pull-ssh $?
  ssh -n -o BatchMode=yes -o ConnectTimeout=20 $DGX "python3 -c 'import json; print(\"DGX-MATCHES\", len(json.load(open(\"/home/arm/junyang/carbon-repo/data/hakimi2_matches.json\"))))'; ls /home/arm/junyang/carbon-repo/data/hakimi2_corpus/*.json.gz | wc -l" || die dgx-counts $?
  mkdir -p $D/hakimi2_src
  rsync -avz --partial -e "ssh -o BatchMode=yes -o ConnectTimeout=20" $DGX:/home/arm/junyang/carbon-repo/data/hakimi2_matches.json $D/hakimi2_src/hakimi2_matches.json || die rsync-matches $?
  rsync -avz --partial -e "ssh -o BatchMode=yes -o ConnectTimeout=20" $DGX:/home/arm/junyang/carbon-repo/data/hakimi2_corpus/ $D/hakimi2_src/hakimi2_corpus/ || die rsync-corpus $?
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
VRC=0
$VENV $D/hakimi2_index.py verify "$DATA" > $D/hakimi2_verify.log 2>&1 || VRC=$?
cat $D/hakimi2_verify.log
if [ "$VRC" -ne 0 ]; then die verify $VRC; fi
# --- 3) write chain (build + train + ship, detached) ---
cat > $D/hakimi2_chain.sh <<'CHEOF'
#!/bin/bash
set -u
MARK=HAKIMI2CHN5
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
$VENV $D/hakimi2_index.py build "$DATA" "$D" || { echo "$MARK FAIL index-build rc=$?"; exit 1; }
cd $C || { echo "$MARK FAIL cd-code"; exit 1; }
export CNET2_TOP_TEACHERS="$TEACHER@2"
$VENV -c "import build_top, cnet2_build as B; g=B.scan(['$D/hakimi2_corpus']); print('SCAN games=', len(g), 'val=', sum(x['val'] for x in g))" 2>&1 | tail -3
BRC=0
$VENV build_top.py $D/ds_hakimi2 $D/hakimi2_corpus > $D/hakimi2_build.log 2>&1 || BRC=$?
echo "build_top rc=$BRC"
tail -3 $D/hakimi2_build.log
if [ "$BRC" -ne 0 ]; then echo "$MARK FAIL build_top rc=$BRC"; exit 1; fi
URC=0
$VENV set_uw.py $D/ds_hakimi2 1 1 1 >> $D/hakimi2_build.log 2>&1 || URC=$?
echo "set_uw rc=$URC"
tail -2 $D/hakimi2_build.log
if [ "$URC" -ne 0 ]; then echo "$MARK FAIL set_uw rc=$URC"; exit 1; fi
$VENV -c "import json; d=json.load(open('$D/ds_hakimi2/meta.json')); print('DS-META games=', len(d['games']), 'total=', d['total'], 'teachers=', d['teachers']); assert d['total'] > 0" || { echo "$MARK FAIL meta-read"; exit 1; }
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
# Trip-5 stdin discipline: this watcher is launched with < /dev/null, so
# the scp/ssh below can never swallow a script stream (chain-side fix
# choice = </dev/null, applied at both nohup launch points in the chain).
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
tail -3 $D/hakimi2r1.log 2>/dev/null
tail -3 $D/hakimi2g1.log 2>/dev/null
ls -lt $D/weights/ | grep -i hakimi2 | head -6
echo "$MARK CHAIN-ATTESTED $(date '+%F %T %Z')"
CHEOF
chmod +x $D/hakimi2_chain.sh
nohup bash $D/hakimi2_chain.sh > $D/hakimi2_chain_outer.log 2>&1 < /dev/null &
CPID=$!
echo "CHAIN_PID=$CPID"
sleep 120
echo '--- CHAIN-ATTEST +120s ---'
ps -o pid,etime,command -p $CPID 2>/dev/null | cut -c1-200 || echo CHAIN-PROC-GONE
pgrep -fl "cnet2_train.py" | cut -c1-200 || echo NO-TRAIN-PROC-YET
tail -15 $D/hakimi2_chain.log 2>/dev/null || echo NO-CHAIN-LOG
ls -ld $D/ds_hakimi2 $D/hakimi2_corpus 2>/dev/null || true
echo "$MARK STUDIO-END $(date '+%F %T %Z')"
STEOF
echo "$MARK L20-END rc=$RC"
if [ $RC -ne 0 ]; then
  echo "$MARK FAIL ssh-leg rc=$RC"
  exit 1
fi
exit 0
