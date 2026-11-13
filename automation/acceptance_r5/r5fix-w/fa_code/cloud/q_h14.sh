#!/bin/bash
# host 14: finish the search-vs-pure exam (first 100 seeds per group, resumed), then car>2nd forks (d9d0d0a)
cd /root/carbon/code
until [ -f exp/fx_m2t3S20.py ]; do sleep 10; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# --- F07 fix (REVIEW_FULL_20261113): run identity + cumulative full-set health ---
# The N02 fix verified only the INCREMENTAL batch: miss() treated any seed
# present in bigeval_live as done (including previously FAILED seeds), the
# health check ignored actor/opponent identity and duplicate seeds, a missing
# summary with missing output warned and passed (fail-open), and a failure
# never revoked a previous run's DONE marker (sx.fail and a stale sx.done
# could coexist). Now:
#  * Every exam group has a run_id: a fingerprint of tag + actor + opponent +
#    actor/opponent file md5 (weights/wrapper identity) + bigeval.py md5
#    (harness/rules identity) + the expected seed list. DONE/FAIL markers
#    carry the run_id; at group start, markers not carrying the current
#    run_id are archived to <marker>.stale -- consumers must only trust a
#    DONE whose run_id matches the current run manifest.
#  * Completion is judged on the CUMULATIVE full expected set: an expected
#    seed is healthy only if a bigeval summary with matching a/b identity and
#    FAIL=false covers it, or its latest live row has err=null, a margin and
#    serr=0. Previously failed seeds are re-run, never skipped. Only the
#    unhealthy/missing seeds are launched (frozen once, N = their count).
#  * ALREADY_COMPLETE (Codex ruling, REVIEW_FULL F07): if the whole expected
#    set already exists, is unique and healthy under the SAME run identity,
#    the group succeeds without launching and writes DONE with the run_id.
#    Seed presence alone never counts as complete.
#  * No summary AND no output is a failure (fail-closed). DONE and FAIL
#    markers never coexist: writing one removes the other.
count_seeds() { if [ -z "$1" ]; then echo 0; else awk -F, '{print NF}' <<< "$1"; fi; }
f07_py() { # subcommands: rid / grid / missing / cumcheck -- see python below
  python3 - "$@" <<'PYEOF'
import hashlib, json, os, sys

def md5_of(p):
    try:
        with open(p, 'rb') as fh:
            return hashlib.md5(fh.read()).hexdigest()
    except Exception:
        return 'nofile:' + str(p)

def same_id(x, y):
    if x is None or y is None:
        return False
    return str(x) == str(y) or os.path.basename(str(x)) == os.path.basename(str(y))

def healthy_seeds(tag, a, b):
    hs = set()
    p = os.path.join('..', 'data', 'bigeval.jsonl')
    if os.path.exists(p):
        for line in open(p):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get('tag') != tag:
                continue
            if not same_id(d.get('a'), a) or not same_id(d.get('b'), b):
                continue
            if d.get('FAIL') is False:
                for g in d.get('games') or []:
                    if g and g[0] is not None:
                        hs.add(int(g[0]))
    lp = os.path.join('..', 'data', 'bigeval_live', tag + '.jsonl')
    latest = {}
    if os.path.exists(lp):
        for line in open(lp):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get('seed') is not None:
                latest[int(d['seed'])] = d
    for s, d in latest.items():
        if d.get('err') is None and d.get('margin') is not None and not (d.get('serr') or 0):
            hs.add(s)
    return hs

def expected_list(csv):
    exp = [int(x) for x in csv.split(',') if x.strip() != '']
    if len(set(exp)) != len(exp):
        print('F07: duplicate seeds in expected list -- refusing', file=sys.stderr)
        sys.exit(3)
    return exp

cmd = sys.argv[1]
if cmd == 'rid':
    tag, a, b, seeds = sys.argv[2:6]
    payload = json.dumps([tag, a, b, md5_of(a), md5_of(b), md5_of('bigeval.py'),
                          hashlib.sha256(seeds.encode()).hexdigest()], sort_keys=True)
    print(hashlib.sha256(payload.encode()).hexdigest()[:16])
elif cmd == 'grid':
    print(hashlib.sha256((sys.argv[2] + '|' + md5_of('bigeval.py')).encode()).hexdigest()[:16])
elif cmd in ('missing', 'cumcheck'):
    tag, a, b, seeds = sys.argv[2:6]
    exp = expected_list(seeds)
    hs = healthy_seeds(tag, a, b)
    miss = [s for s in exp if s not in hs]
    if cmd == 'missing':
        print(','.join(str(s) for s in miss))
    else:
        if miss:
            print(f'F07: tag={tag} cumulative set incomplete: {len(miss)}/{len(exp)} seeds missing-or-unhealthy: {miss[:12]}')
            sys.exit(1)
        print(f'F07: tag={tag} cumulative full set healthy ({len(exp)} seeds)')
        sys.exit(0)
else:
    print('unknown f07_py command', file=sys.stderr); sys.exit(2)
PYEOF
}
f07_rid() { f07_py rid "$1" "$2" "$3" "$4"; }
f07_missing() { f07_py missing "$1" "$2" "$3" "$4"; }
f07_cum_ok() { f07_py cumcheck "$1" "$2" "$3" "$4"; }
f07_archive_stale() { # $1=marker_path $2=run_id : archive marker not carrying this run_id
  if [ -f "$1" ]; then
    if grep -q "^run_id=$2\$" "$1"; then :; else mv -f "$1" "$1.stale"; echo "F07: archived stale marker $1 (run identity mismatch)"; fi
  fi
}
f07_write_done() { # $1=done_path $2=fail_path $3=run_id $4=label
  printf '%s\nrun_id=%s\ntime=%s\n' "$4" "$3" "$(date '+%F %T')" > "$1"
  rm -f "$2"
}
f07_write_fail() { # $1=fail_path $2=done_path $3=run_id
  printf 'FAIL\nrun_id=%s\ntime=%s\n' "$3" "$(date '+%F %T')" > "$1"
  rm -f "$2"
}
check_bigeval_health() { # $1=tag $2=want_n $3=out_file $4=actor $5=opponent
  python3 - "$1" "$2" "$3" "$4" "$5" <<'PYEOF'
import json, os, sys
tag, want, out, a, b = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
def same_id(x, y):
    if x is None or y is None:
        return False
    return str(x) == str(y) or os.path.basename(str(x)) == os.path.basename(str(y))
rec = None
p = os.path.join("..", "data", "bigeval.jsonl")
if os.path.exists(p):
    for line in open(p):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("tag") == tag:
            rec = d
if rec is not None:
    if rec.get("FAIL") is not False or rec.get("n") != want:
        print(f"HEALTH-FAIL tag={tag} FAIL={rec.get('FAIL')} n={rec.get('n')} want={want}")
        sys.exit(1)
    if not same_id(rec.get("a"), a) or not same_id(rec.get("b"), b):
        print(f"HEALTH-FAIL tag={tag} identity mismatch: summary a={rec.get('a')} b={rec.get('b')} want a={a} b={b}")
        sys.exit(1)
    sys.exit(0)
try:
    txt = open(out).read() if out and os.path.exists(out) else ""
except Exception:
    txt = ""
if not txt:
    print(f"HEALTH-FAIL tag={tag}: no bigeval summary record and no output file -- failing closed (F07)")
    sys.exit(1)
if '"FAIL": true' in txt or '"ERROR"' in txt or "FAIL_SEARCH_ERRORS" in txt:
    print(f"HEALTH-FAIL tag={tag} (out file reports failure)")
    sys.exit(1)
print(f"HEALTH-WARN tag={tag}: no bigeval summary record found; relying on exit code only")
sys.exit(0)
PYEOF
}
collect_group() { # arrays: pids tags wants outs actors opponents -> COLLECT_RC
  COLLECT_RC=0
  local _pa=$1 _ta=$2 _wa=$3 _oa=$4 _aa=$5 _ba=$6 _n _i _pid _tag _want _out _a _b _rc
  eval "_n=\${#$_pa[@]}"
  for ((_i = 0; _i < _n; _i++)); do
    eval "_pid=\${$_pa[$_i]}; _tag=\${$_ta[$_i]}; _want=\${$_wa[$_i]}; _out=\${$_oa[$_i]}; _a=\${$_aa[$_i]}; _b=\${$_ba[$_i]}"
    if wait "$_pid"; then _rc=0; else _rc=$?; fi
    if [ "$_rc" -ne 0 ]; then
      echo "FAIL tag=$_tag pid=$_pid exit=$_rc"
      COLLECT_RC=1
    elif [ -n "$_want" ] && ! check_bigeval_health "$_tag" "$_want" "$_out" "$_a" "$_b"; then
      echo "FAIL tag=$_tag (bigeval summary health check)"
      COLLECT_RC=1
    fi
  done
}

# ---- SX group: search-vs-pure exam, expected sets 994000..994099 / 995000..995099 ----
SX_A="exp/fx_m2t3S20.py"
SX_TAGS=(); SX_ACTORS=(); SX_OPPS=(); SX_EXPS=(); SX_ENVS=()
SX_TAGS+=("sx_S20_cnet2_m2s2stand"); SX_ACTORS+=("$SX_A"); SX_OPPS+=("exp/cnet2_m2s2stand.py"); SX_EXPS+=("$(seq -s, 994000 994099)"); SX_ENVS+=("")
SX_TAGS+=("sx_S20_cnet2_m2so");     SX_ACTORS+=("$SX_A"); SX_OPPS+=("exp/cnet2_m2so.py");     SX_EXPS+=("$(seq -s, 994000 994099)"); SX_ENVS+=("")
SX_TAGS+=("sx_S20_vs_pure");        SX_ACTORS+=("$SX_A"); SX_OPPS+=("awr_us_env.py");         SX_EXPS+=("$(seq -s, 995000 995099)"); SX_ENVS+=("pure")
SX_SPEC=""
for i in 0 1 2; do SX_SPEC="$SX_SPEC${SX_TAGS[$i]}|${SX_ACTORS[$i]}|${SX_OPPS[$i]}|$(f07_rid "${SX_TAGS[$i]}" "${SX_ACTORS[$i]}" "${SX_OPPS[$i]}" "${SX_EXPS[$i]}");"; done
SX_RUN_ID=$(f07_py grid "$SX_SPEC")
echo "F07: SX group run_id=$SX_RUN_ID"
f07_archive_stale ../data/sx.done "$SX_RUN_ID"
f07_archive_stale ../data/sx.fail "$SX_RUN_ID"
SX_PIDS=(); SX_LTAGS=(); SX_WANTS=(); SX_OUTS=(); SX_LACTORS=(); SX_LOPPS=()
SX_RC=0; SX_TODO=0
for i in 0 1 2; do
  TAG="${SX_TAGS[$i]}"; A="${SX_ACTORS[$i]}"; B="${SX_OPPS[$i]}"; EXP="${SX_EXPS[$i]}"
  MISS=$(f07_missing "$TAG" "$A" "$B" "$EXP") || { echo "F07: expected seed set invalid for $TAG"; SX_RC=1; continue; }
  if [ -z "$MISS" ]; then
    echo "F07: ALREADY_COMPLETE $TAG (full expected set healthy under this run identity)"
    continue
  fi
  N=$(count_seeds "$MISS"); SX_TODO=1
  echo "F07: $TAG launching $N missing-or-unhealthy seeds (previously failed seeds are re-run, not skipped)"
  if [ "${SX_ENVS[$i]}" = "pure" ]; then
    EVO_SEEDS="$MISS" AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_b128m2t3.npz AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 7, "LATE_STEP": 240}' EVO_WORKERS=21 \
      setsid nohup ../.venv/bin/python bigeval.py "$A" "$B" 0 "$N" "$TAG" > ../data/$TAG.out 2>&1 < /dev/null &
  else
    EVO_SEEDS="$MISS" EVO_WORKERS=21 setsid nohup ../.venv/bin/python bigeval.py "$A" "$B" 0 "$N" "$TAG" > ../data/$TAG.out 2>&1 < /dev/null &
  fi
  SX_PIDS+=($!); SX_LTAGS+=("$TAG"); SX_WANTS+=("$N"); SX_OUTS+=("../data/$TAG.out"); SX_LACTORS+=("$A"); SX_LOPPS+=("$B")
done
if [ "$SX_TODO" -eq 1 ]; then
  collect_group SX_PIDS SX_LTAGS SX_WANTS SX_OUTS SX_LACTORS SX_LOPPS
  if [ "$COLLECT_RC" -ne 0 ]; then SX_RC=1; fi
fi
if [ "$SX_RC" -eq 0 ]; then
  for i in 0 1 2; do
    if ! f07_cum_ok "${SX_TAGS[$i]}" "${SX_ACTORS[$i]}" "${SX_OPPS[$i]}" "${SX_EXPS[$i]}"; then SX_RC=1; fi
  done
fi
if [ "$SX_RC" -ne 0 ]; then
  f07_write_fail ../data/sx.fail ../data/sx.done "$SX_RUN_ID"
  echo "F07: search-vs-pure exam FAILED -- SX_DONE not written, fork stage not started"
  exit 1
fi
f07_write_done ../data/sx.done ../data/sx.fail "$SX_RUN_ID" SX_DONE
# ---- fork stage: car>2nd forks (no bigeval seeds; run identity from the fork spec) ----
FORK_RUN_ID=$(f07_py grid "fork|car>2nd|9200|100|62|cnet2_m2s2stand,cnet2_m2so,s1_net")
echo "F07: fork group run_id=$FORK_RUN_ID"
f07_archive_stale ../data/fkcar.done "$FORK_RUN_ID"
f07_archive_stale ../data/fkcar.fail "$FORK_RUN_ID"
export AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_b128m2t3.npz AWR_TAU=0
export AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 7, "LATE_STEP": 240}' FORK_P=0.3 FORK_H=10,25,50 FORK_END=0.1 FORK_KINDS="car>2nd"
FORK_RC=0
for O in cnet2_m2s2stand cnet2_m2so s1_net; do
  FORK_TAG=car_$O ../.venv/bin/python fork_gen.py ../data/fkcar awr_us_env.py exp/$O.py 9200 100 62 > ../data/fkcar_$O.out 2>&1 || FORK_RC=1
done
if [ "$FORK_RC" -ne 0 ]; then
  f07_write_fail ../data/fkcar.fail ../data/fkcar.done "$FORK_RUN_ID"
  echo "F07: fork stage FAILED -- ALL_DONE not written"
  exit 1
fi
f07_write_done ../data/fkcar.done ../data/fkcar.fail "$FORK_RUN_ID" ALL_DONE
