#!/bin/bash
# F07 key items against the real helpers extracted from code/cloud/q_h15.sh.
set -u
CODE="$HOME/workspace/final-accept/code"
T=$(mktemp -d)
mkdir -p "$T/code" "$T/data/bigeval_live"
cp "$CODE/bigeval.py" "$T/code/bigeval.py"
echo "# actor v1" > "$T/code/actor.py"; echo "# opp v1" > "$T/code/opp.py"
# extract F07 helper block: from count_seeds() to just before collect_group()
awk '/^count_seeds\(\)/,/^collect_group\(\)/' "$CODE/cloud/q_h15.sh" | sed '$d' > "$T/helpers.sh"
cd "$T/code" || exit 1
source "$T/helpers.sh"
PASS=0; FAIL=0
chk() { if [ "$1" = "$2" ]; then echo "PASS $3"; PASS=$((PASS+1)); else echo "FAIL $3 (got [$1] want [$2])"; FAIL=$((FAIL+1)); fi }

TAG=T1; A=actor.py; B=opp.py; SEEDS="1,2,3"
RID=$(f07_rid "$TAG" "$A" "$B" "$SEEDS")
# 1) old failed seed reruns alone: summary FAIL=false covers 1,3 ; live row seed2 has err
cat > ../data/bigeval.jsonl <<EOF
{"tag": "T1", "FAIL": false, "n": 2, "a": "actor.py", "b": "opp.py", "games": [[1, 10, 2], [3, 5, 2]]}
EOF
echo '{"seed": 2, "err": "boom", "margin": null, "serr": 0}' > ../data/bigeval_live/T1.jsonl
MISS=$(f07_missing "$TAG" "$A" "$B" "$SEEDS")
chk "$MISS" "2" "old failed seed is the only rerun"
# 2) seed2 healthy via live row -> full set healthy
echo '{"seed": 2, "err": null, "margin": 12.5, "serr": 0}' > ../data/bigeval_live/T1.jsonl
MISS=$(f07_missing "$TAG" "$A" "$B" "$SEEDS")
chk "$MISS" "" "full set healthy -> nothing missing (ALREADY_COMPLETE gate)"
f07_cum_ok "$TAG" "$A" "$B" "$SEEDS" >/dev/null 2>&1; chk "$?" "0" "cumcheck exit 0 on healthy full set"
# 3) cross-run stale DONE archived; DONE/FAIL never coexist
printf 'DONE\nrun_id=deadbeefdeadbeef\n' > ../data/g.done
f07_archive_stale ../data/g.done "$RID" >/dev/null
[ -f ../data/g.done.stale ] && [ ! -f ../data/g.done ]; chk "$?" "0" "stale DONE from another run_id archived"
printf 'DONE\nrun_id=%s\n' "$RID" > ../data/g.done
f07_archive_stale ../data/g.done "$RID" >/dev/null
[ -f ../data/g.done ]; chk "$?" "0" "current run_id DONE kept"
f07_write_fail ../data/g.fail ../data/g.done "$RID"
{ [ -f ../data/g.fail ] && [ ! -f ../data/g.done ]; }; chk "$?" "0" "write FAIL removes DONE (never coexist)"
f07_write_done ../data/g.done ../data/g.fail "$RID" "DONE"
{ [ -f ../data/g.done ] && [ ! -f ../data/g.fail ]; }; chk "$?" "0" "write DONE removes FAIL"
# 4) duplicate expected seeds refused (exit 3 in helper)
f07_missing "$TAG" "$A" "$B" "1,2,2" >/dev/null 2>&1; chk "$?" "3" "duplicate expected seeds refused exit3"
# 5) identity change changes run_id (weights/wrapper identity binding)
echo "# actor v2" > actor.py
RID2=$(f07_rid "$TAG" "$A" "$B" "$SEEDS")
[ "$RID2" != "$RID" ]; chk "$?" "0" "actor file change -> new run_id"
echo "== F07 helpers: $PASS passed, $FAIL failed =="
[ "$FAIL" = 0 ]
