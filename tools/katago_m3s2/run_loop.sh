#!/usr/bin/env bash
# KataGo warm-start generation loop (plan stages 1-3):
#   vec_selfplay -> train -> pack_submission (payload swap + clock-removal
#   patch + pack verification) -> mirror eval vs master3s2 via the repo's
#   bigeval as an EXTERNAL command -> promotion gate -> genN/ archive.
#
# Timing is checked at exactly one place in the loop (owner ruling): the
# pack stage's verification (packed agent over full games, per-step
# max < 0.75 s).  The training side (selfplay/train) carries no time
# budget at all.  The promotion gate additionally requires the eval's
# reported max step time < 0.75 s.
#
# The evaluator is NOT reimplemented here.  BIGEVAL must point at the
# repo's bigeval entry; it is invoked as
#   $BIGEVAL --candidate FILE --opponent FILE --games N --out JSON
# and must write JSON with numeric keys "mean_margin" and
# "max_step_time".  Adjust BIGEVAL (and, if the repo's CLI differs, the
# invocation below) -- that is the single configurable seam.
set -euo pipefail

# --------------------------------------------------------------------------
# Configuration (override via environment)
# --------------------------------------------------------------------------
TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$(cd "$TOOLS_DIR/../.." && pwd)"
SRC_DIR="$BUILD_DIR/src"
CHASSIS="${CHASSIS:-$SRC_DIR/apexcarbon_v1_fix3.py}"   # owner order 2026-11-19 19:28: fix3 chassis (v1/fix2 alternates via $CHASSIS)
MASTER="${MASTER:-$SRC_DIR/master3s2.py}"
WORK="${WORK:-$BUILD_DIR/runs}"
GEN="${GEN:-1}"                                        # generation to produce
PY="${PY:-python3}"

GAMES="${GAMES:-200}"                                  # self-play games
SP_WORKERS="${SP_WORKERS:-8}"
DEEP_FRAC="${DEEP_FRAC:-0.25}"
EPOCHS="${EPOCHS:-3}"
DEVICE="${DEVICE:-cpu}"
DTYPE="${DTYPE:-bf16}"
HOF_CKPTS="${HOF_CKPTS:-}"                             # comma-separated npz list

FIXED_K="${FIXED_K:-12}"                               # pack: deterministic plan size
PACK_VERIFY_GAMES="${PACK_VERIFY_GAMES:-1}"

BIGEVAL="${BIGEVAL:-$BUILD_DIR/tools/bigeval.py}"
EVAL_GAMES="${EVAL_GAMES:-200}"
STEP_LIMIT="0.75"

GEN_DIR="$WORK/gen$GEN"
REPLAY_DIR="$GEN_DIR/replay"
CKPT="$GEN_DIR/ckpt.npz"
SUBMISSION="$GEN_DIR/submission.py"
EVAL_JSON="$GEN_DIR/eval.json"
BEST_FILE="$WORK/best_margin.txt"

# Init checkpoint: gen0 (from extract_m3s2.py) for generation 1, else the
# previous generation's trained checkpoint.  Value scale of the init
# checkpoint's value head (ruling RULING_KATAGO_VALUE_SCALE_BRIDGE_
# 20261119): gen0 is the native _terminal scale; trained checkpoints are
# on the plan's U scale and their self-play leaf values are bridged.
if [ "$GEN" -eq 1 ]; then
  INIT_CKPT="$WORK/gen0/gen0_m3s2.npz"
  SP_VALUE_SCALE="terminal"
  if [ ! -f "$INIT_CKPT" ]; then
    mkdir -p "$WORK/gen0"
    "$PY" "$TOOLS_DIR/extract_m3s2.py" --src "$MASTER" --out "$WORK/gen0" || true
  fi
else
  INIT_CKPT="$WORK/gen$((GEN - 1))/ckpt.npz"
  SP_VALUE_SCALE="u"
fi
[ -f "$INIT_CKPT" ] || { echo "init checkpoint missing: $INIT_CKPT" >&2; exit 1; }
mkdir -p "$REPLAY_DIR" "$WORK/promoted"

echo "== gen$GEN: selfplay ($GAMES games, init $INIT_CKPT) =="
"$PY" "$TOOLS_DIR/vec_selfplay.py" \
  --src "$MASTER" --ckpt "$INIT_CKPT" --hof "$HOF_CKPTS" \
  --games "$GAMES" --workers "$SP_WORKERS" --deep-frac "$DEEP_FRAC" \
  --value-scale "$SP_VALUE_SCALE" \
  --device "$DEVICE" --dtype "$DTYPE" --out "$REPLAY_DIR"

# Sliding window: this generation plus the two previous ones, where present.
DATA_DIRS="$REPLAY_DIR"
for g in $((GEN - 1)) $((GEN - 2)); do
  if [ "$g" -ge 1 ] && [ -d "$WORK/gen$g/replay" ]; then
    DATA_DIRS="$WORK/gen$g/replay,$DATA_DIRS"
  fi
done

echo "== gen$GEN: train (window: $DATA_DIRS) =="
"$PY" "$TOOLS_DIR/train.py" \
  --data "$DATA_DIRS" --init "$INIT_CKPT" --out "$CKPT" \
  --epochs "$EPOCHS" --device "$DEVICE"

echo "== gen$GEN: pack + verify (clock-removal patch, fixed K=$FIXED_K) =="
# $CKPT is always a train.py output here, i.e. a trained (U-scale) value
# head: pack with the value-scale bridge on (same ruling as above).
"$PY" "$TOOLS_DIR/pack_submission.py" \
  --chassis "$CHASSIS" --ckpt "$CKPT" --out "$SUBMISSION" \
  --value-scale u \
  --fixed-k "$FIXED_K" --verify --verify-games "$PACK_VERIFY_GAMES"

echo "== gen$GEN: mirror eval vs master3s2 ($EVAL_GAMES games, external bigeval) =="
"$PY" "$BIGEVAL" --candidate "$SUBMISSION" --opponent "$MASTER" \
  --games "$EVAL_GAMES" --out "$EVAL_JSON"

read -r MEAN_MARGIN MAX_STEP < <("$PY" - "$EVAL_JSON" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
print(d["mean_margin"], d["max_step_time"])
PYEOF
)
PREV_BEST="-inf"
[ -f "$BEST_FILE" ] && PREV_BEST="$(cat "$BEST_FILE")"

echo "== gen$GEN: gate (margin $MEAN_MARGIN vs best $PREV_BEST; max step ${MAX_STEP}s < ${STEP_LIMIT}s) =="
if "$PY" - "$MEAN_MARGIN" "$PREV_BEST" "$MAX_STEP" "$STEP_LIMIT" <<'PYEOF'
import sys
margin, best, step, limit = (float(x) for x in sys.argv[1:5])
sys.exit(0 if (margin > best and step < limit) else 1)
PYEOF
then
  cp "$SUBMISSION" "$WORK/promoted/gen$GEN.py"
  echo "$MEAN_MARGIN" > "$BEST_FILE"
  echo "PROMOTED: $WORK/promoted/gen$GEN.py"
else
  echo "NOT PROMOTED (archived only at $GEN_DIR)"
fi

{
  echo "gen=$GEN init=$INIT_CKPT games=$GAMES epochs=$EPOCHS fixed_k=$FIXED_K"
  echo "mean_margin=$MEAN_MARGIN max_step_time=$MAX_STEP prev_best=$PREV_BEST"
} > "$GEN_DIR/SUMMARY.txt"
echo "== gen$GEN done: $GEN_DIR =="
