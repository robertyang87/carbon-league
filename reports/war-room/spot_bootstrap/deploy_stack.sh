#!/bin/bash
# deploy_stack.sh - run FROM DGX, as user arm.
#
# Purpose: turn a bootstrapped replacement spot into an exam-ready machine:
#   1. Push the golden stack tarball (built from spot-30's /root/carbon) to
#      the new machine over the tailnet, using DGX's carbon_relay key (the
#      new machine authorized its pubkey in bootstrap.sh).
#   2. Extract it to /root/carbon on the target.
#   3. Refresh the code files from the version anchor (see CODE_ANCHOR
#      below; golden-image code is stale), then verify the manifest: venv
#      python imports numpy, and the harness files match the anchor md5s.
#   4. Print EXAM-READY on success; fail loudly otherwise.
#
# Usage: bash deploy_stack.sh <new-machine-tailnet-ipv4>
set -euo pipefail

TARGET="${1:?usage: deploy_stack.sh <new-machine-tailnet-ipv4>}"
# Code version anchor (owner order 2026-11-13): repository code must match
# carbon-league main HEAD a7ed8f9552ffc7217c8ea4fe432d575713b4295f.
# The golden image supplies environment + weights only; its code files are
# stale and must be refreshed from the anchor before the manifest check.
CODE_ANCHOR="a7ed8f9552ffc7217c8ea4fe432d575713b4295f"
CODE_REFRESH_DIR="/root/code_refresh_a7ed8f95"
GOLDEN="/home/arm/junyang/golden/exam_stack_golden.tar.zst"
KEY="${HOME}/.ssh/id_ed25519_carbon_relay"
SSH_OPTS="-i ${KEY} -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new"

fail() { echo "DEPLOY-FAIL: $*" >&2; exit 1; }

[ -s "${GOLDEN}" ] || fail "golden tarball missing or empty: ${GOLDEN}"
[ -r "${KEY}" ] || fail "relay key not readable: ${KEY}"

# Integrity of the golden image before it leaves DGX.
if [ -f "${GOLDEN%.tar.zst}.md5" ]; then
  want="$(cat "${GOLDEN%.tar.zst}.md5" | tr -d ' \n')"
  got="$(md5sum "${GOLDEN}" | awk '{print $1}')"
  [ "${want}" = "${got}" ] || fail "golden tarball md5 mismatch (sidecar ${want} != actual ${got})"
  echo "golden md5 verified: ${got}"
fi
zstd -t -q "${GOLDEN}" || fail "golden tarball fails zstd integrity test"

echo "pushing golden tarball to root@${TARGET} ..."
# shellcheck disable=SC2086
scp ${SSH_OPTS} "${GOLDEN}" "root@${TARGET}:/root/exam_stack_golden.tar.zst" \
  || fail "scp to ${TARGET} failed"

echo "extracting to /root/carbon on ${TARGET} ..."
# shellcheck disable=SC2086
ssh ${SSH_OPTS} "root@${TARGET}" '
  set -e
  if command -v zstd >/dev/null 2>&1; then
    zstd -dc /root/exam_stack_golden.tar.zst | tar -xf - -C /root
  else
    echo "NO_ZSTD_ON_TARGET"
    exit 42
  fi
' || {
  rc=$?
  if [ "${rc}" = "42" ]; then
    echo "target has no zstd; falling back to DGX-side decompression stream..."
    # shellcheck disable=SC2086
    zstd -dc "${GOLDEN}" | ssh ${SSH_OPTS} "root@${TARGET}" 'tar -xf - -C /root' \
      || fail "streamed extraction failed"
  else
    fail "extraction on target failed (rc=${rc})"
  fi
}

echo "refreshing code from anchor ${CODE_ANCHOR} on ${TARGET} ..."
# shellcheck disable=SC2086
ssh ${SSH_OPTS} "root@${TARGET}" "
  if [ -d ${CODE_REFRESH_DIR} ]; then
    cp -f ${CODE_REFRESH_DIR}/eval_v53.py ${CODE_REFRESH_DIR}/bigeval.py ${CODE_REFRESH_DIR}/certify_pack.py ${CODE_REFRESH_DIR}/cnet2_search.py /root/carbon/code/ \
      && echo CODE-REFRESHED-FROM-STAGING
  else
    echo 'CODE-REFRESH-SKIPPED: no staging dir on target; manifest check below enforces the anchor md5s and will fail on golden-image code'
  fi
" || fail "code refresh step failed on ${TARGET}"

echo "verifying manifest on ${TARGET} ..."
# shellcheck disable=SC2086
ssh ${SSH_OPTS} "root@${TARGET}" '
  set -u
  fail=0
  /root/carbon/.venv/bin/python -c "import numpy; print(\"numpy\", numpy.__version__)" \
    || { echo "MANIFEST-FAIL: venv python cannot import numpy"; fail=1; }
  check() {
    want="$1"; path="$2"
    got="$(md5sum "${path}" 2>/dev/null | awk "{print \$1}")"
    if [ "${got}" = "${want}" ]; then
      echo "ok ${path} ${got}"
    else
      echo "MANIFEST-FAIL: ${path} md5 ${got:-missing} != ${want}"
      fail=1
    fi
  }
  check aa4c5bb63c879caffeb33b0b25c498d7 /root/carbon/code/eval_v53.py
  check 327ff8073932cc8bd59c3e59090d2647 /root/carbon/code/bigeval.py
  check 3c058ddc219272c389c4061907338b41 /root/carbon/code/certify_pack.py
  check 10a0674e5fd572801d2a06876f1de54b /root/carbon/code/cnet2_search.py
  [ -d /root/carbon/data/weights ] || { echo "MANIFEST-FAIL: data/weights missing"; fail=1; }
  exit ${fail}
' || fail "manifest verification failed on ${TARGET}"

echo "EXAM-READY"
echo "target: root@${TARGET} (stack at /root/carbon)"
