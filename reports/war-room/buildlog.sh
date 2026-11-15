#!/bin/bash
# Read the HEAD of build_carago/build_fable logs (build_top traceback lives there,
# stage1 only surfaced set_uw's tail error) + confirm on-disk index shape.
# Marker: BUILOG16A
MARK=BUILOG16A
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "BUILOG16A DGX-BEGIN $(date '+%F %T %Z')"
D=/home/arm/junyang/carbon/data
echo '--- BUILD-CARAGO-LOG-HEAD ---'
head -30 $D/dq_logs/build_carago.log
echo '--- BUILD-FABLE-LOG-HEAD ---'
head -30 $D/dq_logs/build_fable.log
echo '--- MY-INDEX-FIRST-ENTRY ---'
../junyang/carbon/.venv/bin/python -c "import json; e=json.load(open('/home/arm/junyang/carbon/data/carago_corpus/index.json')); print(json.dumps(e[0], ensure_ascii=False)[:400])" 2>/dev/null || /home/arm/junyang/carbon/.venv/bin/python -c "import json; e=json.load(open('/home/arm/junyang/carbon/data/carago_corpus/index.json')); print(json.dumps(e[0], ensure_ascii=False)[:400])"
echo '--- CORPUS-DIR-LISTING ---'
ls /home/arm/junyang/carbon/data/carago_corpus/ | head -8
echo "BUILOG16A DGX-END"
DGXEOF
echo "$MARK SSH_RC=$? L20-END"
