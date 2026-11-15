#!/bin/bash
# War-room staging stage1 (runs ON DGX, detached): carago index+dataset+train launch,
# then fable index+dataset (Studio pulls ds_fable itself afterwards).
# Self-staging per mechanism rule 2026-11-12: verify preconditions in-run, attest each step.
D=/home/arm/junyang/carbon/data
C=/home/arm/junyang/carbon/code
cd $C || exit 1
echo "STAGE1 BEGIN $(date '+%F %T %Z')"
echo '--- PRECONDITIONS ---'
echo -n 'carago_replays='; ls $D/carago_corpus/replays/ | wc -l
echo -n 'fable_replays='; ls $D/fable_corpus/replays/ | wc -l
nvidia-smi --query-gpu=utilization.gpu,temperature.gpu --format=csv,noheader
ps -eo pid,etime,args | grep -E 'cnet2_train' | grep -v grep || echo NO-TRAIN-PROC
echo '--- BUILD-INDEXES (FIX4, format mirrored from m3s2 index sample) ---'
../.venv/bin/python - <<'PYEOF'
import json, gzip, os
D = '/home/arm/junyang/carbon/data'
sample = json.load(open(f'{D}/m3s2_corpus/index.json'))[0]
print('sample keys:', sorted(sample.keys()))
print('sample strategies:', sample.get('strategies'))
for label in ['carago', 'fable']:
    matches = json.load(open(f'{D}/{label}_matches.json'))
    entries, missing = [], 0
    for m in matches:
        fn = f"{m['id']}-{m['game']}.json.gz"
        path = f'{D}/{label}_corpus/replays/{fn}'
        if not os.path.exists(path):
            missing += 1
            continue
        with gzip.open(path, 'rt') as f:
            rep = json.load(f)
        e = {}
        for k in sample.keys():
            if k == 'id': e[k] = m['id']
            elif k == 'game': e[k] = m['game']
            elif k == 'file': e[k] = f'replays/{fn}'
            elif k == 'steps': e[k] = len(rep['steps']) - 1
            elif k == 'strategies':
                e[k] = {'a': m['name_a'], 'b': m['name_b']} if isinstance(sample['strategies'], dict) else [m['name_a'], m['name_b']]
            elif k in ('scores', 'winner', 'order', 'created'): e[k] = m[k]
            else: e[k] = m.get(k)
        entries.append(e)
    json.dump(entries, open(f'{D}/{label}_corpus/index.json', 'w'), ensure_ascii=False)
    print(label, 'index entries:', len(entries), 'missing replays:', missing)
PYEOF
echo '--- BUILD-DS-CARAGO ---'
rm -rf $D/ds_carago
CNET2_TOP_TEACHERS="Carago-V2微调@2" ../.venv/bin/python build_top.py $D/ds_carago $D/carago_corpus > $D/dq_logs/build_carago.log 2>&1; echo "build_carago_rc=$?"
../.venv/bin/python set_uw.py $D/ds_carago 1 1 1 >> $D/dq_logs/build_carago.log 2>&1
tail -4 $D/dq_logs/build_carago.log
if [ -f $D/ds_carago/meta.json ]; then
  echo '--- LAUNCH-CARAGO-TRAIN ---'
  export BC_DEVICE=cuda BC_RAM=1 BC_THREADS=5 BC_USE_W=1 BC_USE_UW=1 BC_BS=256 BC_SAVE_EPOCHS=1 BC_PCT=0.1 BC_LR=3e-4
  unset BC_AMP BC_CLIP BC_TRAIN_PREFIX BC_CACHE_TRUNK BC_SELF_KL BC_EXIT_W BC_AUX_W BC_COMPILE BC_CAR_KL
  BC_INIT=$D/weights/cnet2_arch_n1_last.npz nohup ../.venv/bin/python cnet2_train.py $D/ds_carago $D/weights/cnet2_caragor1.npz 20 '{"C":128,"NB":8,"D":128,"ZI":1,"TA":4,"TNH":4}' > $D/dq_logs/caragor1.log 2>&1 &
  BC_INIT=$D/weights/cnet2_b128m2t3.npz nohup ../.venv/bin/python cnet2_train.py $D/ds_carago $D/weights/cnet2_caragog1.npz 20 '{"C":128,"NB":8,"D":128}' > $D/dq_logs/caragog1.log 2>&1 &
  sleep 5
  echo "train pids: $(pgrep -f 'cnet2_train[.]py' | tr '\n' ' ')"
else
  echo 'BUILD-CARAGO-FAILED: no meta.json, training NOT launched'
fi
echo '--- BUILD-DS-FABLE ---'
rm -rf $D/ds_fable
CNET2_TOP_TEACHERS="Fable5.2-Max-V5@2" ../.venv/bin/python build_top.py $D/ds_fable $D/fable_corpus > $D/dq_logs/build_fable.log 2>&1; echo "build_fable_rc=$?"
../.venv/bin/python set_uw.py $D/ds_fable 1 1 1 >> $D/dq_logs/build_fable.log 2>&1
tail -4 $D/dq_logs/build_fable.log
du -sh $D/ds_fable 2>/dev/null
echo "STAGE1 END $(date '+%F %T %Z')"
