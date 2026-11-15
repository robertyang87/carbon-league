echo "DGXSTATUS1043B START $(date '+%F %T %Z')"
nvidia-smi
echo "--- data dirs in carbon-repo ---"
ls -la /home/arm/junyang/carbon-repo/data/ 2>&1 | head -30
echo "--- m3s2 corpus ---"
ls /home/arm/junyang/carbon-repo/data/m3s2_corpus/ 2>&1 | head -5
ls /home/arm/junyang/carbon-repo/data/m3s2_corpus/replays 2>&1 | head -3
python3 -c "import json; d=json.load(open('/home/arm/junyang/carbon-repo/data/m3s2_corpus/index.json')); print('m3s2_index', len(d))"
echo "--- carago/fable index ---"
python3 -c "import json; d=json.load(open('/home/arm/junyang/carbon-repo/data/carago_corpus/index.json')); print('carago_index', len(d))"
python3 -c "import json; d=json.load(open('/home/arm/junyang/carbon-repo/data/fable_corpus/index.json')); print('fable_index', len(d))"
echo "--- ds meta ---"
python3 -c "import json; print('ds_carago', json.load(open('/home/arm/junyang/carbon-repo/data/ds_carago/meta.json')))"
python3 -c "import json; print('ds_fable', json.load(open('/home/arm/junyang/carbon-repo/data/ds_fable/meta.json')))"
echo "--- working dir carbon/data ---"
ls /home/arm/junyang/carbon/data/ 2>&1 | head -20
echo "DGXSTATUS1043B END"
