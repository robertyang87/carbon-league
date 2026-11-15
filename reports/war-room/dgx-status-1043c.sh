echo "DGXSTATUS1043C START $(date '+%F %T %Z')"
ls /home/arm/junyang/carbon-repo/data/m3s2_corpus/*.json.gz 2>/dev/null | wc -l
ls /home/arm/junyang/carbon-repo/data/m3s2_corpus/ | wc -l
ls /home/arm/junyang/carbon/data/ | grep -E 'corpus|ds_' 
python3 -c "import json; d=json.load(open('/home/arm/junyang/carbon/data/m3s2_corpus/index.json')); print('wd m3s2_index', len(d))"
python3 -c "import json; d=json.load(open('/home/arm/junyang/carbon/data/carago_corpus/index.json')); print('wd carago_index', len(d))"
python3 -c "import json; d=json.load(open('/home/arm/junyang/carbon/data/fable_corpus/index.json')); print('wd fable_index', len(d))"
python3 -c "import json; print('wd ds_carago', json.load(open('/home/arm/junyang/carbon/data/ds_carago/meta.json')))"
python3 -c "import json; print('wd ds_fable', json.load(open('/home/arm/junyang/carbon/data/ds_fable/meta.json')))"
echo "DGXSTATUS1043C END"
