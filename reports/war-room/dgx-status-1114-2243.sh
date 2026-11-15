echo MARK_DGXSTATUS2243
hostname; uptime
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
ps -eo pid,etime,args | grep -E 'k0_annot|rl2_ppo|bigeval|git fetch|git pull|git checkout|rsync' | grep -v grep || echo NO_TRAIN_GIT_RSYNCPROCS
cd /home/arm/junyang/carbon-repo && git rev-parse HEAD && timeout 60 git ls-remote origin main && git status --short | head -8
ls /home/arm/junyang/k0_dist/returns/c1/*.npz 2>/dev/null | wc -l; ls /home/arm/junyang/k0_dist/returns/c2/*.npz 2>/dev/null | wc -l
ls -lt --time-style=+%H:%M:%S /home/arm/junyang/k0_dist/returns/c1/progress_c1_wave2.json /home/arm/junyang/k0_dist/returns/c2/progress.json /home/arm/junyang/k0_dist/returns/c1/progress_c1.json 2>/dev/null
python3 -c "
import json
for n in ['c1/progress_c1_wave2','c2/progress']:
    try:
        d = json.load(open(f'/home/arm/junyang/k0_dist/returns/{n}.json'))
        g = d.get('games', {})
        done = sum(1 for v in g.values() if isinstance(v, dict) and v.get('status') == 'done')
        print(n, 'entries', len(g), 'done', done)
    except Exception as e:
        print(n, 'ERR', e)
"
python3 -c "
import json, os
for p in ['/home/arm/junyang/carbon/data/m3s2_corpus/index.json','/home/arm/junyang/carbon/data/carago_corpus/index.json','/home/arm/junyang/carbon/data/fable_corpus/index.json']:
    try:
        d = json.load(open(p)); print(p.split('/')[-2], 'index', len(d) if isinstance(d, list) else d.get('count', '?'))
    except Exception as e:
        print(p, 'ERR', e)
for p in ['/home/arm/junyang/carbon/data/m3s2_corpus','/home/arm/junyang/carbon/data/carago_corpus','/home/arm/junyang/carbon/data/fable_corpus']:
    try:
        print(p.split('/')[-1], 'replays', len([f for f in os.listdir(p) if f.endswith('.jsonl') or f.endswith('.gz') or f.endswith('.json') and f != 'index.json']))
    except Exception as e:
        print(p, 'ERR', e)
for p in ['/home/arm/junyang/carbon/data/ds_carago/meta.json','/home/arm/junyang/carbon/data/ds_fable/meta.json']:
    try:
        d = json.load(open(p)); print(p.split('/')[-2], 'meta', d.get('total', d.get('num_samples', d.get('count', '?'))))
    except Exception as e:
        print(p, 'ERR', e)
"
ls -lt /home/arm/junyang/carbon/data/bigeval_live/ 2>/dev/null | head -5 || echo NO_BIGEVAL_LIVE
echo END_DGXSTATUS2243
