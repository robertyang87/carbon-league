#!/usr/bin/env python3
"""rsync the 114 gap3-missing files from DGX clone data dirs; md5 vs manifest + gzip -t."""
import json, os, subprocess, hashlib, gzip
STAGE = '/home/hatch/workspace/carbon-live/corpus_staging_gap3'
DGX = 'arm@100.84.148.73'
SSH = '/home/hatch/workspace/carbon-incoming/ssh_dgx.sh'
diff = json.load(open(f'{STAGE}/gap3_diff.json'))
def md5_file(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()
report = {}
for label, d in diff.items():
    missing = d['missing']
    staging = os.path.join(STAGE, label)
    os.makedirs(staging, exist_ok=True)
    listfile = os.path.join(STAGE, f'{label}_fetch.txt')
    with open(listfile, 'w') as f:
        f.write('\n'.join(missing) + '\n')
    src = f'{DGX}:/home/arm/junyang/carbon-repo/data/{label}_corpus/'
    r = subprocess.run(['rsync', '-a', '--files-from=' + listfile, '-e', SSH, src, staging + '/'],
                       capture_output=True, text=True)
    print(f'[{label}] rsync rc={r.returncode} {r.stderr.strip()[:200]}', flush=True)
    man = {}
    for line in open(f'{STAGE}/pull0031_{label}_gap3.manifest'):
        parts = line.split()
        if len(parts) == 2:
            man[parts[1].split('/')[-1]] = parts[0]
    ok = fail = 0
    fails = []
    for name in missing:
        p = os.path.join(staging, name)
        good = os.path.exists(p) and md5_file(p) == man[name]
        if good:
            try:
                json.load(gzip.open(p))
            except Exception as e:
                good = False; fails.append((name, f'gzip/json: {e}'))
        if good: ok += 1
        else:
            fail += 1
            if (name, '') not in fails: fails.append((name, 'md5/missing'))
    print(f'[{label}] verify: ok {ok} / fail {fail}', flush=True)
    for f in fails[:5]: print('  FAIL', f, flush=True)
    report[label] = {'ok': ok, 'fail': fail}
    assert fail == 0, f'{label}: {fail} files failed fetch/verify'
json.dump(report, open(f'{STAGE}/fetch_report.json', 'w'))
print('FETCH_DONE', json.dumps(report), flush=True)
