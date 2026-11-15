#!/usr/bin/env python3
"""Warehouse trigger corpora batch 3 from DGX direct pull to main.
Manifests are FULL corpus listings (903 lines); only files missing on main
(176 expected) are pulled from DGX and put.
Phase A (--phase list): read manifests + main dir listings, write the
rsync files-from list of missing files to trigger_corpus3/needed.txt.
Phase B (--phase put): verify md5s of pulled missing files vs manifests,
put them via API, read back per-label counts vs manifest totals.
"""
import sys, os, hashlib, argparse
sys.path.insert(0, '/home/hatch/workspace')
import gh_api

BASE = '/home/hatch/workspace/carbon-incoming/trigger_corpus3'
OWNER, REPO = 'robertyang87', 'carbon-league'
JOBS = [
    ('m3s3z', 'pull0030_m3s3z.manifest', 'data/m3s3z_corpus'),
    ('hakimi31', 'pull0030_hakimi31.manifest', 'data/hakimi31_corpus'),
    ('krypton', 'pull0030_krypton.manifest', 'data/krypton_corpus'),
    ('hakimi3pro', 'pull0030_hakimi3pro.manifest', 'data/hakimi3pro_corpus'),
    ('lite-v8', 'pull0030_lite-v8.manifest', 'data/lite-v8_corpus'),
    ('fable', 'pull0030_fable.manifest', 'data/fable_corpus'),
    ('m3s3', 'pull0030_m3s3.manifest', 'data/m3s3_corpus'),
]

def md5_of(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def load_manifest(name):
    entries = []
    with open(os.path.join(BASE, name)) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            md5, rel = line.split(None, 1)
            entries.append((md5, rel.strip()))
    return entries

def main_contents(remote_dir):
    try:
        items = gh_api.req('GET', f'/repos/{OWNER}/{REPO}/contents/{remote_dir}?ref=main')
        if isinstance(items, list):
            return {it['name'] for it in items}
        return set()
    except Exception as e:
        print(f'{remote_dir}: contents list failed ({e}); treating as empty', flush=True)
        return set()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['list', 'put'], required=True)
    args = ap.parse_args()
    needed = []  # (label, md5, rel, remote_dir)
    for label, manifest_name, remote_dir in JOBS:
        entries = load_manifest(manifest_name)
        existing = main_contents(remote_dir)
        missing = [(md5, rel) for md5, rel in entries
                   if os.path.basename(rel) not in existing]
        print(f'{label}: manifest {len(entries)} on-main {len(existing)} missing {len(missing)}', flush=True)
        for md5, rel in missing:
            needed.append((label, md5, rel, remote_dir))
    print(f'TOTAL missing {len(needed)}', flush=True)
    if args.phase == 'list':
        with open(os.path.join(BASE, 'needed.txt'), 'w') as f:
            for label, md5, rel, remote_dir in needed:
                f.write(rel + '\n')
        with open(os.path.join(BASE, 'needed_meta.txt'), 'w') as f:
            for label, md5, rel, remote_dir in needed:
                f.write(f'{label}\t{md5}\t{rel}\t{remote_dir}\n')
        print('wrote needed.txt + needed_meta.txt', flush=True)
        return
    # phase put
    bad = []
    for label, md5, rel, remote_dir in needed:
        local = os.path.join(BASE, rel)
        if not os.path.exists(local):
            bad.append((rel, 'MISSING'))
        elif md5_of(local) != md5:
            bad.append((rel, 'MD5-MISMATCH'))
    print(f'local md5 check bad={len(bad)}', bad[:5], flush=True)
    if bad:
        print('ABORT (md5 failures)', flush=True)
        return
    for label, manifest_name, remote_dir in JOBS:
        entries = load_manifest(manifest_name)
        todo = [(md5, rel) for md5, rel, rd in
                [(m, r, d) for (lb, m, r, d) in needed if lb == label]]
        # todo recomputed cleanly below
        todo = [(md5, rel) for (lb, md5, rel, rd) in needed if lb == label]
        ok, fail = 0, []
        for md5, rel in todo:
            local = os.path.join(BASE, rel)
            with open(local, 'rb') as f:
                data = f.read()
            name = os.path.basename(rel)
            try:
                gh_api.put_file(OWNER, REPO, f'{remote_dir}/{name}', data,
                                f'corpus {label}: add {name} (trigger insurance, DGX direct pull)',
                                branch='main')
                ok += 1
            except Exception as e:
                fail.append((name, str(e)[:120]))
            if ok % 10 == 0 and ok:
                print(f'{label}: put progress {ok}/{len(todo)}', flush=True)
        print(f'{label}: put ok={ok} fail={len(fail)}', fail[:5], flush=True)
        items = gh_api.req('GET', f'/repos/{OWNER}/{REPO}/contents/{remote_dir}?ref=main')
        print(f'{label}: READBACK main count = {len(items)} (target {len(entries)})', flush=True)

if __name__ == '__main__':
    main()
