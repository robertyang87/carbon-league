#!/usr/bin/env python3
"""Warehouse trigger corpora batch 2 (m3s3z 73, hakimi31 90, hakimi3 132,
krypton 160) from local pull to main.
Idempotent: verifies md5s vs DGX manifests, puts only files missing on main,
then reads back directory counts.
"""
import sys, os, hashlib
sys.path.insert(0, '/home/hatch/workspace')
import gh_api

BASE = '/home/hatch/workspace/carbon-incoming/trigger_corpus2'
OWNER, REPO = 'robertyang87', 'carbon-league'
JOBS = [
    ('m3s3z', 'pull2313_m3s3z.manifest', 'data/m3s3z_corpus'),
    ('hakimi31', 'pull2313_hakimi31.manifest', 'data/hakimi31_corpus'),
    ('hakimi3', 'pull2313_hakimi3.manifest', 'data/hakimi3_corpus'),
    ('krypton', 'pull2313_krypton.manifest', 'data/krypton_corpus'),
]

def md5_of(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    for label, manifest_name, remote_dir in JOBS:
        manifest_path = os.path.join(BASE, manifest_name)
        entries = []
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                md5, rel = line.split(None, 1)
                entries.append((md5, rel.strip()))
        bad = []
        for md5, rel in entries:
            local = os.path.join(BASE, rel)
            if not os.path.exists(local):
                bad.append((rel, 'MISSING'))
            elif md5_of(local) != md5:
                bad.append((rel, 'MD5-MISMATCH'))
        print(f'{label}: manifest {len(entries)} files, local md5 check bad={len(bad)}', bad[:5], flush=True)
        if bad:
            print(f'{label}: ABORT (md5 failures)', flush=True)
            continue
        existing = set()
        try:
            items = gh_api.req('GET', f'/repos/{OWNER}/{REPO}/contents/{remote_dir}?ref=main')
            if isinstance(items, list):
                existing = {it['name'] for it in items}
        except Exception as e:
            print(f'{label}: contents list failed: {e}', flush=True)
        print(f'{label}: on main already {len(existing)}', flush=True)
        missing = [(md5, rel) for md5, rel in entries
                   if os.path.basename(rel) not in existing]
        print(f'{label}: to put {len(missing)}', flush=True)
        ok, fail = 0, []
        for md5, rel in missing:
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
            if ok % 10 == 0:
                print(f'{label}: put progress {ok}/{len(missing)}', flush=True)
        print(f'{label}: put ok={ok} fail={len(fail)}', fail[:5], flush=True)
        items = gh_api.req('GET', f'/repos/{OWNER}/{REPO}/contents/{remote_dir}?ref=main')
        print(f'{label}: READBACK main count = {len(items)} (target {len(entries)})', flush=True)

if __name__ == '__main__':
    main()
