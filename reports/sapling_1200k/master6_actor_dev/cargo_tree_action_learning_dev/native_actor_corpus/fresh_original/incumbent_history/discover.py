"""Inventory structured game evidence in a fixed Git tree, including nested archives."""
import gzip
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile
import zipfile
from collections import Counter, defaultdict

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/'.git').exists())
SNAPSHOT = '8a74cf5a55669c1d2e542500d0b69ad202f0563c'
PACKAGE = '8da4b465f7fa6e601e51bd7c2e81f25825cb59b9b5ab285f7f56d0f6bbfe1cac'
FORMATS = ('.json','.jsonl','.gz','.zip','.tar','.tgz')


def scan():
    found = defaultdict(list)
    counts = Counter()
    seen = set()
    reserved = re.compile(rb'(?<!\d)62175835[01](?!\d)')
    def visit(name, data):
        if name.endswith(('.tar','.tar.gz','.tgz')):
            with tarfile.open(fileobj=io.BytesIO(data),mode='r:*') as arc:
                for m in arc:
                    if m.isfile() and m.name.endswith(FORMATS):
                        visit(name+'!'+m.name, arc.extractfile(m).read())
        elif name.endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(data)) as arc:
                for m in arc.namelist():
                    if m.endswith(FORMATS): visit(name+'!'+m, arc.read(m))
        else:
            if name.endswith('.gz'): data=gzip.decompress(data)
            counts['structured_leaves'] += 1
            if reserved.search(data): found['350_351_mentions'].append(name)
            if PACKAGE.encode() not in data: return
            counts['leaves_with_full_sha'] += 1
            found['all_full_sha_references'].append(name)
            if name.endswith(('.json','.json.gz')):
                doc=json.loads(data)
                if not isinstance(doc,dict): return
                if doc.get('pack_sha256')==PACKAGE and 'actions' in doc and 'official_score_by_seat' in doc:
                    found['exact_package_raw_games'].append(name)
                if doc.get('files',{}).get('candidate') and isinstance(doc['files']['candidate'],dict) and doc['files']['candidate'].get('sha256')==PACKAGE and 'candidate_seat' in doc:
                    found['exact_package_registrations'].append(name)
                if 'frozen_files' in doc and doc['frozen_files'].get('candidate',{}).get('sha256')==PACKAGE and 'independently_replayed_official_scores_by_seat' in doc:
                    found['directly_bound_hosted_verifications'].append(name)
    entries=subprocess.check_output(['git','-C',str(ROOT),'ls-tree','-rz',SNAPSHOT]).split(b'\0')
    with subprocess.Popen(['git','-C',str(ROOT),'cat-file','--batch'],stdin=subprocess.PIPE,stdout=subprocess.PIPE) as objects:
        for e in entries:
            if not e: continue
            meta,path=e.split(b'\t',1); name=path.decode()
            if not name.endswith(FORMATS): continue
            oid=meta.split()[-1];counts['top_level_paths']+=1
            seen.add(oid);objects.stdin.write(oid+b'\n');objects.stdin.flush()
            size=int(objects.stdout.readline().split()[-1]);content=objects.stdout.read(size);assert objects.stdout.read(1)==b'\n'
            visit(name,content)
        objects.stdin.close()
    return dict(snapshot=SNAPSHOT, formats=FORMATS, unique_git_blobs=len(seen),counts=counts, **{k:sorted(v) for k,v in found.items()})


if __name__=='__main__':
    (HERE/'discovery.json').write_text(json.dumps(scan(),indent=2,sort_keys=True)+'\n')
