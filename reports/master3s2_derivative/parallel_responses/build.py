"""Reconstruct either local ablation from the live v2, retaining neural weights and original wall deadline."""
import argparse
import ast
import json
from pathlib import Path
from register import HERE,PARENT,sha

def build():
    raw=PARENT.read_bytes()
    expected=json.loads((HERE/'protocol.json').read_bytes())['parent']
    if sha(raw)!=expected:raise ValueError('Parent source changed')
    replacements={b'_S_M = 1 ':b'_S_M = 2 ', b'_S_MMAX = 1 ':b'_S_MMAX = 2 '}
    patched=raw
    for old,new in replacements.items():
        if patched.count(old)!=1:raise ValueError('Missing native assignment')
        patched=patched.replace(old,new)
    diff=[i for i,(a,b) in enumerate(zip(raw,patched,strict=True)) if a!=b]
    assert len(diff)==2 and all(raw[i]==ord('1') and patched[i]==ord('2') for i in diff)
    compile(patched,'offline two-response ablation','exec')
    return patched,dict(base_sha256=expected,research_source_sha256=sha(patched),bytes=len(patched),changed_byte_offsets=diff,
                        M=2,MMAX=2,changed_weights=0,original_time_seconds=.58,
                        purpose='registered root-level response breadth ablation; no official qualification or ladder submission')

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--write-research-copy',type=Path)
    args=ap.parse_args()
    raw,meta=build()
    (HERE/'source_identity.json').write_text(json.dumps(meta,indent=2)+'\n')
    if args.write_research_copy:
        with args.write_research_copy.open('xb') as writer:writer.write(raw)
    print(meta)
