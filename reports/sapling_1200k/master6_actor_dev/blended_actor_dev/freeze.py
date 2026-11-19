"""Bind every existing annotated train/validation/holdout game before fitting."""
import collections
import gzip
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ACTOR = HERE.parent
ROOT = ACTOR.parents[2]
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def freeze():
    original = ACTOR / 'data_split.json'
    receipts = ACTOR / 'data_receipts.json.gz'
    previous = json.loads(original.read_bytes())
    rr = json.loads(gzip.open(receipts, 'rb').read())
    assert rr['split_sha256'] == sha(original)
    signed = {g['id']:g for g in rr['games']}
    panel = []
    for game in previous['games']:
        if game['split'] not in ('train','val','holdout'):
            continue
        r = signed[game['id']]
        assert r['source_sha256'] == game['sha256'] and r['gid'] == game['gid']
        panel.append(dict(game, shard_sha256=r['sha256'], rows=r['rows']))
    assert len(panel) == len(signed)
    for field in ('id','seed','terrain_sha256','gid'):
        by_split = {s:{g[field] for g in panel if g['split']==s} for s in ('train','val','holdout')}
        assert all(not (by_split[s]&by_split[t]) for s in by_split for t in by_split if s!=t)
    document = {'source_sha256':sha(original),'shards_receipt_sha256':sha(receipts),'games':panel,
                'prior_status':'Previously established actor development panels, excluded from formal evaluation.'}
    def write(path,obj):
        payload=(json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode()
        if path.exists() and path.read_bytes()!=payload:
            raise ValueError(f'existing freeze differs: {path}')
        path.write_bytes(payload)
    write(HERE/'panel.json',document)
    protocol={'panel_sha256':sha(HERE/'panel.json'),'partner_pack_sha256':sha(ACTOR/'actor_partner_dev/candidate.py'),
              'parent_weights_sha256':sha(ROOT/'data/weights/sapling_1200k/master6_actor_v1.npz'),
              'diagnostic_previously_exposed_losses_receipt':sha(ACTOR/'data_loss_receipts.json.gz'),
              'batch_size':16,'updates':128,'random_seed':26368871,'optimizer':'AdamW','lr':0.0006,'weight_decay':0.01,
              'clip_grad_norm':1.0,'train_modules':['h0','din','gru','aemb','whead','bhead','U','V'],
              'schedule':'One deterministic permutation, first 2048 annotated train roots, same order in both arms.',
              'candidate':'Legal 0.75 frozen deployed Echo4 it20 logit + 0.25 Sapling logit, recorded common prefix, CE averaged over all active multi-choice slots.',
              'control':'Standalone Sapling CE, same parent, trainable modules, optimizer, examples, minibatch schedule and update count.',
              'distribution':'Diagnostic softmax over legal mixed scores. Runtime chooses argmax and applies registered return residual and joint fallback.',
              'gate':'Arena freezes its own primary gate. Mechanism check: map-equal deployed validation CE below original, plus actual altered joint command. Superiority to same-budget CE control reported separately.',
              'holdout':'All 10 val / 18 holdout maps, separate unseen registration opponent and role reports. These historical monitoring sets contain wins only. All 18 exposed losses form distinct diagnostic set.'}
    write(HERE/'protocol.json',protocol)

if __name__=='__main__':
    freeze()
