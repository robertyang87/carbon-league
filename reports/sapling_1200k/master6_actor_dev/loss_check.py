"""Read the excluded historical losses once the registered student is frozen."""
import gzip
import json
from pathlib import Path
import torch
from train import HERE, WEIGHTS, digest, model, public, predict, quantized, score, write_json


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    split = json.loads((HERE/'data_loss_split.json').read_bytes())
    if split['original_split_sha256'] != digest((HERE/'data_split.json').read_bytes()):
        raise ValueError('different frozen data split')
    with gzip.open(HERE/'data_loss_receipts.json.gz','rt') as f:
        receipt=json.load(f)
    rows=split['games']
    if any(g['wdl']!='L' or g['original_split']!='excluded' for g in rows):
        raise ValueError('loss diagnostic must use only excluded old losses')
    main_ids={g['id'] for g in json.loads((HERE/'data_split.json').read_bytes())['games'] if g['split']!='excluded'}
    if main_ids & {g['id'] for g in rows}:
        raise ValueError('overlap with this training selection')
    locations={g['id']:g['sha256'] for g in receipt['games']}
    d,_=public('diagnostic_seen_loss',rows,Path(receipt['shard_directory']),locations)
    weights=json.loads((HERE/'weight_receipt.json').read_bytes())
    a,_=model(WEIGHTS/'cash_v3.npz',weights['recipe']['parent_sha256'])
    b,_=model(WEIGHTS/'master6_actor_v1.npz',weights['npz_sha256'])
    scores=score(d,predict(quantized(a),d),predict(quantized(b),d),{int(g['gid']):g for g in rows})
    write_json(HERE/'exposed_loss_metrics.json',dict(status='exposed games diagnostic; not an independent distribution of failures',
               roots=len(d['y']),weights=weights['npz_sha256'],scores=scores))
    print('Previously exposed losses only',scores['all'])

if __name__=='__main__':main()
