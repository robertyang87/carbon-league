"""Verified replay reader; labels and public-view feature extraction are separate."""
from hashlib import sha256
import gzip
import json
from pathlib import Path
import numpy as np
from value import History, HORIZON,FEATURE_NAMES

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'top5_h2h_data.json'
NEW=HERE.parent/'latest5_inventory.json'
STRIDE=3


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def splits():
    old=json.loads(OLD.read_bytes())['games'];new=json.loads(NEW.read_bytes())['games']
    splits={'train':[g for g in old if g['split']=='train'],'old_val':[g for g in old if g['split']=='val'],'new_diag':new}
    all_games=list(old)+list(new)
    if [len(splits[k]) for k in splits] != [79,21,26] or any(len({g[k] for g in all_games})!=len(all_games) for k in ('id','seed','canonical_sha256')):
        raise ValueError('unapproved split or overlap')
    return splits


def read_game(folder,g):
    path=folder/g.get('path',g['id']+'.json.gz')
    blob=path.read_bytes();r=json.loads(gzip.decompress(blob))
    canonical=json.dumps(r,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
    if sha256(canonical).hexdigest()!=g['canonical_sha256'] or r['configuration']['randomSeed']!=g['seed']:
        raise ValueError(f"unverified game {g['id']}")
    return r


def view(frame,seat):
    shared=frame[0]['observation'];own=frame[seat]['observation']
    if own['player']!=seat:
        raise ValueError('seat mapping')
    return {'player':seat,'step':shared['step'],'players':shared['players'],'carbon':own['carbon'],'trees':shared['trees']}


def future_label(frames,t,seat):
    early=frames[t][0]['observation']['players']
    late=frames[t+HORIZON][0]['observation']['players']
    return (late[seat][0]-late[1-seat][0])-(early[seat][0]-early[1-seat][0])


def dataset(items,folder):
    X=[];Y=[];B=[];map_ids=[];seats=[];ticks=[]
    for idx,item in enumerate(items):
        replay=read_game(folder,item);frames=replay['steps'];cfg=replay['configuration']
        history=(History(),History())
        for t in range(len(frames)-HORIZON):
            for seat,h in enumerate(history):
                v=view(frames[t],seat);h.update(v)
                if t%STRIDE:
                    continue
                X.append(h.encode(v,cfg));Y.append(future_label(frames,t,seat))
                B.append((0.,h.trend(10),h.trend(30)));map_ids.append(idx);seats.append(seat);ticks.append(t)
    x,y,b=np.asarray(X),np.asarray(Y),np.asarray(B)
    if x.shape!=(len(y),len(FEATURE_NAMES)) or not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(b).all():
        raise ValueError('bad data')
    return x,y,b,np.asarray(map_ids),np.asarray(seats),np.asarray(ticks)
