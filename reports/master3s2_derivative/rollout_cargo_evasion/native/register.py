"""Register the first eligible previous development root of each map/seat without next actions."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

HERE=Path(__file__).resolve().parent
D=HERE.parents[1]
PARENT='cd230d40a08f7dea07e261202b865f951b88b3b030febc9828627aa5f47c7d2a'
ORIGINAL='65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5'
def sha(raw):return hashlib.sha256(raw).hexdigest()

def main():
    runtime=HERE.parent/'runtime.py';spec=importlib.util.spec_from_file_location('public_evasion',runtime);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    base=D/'nn_policy_development/hosted';settle=json.loads((base/'settlement.json').read_bytes())
    tickets=[];arm='new_vs_original'
    for a in sorted(settle['archives'],key=lambda t:(t['seed'],t['seat'])):
        seed,seat=a['seed'],a['seat']
        assert seed in (621758608,621758609)
        file=base/f"{a['id']}.zip";assert sha(file.read_bytes())==a['sha256']
        with zipfile.ZipFile(file) as z:
            reg=z.read(arm+'/registration.json');j=json.loads(reg)
            assert j['candidate_seat']==seat and j['files']['candidate']['sha256']==PARENT and j['files']['opponent']['sha256']==ORIGINAL
            obs=gzip.decompress(z.read(arm+'/observations.jsonl.gz')).splitlines()
            for t in range(3,296,4):
                prev=json.loads(obs[t-1])['observations'][seat]['players'];now=json.loads(obs[t])['observations'][seat]
                units=m._ev_units(now['players'],1-seat,prev)
                if units:break
            else:raise ValueError('No qualifying public root')
        tickets.append(dict(archive=str(file.relative_to(D)),archive_sha256=a['sha256'],arm=arm,registration_sha256=sha(reg),seed=seed,seat=seat,
                            terrain=j['physical_map_sha256'],step=t,eligible_root_workers=[u[0] for u in units],
                            rng=int(sha(f'roll-evasion/search/{seed}/{seat}/{t}'.encode())[:8],16)))
    assert len(tickets)==4
    p=dict(parent=PARENT,original=ORIGINAL,tickets=tickets,source_geometry_sha256=sha(runtime.read_bytes()),
           scope='Both physical seats on the same two archived maps 608/609. First eligible historical public state at t=3,7,...295 contemporaneously, no actions read for selection. Only diagnostics; as a rollout model, a root opportunity does not guarantee a qualifying successor.',
           method='Live ABBA consecutive original real 0.58 s calls, then one parent/child diagnostic with K,H,R fixed to first live parent, common seeds. Check root AI logits and enemy joint order; capture actual downstream sampled enemy order on shared same afterstate and RNG. One unchanged A0 and one equal-work final switch are distinguished. No new game.')
    raw=(json.dumps(p,indent=2,sort_keys=True)+'\n').encode();f=HERE/'protocol.json'
    if f.exists() and f.read_bytes()!=raw:raise ValueError('Different registered selections')
    f.write_bytes(raw);print(sha(raw),[(c['seed'],c['seat'],c['step']) for c in tickets])
if __name__=='__main__':main()
