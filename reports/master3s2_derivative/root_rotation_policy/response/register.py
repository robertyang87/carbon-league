"""Freeze one latest robust old root per changed map before simulating any suffix."""
import gzip
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
D=HERE.parents[1]
NATIVE=HERE.parent/'native/roots.json.gz'
SHA='9de880b60a05c0c218584452d0f122d980b884131cb68872290bf93446afa240'
ORIGINAL='65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5'
sha=lambda x:hashlib.sha256(x).hexdigest()


def main():
    raw=NATIVE.read_bytes();assert sha(raw)==SHA;n=json.loads(gzip.decompress(raw));by={}
    for r in n['roots']:
        if r['check']['robust_live_effect']:by.setdefault(r['ticket']['terrain'],[]).append(r)
    assert sum(map(len,by.values()))==4 and len(by)==2
    tickets=[]
    for rows in sorted(by.values(),key=lambda rs:rs[0]['ticket']['seed']):
        root=min(rows,key=lambda r:(-r['ticket']['step'],r['ticket']['seat']))
        c=root['ticket'];r=rng=c['rng'];a=root['live']['parent'][0]['issued'];b=root['live']['candidate'][0]['issued'];assert a!=b
        tickets.append(c|dict(orders={'parent':a,'rotated':b},own_public_sha256=root['visible_before_sha256'],
                              initial_cost=root['warm_cost'],rival_rng=rng^0x7d4a1f6b,branch_order=['parent','rotated'] if len(tickets)==0 else ['rotated','parent']))
    doc=dict(schema='root_rotation_true_response/1',native_sha256=SHA,continuation_parent=n['parent_sha256'],
             source_candidate=n['candidate_sha256'],examiner=ORIGINAL,selection='Across the four robust real-time + matched-work roots, choose the latest step separately on every represented terrain, seat as tie break. No future response or score used.',
             method='Apply the registered actual full own commands, recompute one common opponent response from the unmodified Master3s2 public entry with a prespecified RNG; both root observations are restored from full official-prefix actions. After intervention both branches use v4 vs unchanged original with native 0.58s, temperature 1, clone identical root-side game, agent belief, RNG, and timing state, follow branch-specific public history through time. Historical private RNG is unavailable. CPUs are shared and natural clocks may give different work. Include 10-step and terminal cash components, all four suffixes, no additional trials or recovery by resampling.',tickets=tickets)
    target=HERE/'protocol.json';data=(json.dumps(doc,indent=2,allow_nan=False)+'\n').encode()
    if target.exists():assert target.read_bytes()==data
    else:target.write_bytes(data)
    print(sha(data),[(x['seed'],x['seat'],x['step']) for x in tickets]);return doc

if __name__=='__main__':main()
