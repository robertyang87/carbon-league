"""Freeze all own actual new planting roots in RyanX and the first two v4 wins."""
import gzip,json
from build import HERE,D,sha

OBS=D/'ladder/v4_observations'
IDS=(('RyanX','1c80b916b203238a43267baeabc88fbe'),('Krypton','2cba12ae1302e3e6392cf0bc74476b12'),('stephen','00b13bf8e22eea6cd830d1ab82ae2c85'))


def main():
    games=[]
    for name,mid in IDS:
        lp=OBS/(mid+'-1.ledger.json.gz');rp=OBS/'replays'/(mid+'-1.json.gz');sg=json.loads(gzip.decompress(lp.read_bytes())); ev=sg['planting'][0]['individual']; seat=ev[0]['seat']
        assert all(r['seat']==seat for r in ev); roots=[]
        for t in sorted(set(r['t'] for r in ev)):
            root=[{'uid':r['planter'],'cell':r['cell'],'tree':r['tree']} for r in ev if r['t']==t]
            roots.append(dict(step=t,plants=root,rng=int(sha(f'ryan-tree-soft/1/{mid}/{seat}/{t}'.encode())[:8],16)))
        games.append(dict(name=name,match=mid,seat=seat,archive=str(rp.relative_to(D)),sha256=sha(rp.read_bytes()),ledger=str(lp.relative_to(D)),ledger_sha256=sha(lp.read_bytes()),roots=roots))
    protocol=dict(schema='frozen-first-tenure-root-policy/1',identity=json.loads((HERE/'identity.json').read_bytes()),games=games,
       population='All actual new planting actions of v4 in its RyanX defeat and the first two chronological v4 wins (Krypton and stephen); group simultaneous purchases into one root. Root selection never consults payoff signs. Directly compare fitted cash prediction to future ledger only after calls. These historical roots are selected by behavior and do not measure all planting opportunities or win probability.',
       intervention='One frozen prediction head. Before own root sampling, add clip(predicted_net/public quoted price,-1,0) to NONE only for a planter on a currently empty nonbase cell with enough public cash. Never consult contemporaneous enemy order; pre-action eligibility cannot guarantee the actual planter survives joint collision/payment.',
       measurements='Every selected root: full v4 and candidate agent() on same seat public prefix and RNG, unmodified temperature=1, wall=.58, common initial deployment costs. For any negative delta on ANY eligible own planter (including teammates other than archive purchases) or a difference in the two returned team commands: additionally force first live v4 K/H/R, with identical RNG, opponent and startup state, both arms rerun. Compare menu, returned actions, each actual purchaser vs historical action and first-tenure cash. A contributor requires live and matched-work commands agree, parent replay restored, enemy/granted work equal. Negative or false positive economics of archived purchases are stated separately from projected counterfactual returns. No silent deadline/weather selection.',
       historical_comparison='The original ladder campaigns have unarchived natural private draws; historical plant, experimental parent plant, and new experimental plant must remain separate. No counterfactual return or profit is implied by canceling an archived purchase.')
    raw=(json.dumps(protocol,indent=2,allow_nan=False)+'\n').encode();path=HERE/'protocol.json'
    if path.exists():assert path.read_bytes()==raw
    else:path.write_bytes(raw)
    print(sha(raw),[(c['name'],len(c['roots']),sum(len(r['plants']) for r in c['roots'])) for c in games])

if __name__=='__main__':main()
