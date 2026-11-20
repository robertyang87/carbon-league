"""Register historical roots without reading commands, subsequent boards, or scores."""
import hashlib
import json
from pathlib import Path
import zipfile

HERE=Path(__file__).resolve().parent
D=HERE.parents[1]
PARENT='cd230d40a08f7dea07e261202b865f951b88b3b030febc9828627aa5f47c7d2a'
CANDIDATE='cb64b1065192631f0d03b065bdcdfdb81e93b0c11788cad66cb27c54e8aae271'
ORIGINAL='65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5'
STEPS=(117,247)
sha=lambda data:hashlib.sha256(data).hexdigest()


def main():
    tickets=[]
    for stage,seeds,arm in [('dev1',(621758580,621758581),'opponent_policy'),('nn_policy_development',(621758608,621758609),'new_vs_original')]:
        for path in sorted((D/stage/'hosted').glob('*.zip')):
            with zipfile.ZipFile(path) as z:
                regraw=z.read(f'{arm}/registration.json');reg=json.loads(regraw)
                seed,side=reg['seed'],reg['candidate_seat']
                if seed not in seeds:continue
                assert reg['files']['opponent']['sha256']==ORIGINAL
                for t in STEPS:
                    rng=int(sha(f'root-rotation-native/1/{seed}/{side}/{t}'.encode())[:8],16)
                    tickets.append(dict(stage=stage,archive=str(path.relative_to(D)),archive_sha256=sha(path.read_bytes()),
                                    arm=arm,registration_sha256=sha(regraw),seed=seed,seat=side,terrain=reg['physical_map_sha256'],
                                    history_source_package=reg['files']['candidate']['sha256'],step=t,rng=rng))
    tickets.sort(key=lambda v:(v['seed'],v['seat'],v['step']))
    assert len(tickets)==16 and len({r['terrain'] for r in tickets})==4 and len({(r['seed'],r['seat'],r['step']) for r in tickets})==16
    out=dict(schema='root-rotation-native-crossover/1',parent_sha256=PARENT,candidate_sha256=CANDIDATE,
             steps=STEPS,tickets=tickets,eligibility='First two archived seeds of dev1, both completed seeds of v4 new_vs_original; both physical seats; t117 and t247 on every trajectory, retaining all outcomes and fallback roots.',
             protocol='One warm native v4 request at previous step to initialize costs. Snapshot action chronology from each seat replay. Live 0.58s, temperature 1, both bytes unchanged, original-vs-candidate ABBA (BAAB alternate), same private RNG, belief, and native starting cost. On every one of the 16 roots, a second diagnostic forces parent first-call achieved K/H/rolls, and checks the parent itself is exactly reproduced, then compares two candidate and parent repeats. Count as a mechanism change if the live parent and candidate repeat, they disagree, and both agree with their equal-work answers; report all denominators and reasons. Count opponent scenario stability separately. Real wall results and forced-work-only changes separated; no full games.',
             count_all_issued=True,online_claim=False)
    dest=HERE/'protocol.json'; data=(json.dumps(out,indent=2,allow_nan=False)+'\n').encode()
    if dest.exists() and dest.read_bytes()!=data:raise ValueError('Frozen protocol differs')
    dest.write_bytes(data);print(sha(data),len(tickets))

if __name__=='__main__':main()
