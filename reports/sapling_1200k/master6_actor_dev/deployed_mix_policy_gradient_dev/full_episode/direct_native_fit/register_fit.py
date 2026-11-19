"""Register a single signed direct-original finite difference before reading a challenger outcome."""
import hashlib,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;FULL=HERE.parent;ACTOR=FULL.parents[1];ROOT=ACTOR.parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return {'path':str(p.relative_to(ROOT)),'sha256':sha(p)}
def main():
    old=ACTOR/'actor_partner_dev'; probes=FULL/'direct_original'
    signed=[ref(old/'native/candidate'/f'raw_{s}_{i}.json.gz') for s in (621461321,621461322) for i in (0,1)]
    protocol={'center':ref(FULL/'selected.py'),'center_npz':ref(FULL/'selected.npz'),'upstream_selection':ref(FULL/'selection.json'),'upstream_economy':ref(FULL/'independent_audit.json'),
        'examiner':{'path':'sources/master3s2/master3s2.py','blob':'9c46f6fa4ff401b05939150054c2f88f549c2f7e','sha256':hashlib.sha256(subprocess.check_output(['git','cat-file','blob','9c46f6fa4ff401b05939150054c2f88f549c2f7e'],cwd=ROOT)).hexdigest()},'seeds':[621461321,621461322],'historic_native_tapes':signed,'previous_development':ref(old/'protocol.json'),
        'calibration_seed':621758343,'calibration_center_games':[ref(probes/'games/candidate'/f'raw_621758343_{s}.json.gz') for s in (0,1)],
        'excluded_recent':ref(FULL/'register.json'),'original_recent':ref(probes/'protocol.json'),
        'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'direction':'Use only the two shared worker readout singular components from the first episode study, retaining the original sign convention. In coefficient coordinates choose (−c1*||D1||², c0*||D0||², 0), normalized to unit coefficient norm, where c are the previous update coefficients. Its FP32 Frobenius inner product with the entire previous update is zero up to rounding. Base and all other learned tensors are fixed, as are the it50 partner and mixed policy logic. Neither original results nor proxy return signs tune this direction.',
        'radii':[.30,.50,.80],
        'radius_rule':'Read both actual center trajectories on map 343. Test every third step from 0 through 297 inclusive (100 roots per physical seat when game lasts the limit), retaining all prior seat-public observations. Try radii in the given order, with the same physical original snapshots in both signs. For each sign require >=3 joint differences vs center and <=30 percent changed overall; union of plus-vs-minus changes must include all three periods [0,100),[100,200),[200,299) and both seats. Choose the smallest passing radius; failure of all means retain center. Never read live training outcomes to choose a radius.',
        'scheduling':'Original public submission, unmodified 0.58s, temperature=1, independent natural RNG unseeded, exclusive native-clock lock, simultaneous physical orders, actual original official terminal. Historical maps permanently marked exposed/training here. Schedule alternates signed arm by core every wave and reverses seat/core associations across maps. Exactly 8 games, keep all failures, no new terrain.',
        'waves':[[['plus',621461321,0,0],['minus',621461321,1,1]],[['minus',621461321,0,0],['plus',621461321,1,1]],
                 [['plus',621461322,1,0],['minus',621461322,0,1]],[['minus',621461322,1,0],['plus',621461322,0,1]]],
        'selection':'Wins first. If total wins differ, consider ONLY the sign with more wins. If tied, choose direction from aggregate official seat-paired score-margin contrast; tie or zero leaves center. To retain either direction it must have no map with fewer wins, mean official score margin higher on BOTH maps, at least 3/4 individually positive margin pairs and no extra extinctions. When the sign has more total wins (and passes these requirements), retain the actually tested endpoint. If win counts tie, demand both map margin advantages >=250 points, then retain HALF of the tested radius, flagged as a new unplayed training candidate. Otherwise retain the byte-identical center. These 2 reused boards identify a direction, not evidence of strength vs the center or independent generalization. No further board is opened here.'}
    assert protocol['center']['sha256']=='09b41c24d0b478b8fea42a8919c98674841a690c20176d6992fd02bd0938ec01'
    assert protocol['examiner']['sha256']=='65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5'
    with (HERE/'protocol.json').open('x') as f:json.dump(protocol,f,indent=2);f.write('\n')
if __name__=='__main__':main()
