"""Freeze all archived contrast families and scoring before obtaining the new network outputs."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
DER=HERE.parents[1]
LEGACY=DER/'parallel_responses/terminal_teacher'
sha=lambda b:hashlib.sha256(b).hexdigest()

def register():
    registrations = [
        LEGACY/'protocol.json',LEGACY/'score.json',DER/'tree_action_outcomes/protocol.json',DER/'tree_action_outcomes/settlement.json',
        DER/'parallel_responses/terminal.json',DER/'parallel_responses/equal_work/terminal.json',
        DER/'v3_release/development/return_turns/621758607_0_290_result.json',
        DER/'v3_release/development/return_turns/621758607_1_271_result.json']
    old=json.loads((LEGACY/'protocol.json').read_bytes())['tickets']
    keys=[(r['seed'],r['seat'],r['step'],r['group']) for r in old if r['status']!='skipped']
    keys += [(621758592,0,272,'tree'),(621758593,0,273,'tree'),(621758594,0,277,'tree'),(621758595,0,272,'tree'),
             (621758604,0,293,'width'),(621758606,0,293,'width'),(621758607,1,293,'mean'),(621758604,1,282,'mean'),
             (621758607,0,290,'block'),(621758607,1,271,'block')]
    assert len(keys)==len(set((x[:3] for x in keys)))==25
    srcdict=json.loads((HERE.parent/'identity.json').read_bytes())
    doc=dict(version=1,parent_sha=srcdict['parent_sha256'],child_sha=srcdict['candidate_sha256'],
        source_sha256={str(p.relative_to(DER)):sha(p.read_bytes()) for p in registrations},
        keys=keys,primary='For each archived fixed A/B contrast, sum unguarded joint actor log-odds log p(B_u)/p(A_u) over all differing acting-seat entities. At T1. Stable log-softmax. Native and rotation separately prefer larger score (tie A). Orient change in relative support toward higher realized own official terminal margin; exact tie is separate. Pool map-equal and per cohort, bootstrap 10000 physical maps seed 162801. These are proposal factor odds, not true final joint policy probability. Isolate current cargo, empty collectors, planters and bases and each direction contributing to odds.',
        secondary='Exactly 128 common RNG root decoder draws per physical root, no search; rng per draw is low 31 bits of unsigned sha256(UTF8 180pn/sample/seed/seat/t/index), first 8 hex digits. Report empirical A,B,other; third actions have unknown reward and must not be counted as zero-outcome. Outcome of mode choice only if choice is among documented options. Count whether either branch can change real win/loss/draw, what would change when choosing larger raw odds, and actual probability mass at the one root that can cross wins. Explicit comparison over two known orders and prior 25 root selection, never extrapolate to win rate. Original, held, close and other historically exposed labels known at registration; no training or tuning.',
        execution='Both arms only read observation and actually executed own actions before root, own current public observation, same unchanged model trunk and dimensions. No archive future data or actual simultaneous enemy command fed to policy. Exact live own_root neural functions feed original sampling/go_home/joint occupancy, before search. Both native inputs and actions retain physical dictionary order; authorization root diagnostics only, no new full game.')
    p=HERE/'protocol.json';p.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
    print(sha(p.read_bytes()))
if __name__=='__main__':register()
