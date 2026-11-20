"""Fix the physical maps, work comparison, outcome sampling and provenance before measuring search."""
import json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
FAMILY=HERE.parent
sha=lambda b:hashlib.sha256(b).hexdigest()
PARENT=FAMILY/'ladder/prediction_feedback/package/candidate.py'
ORIGINAL=FAMILY.parents[1]/'data/master3s2_source.py'
def register():
    tickets=[]
    for source,low in [('identity_replication_dev',621758604),('v3_release/development',621758606)]:
        folder=FAMILY/source/'hosted'
        settlement=json.loads((folder/'settlement.json').read_bytes())
        for arc in settlement['archives']:
            if arc['seed'] not in (low,low+1):continue
            g=next(g for g in settlement['games'] if (g['arm'],g['seed'],g['seat'])==('parent',arc['seed'],arc['seat']))
            assert g['status']=='complete'
            tickets.append(dict(seed=arc['seed'],seat=arc['seat'],terrain=g['terrain'],source=str((folder/f"{arc['id']}.zip").relative_to(FAMILY)),
                                sha256=arc['sha256'],registration_receipt_sha=g['receipt_sha256'],run=settlement['workflow_run']))
    tickets.sort(key=lambda r:(r['seed'],r['seat']))
    assert len(tickets)==len({(r['seed'],r['seat']) for r in tickets})==8
    p=dict(parent=sha(PARENT.read_bytes()),examiner=sha(ORIGINAL.read_bytes()),tickets=tickets,steps=[95,188,282,293,298],
           design='Known-map diagnostics only; every parent trajectory both seats in the declared four maps. Single common calibrated cost vector from M1 at the immediately preceding turn for each root; alternating arm order by ordinal. Fresh common salted policy random state at each target. Original .58s budget without monkeypatching, single pinned thread. M1=(1,1) and M2=(2,2) native parent. Two scenarios draw their own and enemy subsequent orders, not just the enemy. No fresh games.',
           work='Keep candidate menu, both sampled enemy root orders, surviving indices, each arm value and depth for each rollout. Recompute roll0 winner within the exact M2 survivors at same scored depth for second-sample decision effect. Log separate adaptive differences against actual M1 action. Retain all timeouts and budget differences.',
           outcomes='No future orders seen by methods. Held-out historical original opponent real simultaneous action for first transition, all eight t298 official terminals. For t293 select at most the lowest two distinct map seeds with a differing issued command, lower seat wins ties, increasing order. Continue both branches with same unmodified parent M1 and original opponent acting naturally on new states to official terminal. Reverse branch order at second root. Report all outcomes, single root nonrepresentative of a win-rate change. Show differences in M1-vs-real opponent unit action and M2 mixture unit score as supporting metrics; min of two is explicitly oracle.')
    (HERE/'protocol.json').write_text(json.dumps(p,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':register()
