"""Freeze return-heavy, pre-action public roots before inspecting any ranker prediction."""
import hashlib
import gzip
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
FAMILY = HERE.parents[1]
PARENT = '7cf5c6c91fc59b46b2a38924c4c5ab56143fb1242d082f9426303af7336ac3ed'
ORIGINAL = '65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5'
WINDOWS = [(80, 150), (150, 220), (220, 280)]
LANES = ('prediction_feedback', 'cargo_defense')


def sha(b):
    return hashlib.sha256(b).hexdigest()


def distance(a,b):
    return sum(min(abs(i-j),15-abs(i-j)) for i,j in zip(divmod(a,15),divmod(b,15)))


def opportunity(obs):
    own,other = obs['players'][obs['player']],obs['players'][1-obs['player']]
    workers=[]
    for uid,(pos,load,role) in own[2].items():
        home=min(distance(pos,b) for b in own[1].values())
        if role!='COLLECTOR' or load<100 or not 2<=home<=4:
            continue
        threats=sum(w[2]=='COLLECTOR' and w[1]<=load and distance(pos,w[0])<=3 for w in other[2].values())
        workers.append((load,uid,pos,home,threats))
    if not workers:return None
    load,uid,pos,home,threats=min(workers,key=lambda x:(-x[0],x[1]))
    return dict(uid=uid,position=pos,cargo=load,home_distance=home,nearby_lighter_collectors=threats)


def generate():
    manifest=(HERE.parent/'train_manifest.json').read_bytes(); fit=json.loads(manifest)['tickets']
    assert sha(json.dumps(fit,sort_keys=True,separators=(',',':')).encode())=='0bc6178f65b91767f448f6030f0fdb6df58a92e6284e4c4609c12c5d49be8171'
    blocked={r['terrain'] for r in fit}; seeds={r['seed'] for r in fit}
    result=[]; sources={}
    for lane in LANES:
        folder=FAMILY/'ladder'/lane/'development';plan=json.loads((folder/'protocol.json').read_bytes());settled=(folder/'hosted/settlement.json').read_bytes();doc=json.loads(settled)
        assert plan['files']['parent']['sha256']==PARENT and plan['files']['opponent']['sha256']==ORIGINAL and doc['records_reconciled']
        sources[lane]=dict(physical_boards=plan['physical_boards'],settlement_sha256=sha(settled),run=doc['workflow_run'])
        for a in sorted(doc['archives'],key=lambda a:(a['seed'],a['seat'])):
            seed,seat=a['seed'],a['seat'];terr=plan['physical_boards'][str(seed)]
            assert seed not in seeds and terr not in blocked
            path=folder/'hosted'/f"{a['id']}.zip";assert sha(path.read_bytes())==a['sha256']
            with zipfile.ZipFile(path) as z: public=z.read('parent/observations.jsonl.gz')
            frames=[json.loads(l) for l in gzip.decompress(public).splitlines()]
            for lo,hi in WINDOWS:
                options=[]
                for t in range(lo,min(hi,len(frames))):
                    obs=frames[t]['observations'][seat];assert obs['step']==t and obs['player']==seat
                    w=opportunity(obs)
                    if w: options.append((min(2,w['nearby_lighter_collectors']),w['cargo'],-t,w))
                ticket=dict(ordinal=len(result),lane=lane,seed=seed,terrain=terr,seat=seat,window=[lo,hi],archive_id=a['id'],
                            archive_sha256=a['sha256'],public_stream_sha256=sha(public),eligible_pre_action_turns=len(options))
                if options:
                    _,_,nt,w=max(options,key=lambda item:item[:3]);t=-nt
                    ticket.update(step=t,public_observation_sha256=frames[t]['public_observation_sha256'][seat],physical_opportunity=w)
                result.append(ticket)
    assert len(result)==24 and len({r['terrain'] for r in result})==4
    return dict(schema='return-network-single-root-real-response-preselection/1',parent_sha256=PARENT,original_sha256=ORIGINAL,
                training_exclusion_sha256=sha(manifest),train_or_fit_map_intersections=[],method='Four previously archived maps held away from all 64 learner maps. Two historical parent-vs-original seats per map and three predeclared time windows. No historical root commands, final scores or return-model predictions are used for selection. Within each interval favor largest number (capped at two) of lighter/equal enemy collectors within three of the heaviest eligible carrier, then cargo, then earliest turn. Include every empty interval.',
                physical_test='Each selected root gets one full parent, two enabled, one disabled public decision with identical reconstructed public history, private seed 19073+7*t+seed+seat and native .58s search, alternated P/E/E/D and D/E/E/P. A qualifying change requires matched parent=disabled, both enabled repeated, enabled physically selects the new offer, completed depth, menu width and rollouts agree across arms; log all exclusions. No reading future enemy actions. No tuning or replacement.',
                continuations='If a root qualifies, run at most two paired responses, first qualifying ordinal per distinct map in fixed ascending ticket order. Root original opponent acts once on the pre-action state and is shared. Both branches thereafter use the same frozen parent and unchanged original, complete to the official terminal with independent evolving observations and cloned stochastic/cost/belief states at the root. Reverse arm order on second selected map. Report both full outcomes, initial-carrier actual remittance and each return, early failures, root opportunities and nonqualifiers. Remaining qualifying roots are budget-censored, not scored as zero. All unchanged roots have zero single-root policy difference, not counterfactual sampled scores. This probe is a local diagnostic, not a whole-game/cgroup-qualified strength estimate.',
                sources=sources,roots=result)


if __name__=='__main__':
    payload=(json.dumps(generate(),sort_keys=True,indent=2)+'\n').encode(); dest=HERE/'root_protocol.json'
    if dest.exists():assert dest.read_bytes()==payload
    else:dest.write_bytes(payload)
    print(sha(payload))
