"""Screen published v1 training transcripts, preserving physical-seat observations and genuine parent commands."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
import types

from policy import LateRecruit

HERE=Path(__file__).resolve().parent
T=HERE.parent/'tree_response_training'
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'code'))
from fastsim import FastCarbon
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())


def save(p,data):
    raw=(json.dumps(data,indent=2,allow_nan=False,ensure_ascii=False)+'\n').encode()
    if p.suffix=='.gz':raw=gzip.compress(raw,mtime=0)
    if p.exists() and p.read_bytes()!=raw:raise ValueError(f'changed outcome {p}')
    p.write_bytes(raw)


def registered():
    return [T,*[T/'batches'/f'intake_0{k}' for k in range(2,6)],*[T/'supplement_01/batches'/f'intake_0{k}' for k in (6,7)]]


def scan():
    trials=[];all_rows=[];checks=0
    parent=read(HERE/'protocol.json')['parent_sha256']
    for d in registered():
        s=read(d/'source_receipt.json');h=read(d/'source/results.json');official=read(d/'source/official_replay.json')['runs']['source']
        if (h['header']['pack_sha256']!=parent or h['status']!='complete' or len(s['games'])!=4 or sha(d/'source/official_replay.json')!=s['official_sha256']
                or h['header']['examiner_md5']!='57f9a594cce4f2ec9e3573d8d9d26ca1'):raise ValueError('wrong native source')
        for g in s['games']:
            path=ROOT/g['raw_path'];raw=read(path);seat=g['seat'];seed=g['seed'];cfg=raw['game_configuration']
            match=[o for o in official['games'] if o['seed']==seed and o['seat']==seat]
            if len(match)!=1 or g['raw_sha256']!=match[0]['transcript_sha256'] or sha(path)!=g['raw_sha256'] or match[0]['official_reward_by_seat']!=g['scores']:
                raise ValueError('native seat evidence differs')
            sim=FastCarbon(cfg).reset(seed=seed);state=LateRecruit();rows=[];reasons=Counter();total=0;late=0
            for turn,pair in enumerate(raw['actions']):
                obs=sim.obs(seat);cmd=pair[seat];alt,event=state.propose(obs,cfg,cmd)
                if 235<=turn<=285:
                    total+=1
                    if 'RECPLANTER' in cmd.values():
                        late+=1;reasons[event['reason']]+=1
                        record=dict(event,seed=seed,source_sha256=g['raw_sha256'],old=cmd,new=alt)
                        rows.append(record)
                sim.step(*pair)
            checks+=len(raw['actions'])
            opportunities=[r for r in rows if r['reason']=='proposed']
            trials.append({'seed':seed,'seat':seat,'path':g['raw_path'],'raw_sha256':g['raw_sha256'],'terrain':g['map_sha256'],
                           'steps_checked':len(raw['actions']),'late_turns':total,'ordered_planter_turns':late,'proposal_turns':len(opportunities),
                           'first_proposal_t':opportunities[0]['step'] if opportunities else None,'reasons':dict(reasons)})
            all_rows.extend(rows)
    keys={(a['seed'],a['seat']) for a in trials}
    if len(keys)!=28 or len({a['terrain'] for a in trials})!=14:raise ValueError('repeated episode or terrain')
    by_seat={v:sum(r['proposal_turns']>0 for r in trials if r['seat']==v) for v in (0,1)}
    save(HERE/'recruit_orders.json.gz',all_rows)
    stats={'protocol_sha256':sha(HERE/'protocol.json'),'source_games':len(trials),'source_boards':14,'source_steps':checks,
           'turns_in_window':sum(a['late_turns'] for a in trials),'raw_planter_orders':sum(a['ordered_planter_turns'] for a in trials),
           'proposal_turns':sum(a['proposal_turns'] for a in trials),'games_with_proposal_by_seat':by_seat,
           'reasons':dict(sum((Counter(r['reasons']) for r in trials),Counter())),
           'evidence_sha256':sha(HERE/'recruit_orders.json.gz'),'games':trials}
    save(HERE/'screen.json',stats);print({k:v for k,v in stats.items() if k!='games'},flush=True)


def policy_ns(path):
    name='archive_check_'+path.stem
    ns={'__name__':name,'__file__':str(path)}
    exec(compile(path.read_bytes(),str(path),'exec'),ns);return ns


def replay():
    stats=read(HERE/'screen.json')
    # Preselected rule: first game with a raw proposal in registered order for each seat.
    # When none exists for a seat, replay its first game to verify parent parity while inactive.
    games=[]
    for seat in (0,1):
        same=[r for r in stats['games'] if r['seat']==seat];games.append(next((r for r in same if r['proposal_turns']),same[0]))
    proof=[];model=read(HERE/'manifest.json')
    for row in games:
        d=read(ROOT/row['path']);s=row['seat'];cfg=d['game_configuration'];sim=FastCarbon(cfg).reset(seed=row['seed'])
        n=policy_ns(HERE/'agent.py');edits=[];parent_changes=0;blocked=0
        parent_out=[];original_select=n['_lw_select']
        def capture(x,aux,enc,labels,branch):
            parent_out.append(n['_FEAT'].to_actions(labels).copy())
            return original_select(x,aux,enc,labels,branch)
        n['_lw_select']=capture
        for t,actions in enumerate(d['actions']):
            mine=actions[s];got=n['agent'](sim.obs(s),cfg)
            if len(parent_out)!=1 or parent_out.pop()!=mine:raise ValueError(f'parent decode differs on {row["seed"]}/{s}/{t}')
            ev=n['_lw_observer'].event;blocked+=ev['reason']=='prefix_redecode_rejected'
            if got!=mine:
                changed=[k for k in set(mine)|set(got) if mine.get(k)!=got.get(k)]
                if len(changed)!=1 or mine[changed[0]]!='RECPLANTER' or got[changed[0]]!='RECCOLLECTOR' or ev['reason']!='one_hire_replaced':
                    raise ValueError('different physical action or logging mismatch')
                edits.append(dict(ev))
            sim.step(*actions)
        if len(edits)>1:raise ValueError('repeated intervention')
        # _COUNTS is from unperturbed v1 root decoding, candidate recomputation does not update it.
        attribution={k:dict(v) for k,v in n['_COUNTS'].items()}
        proof.append(dict(seed=row['seed'],seat=s,steps=len(d['actions']),raw_sha256=row['raw_sha256'],first_geometric=row['first_proposal_t'],
                          valid_prefix_edits=edits,rejected_prefixes=blocked,student_attribution_unchanged_parent=attribution))
    save(HERE/'package_replay.json',{'candidate_sha256':model['policy_sha256'],'protocol_sha256':sha(HERE/'protocol.json'),
         'screen_evidence_sha256':sha(HERE/'screen.json'),'games':proof,
         'public_mechanism_both_seats':all(r['valid_prefix_edits'] for r in proof),'new_native_games':0})
    print([(r['seed'],r['seat'],len(r['valid_prefix_edits'])) for r in proof],flush=True)


if __name__=='__main__':
    if sys.argv[1:] == ['replay']:replay()
    else:scan()
