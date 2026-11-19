"""Read every train/val/held raw teacher opening, including those never selected as gradient examples."""
from collections import Counter
from shared import HERE,TRAIN,read,require,save,sha
from opening_transfer import CACHE,key


def signature(commands,locations):
    # IDs are stable physical-order labels, not destinations or relative directions.
    by_id=sorted(locations,key=lambda b:int(b.rsplit('-',1)[-1]))
    return '/'.join((commands.get(uid) or 'NONE') for uid in by_id)


def run():
    panel=read(TRAIN/'panel.json')['games'];previous=read(HERE/'opening_transfer.json.gz')
    watched={g['id']:g for g in previous['rows']};obs=[];summary={}
    for split in ('train','val','holdout'):
        gg=[r for r in panel if r['split']==split];by_slot=Counter();by_composition=Counter();models={a:Counter() for a in ('parent','control','candidate')}
        for r in gg:
            p=CACHE/(r['id']+'.json.gz');require(sha(p)==r['sha256'],'historical replay mismatch')
            z=read(p);s=r['seat'];locations=z['steps'][0][0]['observation']['players'][s][1]
            cmd=z['steps'][1][s]['action'] or {};require(len(locations)==2 and set(cmd)<=set(locations),'other commands')
            seq=signature(cmd,locations);by_slot[seq]+=1;by_composition[key(cmd)]+=1;row=dict(id=r['id'],seed=r['seed'],split=split,seat=s,
                physical_bases=locations,actual_commands=cmd,received_role_by_base_id=seq,full_replay_sha256=sha(p))
            if r['id'] in watched:
                e=watched[r['id']];require(e['logged_master6']==cmd,'teacher proof changed')
                row['model_actual_by_base_id']={a:signature(c,locations) for a,c in e['agent'].items()}
                for a,actions in row['model_actual_by_base_id'].items():models[a][actions]+=1
            obs.append(row)
        summary[split]=dict(maps=len(gg),actual_by_role_mix=dict(by_composition),actual_by_physical_base_id=dict(by_slot),
                            deployed_by_physical_base_id={a:dict(v) for a,v in models.items()} if split!='train' else None)
    result={'counts':summary,'ordering':'signed numeric suffix of the two physical friendly base IDs; both exact command dictionaries stored. 1C1P alone does not say which base recruited which worker.',
            'panel_sha256':sha(TRAIN/'panel.json'),'cases':obs}
    save(HERE/'opening_teacher.json.gz',result);print(summary)
if __name__=='__main__':run()
