"""Bind independently verified on-support decisions to immutable training-only comparisons."""
from pathlib import Path
from collect import HERE, LEARN, K, R
from summarize import prior_pairs
from scan import compact as census_compact, public_frames

def digest(obj):
    return K.digest(K.compact(obj))

def strip_clock(obs):
    return {k:v for k,v in obs.items() if k!='remainingOverageTime'}

def freeze():
    base=LEARN/'active_roots'
    src=base/'roots.json';ver=base/'verification.json'
    proposals=R.load(src);verified=R.load(ver)
    assert verified['roots_sha256']==R.sha(src)
    assert len(proposals['cases'])==len(verified['cases'])==4
    cards={c['seed']:c for c in R.load(HERE/'protocol.json')['training_cards']}
    outputs=[]
    for case,checks in zip(proposals['cases'],verified['cases'],strict=True):
        seat=case['seat'];t=case['t'];uid=case['focal_uid'];card=cards[case['seed']]
        assert card['terrain']==case['terrain'] and card['sha256']==case['source_sha256'] and card['split']=='train'
        assert (checks['seed'],checks['seat'],checks['t'],checks['focal_uid'])==(card['seed'],seat,t,uid)
        prob=checks['forced_final_native_prior']['p'];rank=checks['forced_final_native_prior']['rank']
        diff=checks['contrast'];role=checks['role']
        contrast=max(abs(diff[0]),abs(diff[2])) if role=='collector' else abs(diff[1])
        assert contrast>=1 and prob>=.08 and rank<=3 and checks['no_new_conflicts'] and not checks['previous_signed_root_overlap']
        own,alt=case['incumbent'],case['alternative']
        assert set(k for k in set(own)|set(alt) if own.get(k,'NONE')!=alt.get(k,'NONE'))=={uid}
        assert checks['full_action_sha256']==[digest(own),digest(alt)]
        raw=R.load(R.ROOT/card['path']);env=K.checked_env(raw)
        record=R.load(HERE/(str(card['seed'])+'.json.gz'))['seats'][seat]['all_public_commands']
        public_hashes=[];normal={n:R.live.module(path,'training_preflight_'+n) for n,path in [('parent',K.PARENT),('student',K.STUDENT)]}
        for step,(states,_,joint,cfg) in enumerate(public_frames(card)):
            own_public=K.public(env,seat);opp=K.public(env,1-seat)
            assert strip_clock(own_public)==strip_clock(states[seat]), (card['seed'],seat,step)
            assert strip_clock(opp)==strip_clock(states[1-seat])
            assert K.digest(census_compact(states[seat]))==record[step][0]
            actions=[K.own_call(m,own_public,cfg)[0] for m in normal.values()]
            assert actions==record[step][1:]
            public_hashes.append([digest(own_public),digest(opp)])
            if step==t:
                assert digest(strip_clock(own_public))==case['public_obs_sha256']==checks['public_sha256']
                assert actions==[own,case['free_student']]
                assert joint[1-seat]==case['root_enemy']
                break
            env.step(joint)
        assert digest(raw['actions'][:t])==case['prefix_sha256']
        root_id=digest([card['terrain'],seat,t,case['public_obs_sha256'],own,alt])
        outputs.append(dict(root_id=root_id,mode='active_supported_proposal',card=card,seat=seat,t=t,focal_uid=uid,role=role,
                public_sha256=case['public_obs_sha256'],full_public_hashes=public_hashes,historical_prefix_sha=case['prefix_sha256'],
                parent=own,student=alt,expected_free_student=case['free_student'],historical_enemy=case['root_enemy'],
                parent_sha=digest(own),student_sha=digest(alt),selection_path=str(src.relative_to(R.ROOT)),
                selection_sha256=R.sha(src),verification_path=str(ver.relative_to(R.ROOT)),verification_sha256=R.sha(ver),
                supported_contrast=contrast,all_signed_contrasts=diff,policy_probability=prob,policy_rank=rank,joint_legal_verified=True))
    duplicates=prior_pairs(outputs);R.write(HERE/'active_duplicates.json',duplicates)
    for c,dup in zip(outputs,duplicates['flags'],strict=True):
        assert c['root_id']==dup['root_id'] and dup['same_both_joint_orders']==0
        c['no_previous_identical_root_label']=True
    R.write(HERE/'batch.json',dict(roots=outputs,pairs=4,arms=8,max_simultaneous_cases=2,scope='training_only',independent_full_map_wins=0,
                arm_order=['parent, proposal','proposal, parent','parent, proposal','proposal, parent'],
                selection_type='A supported, one-worker policy proposal with the rest of the legal joint re-decoded. It need not be the free network argmax.',
                plan=R.sha(HERE/'collect_plan.json'),duplicate_check=R.sha(HERE/'active_duplicates.json'),
                selection_commit='4c259d0c',active_protocol=R.sha(base/'protocol.json'),verified=R.sha(ver),selection=R.sha(src)))
if __name__=='__main__':freeze()
