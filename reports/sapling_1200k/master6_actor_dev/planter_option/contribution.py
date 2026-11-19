"""Independently run the exact submission through each first trigger and remove Sapling on the same inputs."""
from collections import Counter
from screen import HERE,ROOT,FastCarbon,read,retain,live,clean,sha

def audit():
    policy_path=HERE/'agent.py'
    mechanism=read(HERE/'mechanism.json')
    if sha(policy_path.read_bytes())!=mechanism['package_sha256']:
        raise ValueError('package mismatch')
    reports=[]
    for reg in mechanism['selected_roots']:
        raw=read(ROOT/reg['source']);root=read(HERE/reg['root'])
        sim=FastCarbon(raw['game_configuration']).reset(seed=reg['seed'])
        complete=live.module(policy_path,'complete_plant_option')
        ablated=live.module(policy_path,'no_student_plant_option')
        if (complete._HYBRID['expert_weight'],complete._HYBRID['student_weight'])!=(.75,.25):
            raise ValueError('unexpected policy mixing')
        ablated._HYBRID['expert_weight']=1.;ablated._HYBRID['student_weight']=0.
        changed=[];roles=Counter();seat=reg['seat']; root_diff=[]
        for t,joint in enumerate(raw['actions'][:reg['t']+1]):
            obs=sim.obs(seat);cfg=raw['game_configuration']
            moves=clean(complete.agent(obs,cfg));disabled=clean(ablated.agent(obs,cfg))
            expected=root['root_option_joint'] if t==reg['t'] else clean(joint[seat])
            if moves!=expected or bool(complete._OPTION.last['committed'])!=(t==reg['t']):
                raise ValueError(f'actual submission differs at {reg["seed"]} turn {t}')
            differences=sorted(k for k in moves.keys()|disabled.keys() if moves.get(k)!=disabled.get(k))
            if differences:
                changed.append(t)
            if t==reg['t']:
                root_diff=differences
            for uid in differences:
                roles['base' if uid in obs['players'][seat][1] else obs['players'][seat][2][uid][2].lower()]+=1
            sim.step(*joint)
        reports.append({'seed':reg['seed'],'seat':seat,'steps_checked':reg['t']+1,
                        'first_deployed_divergence_t':reg['t'],'source_action_tape_sha256':reg['source_sha256'],
                        'changed_roots_if_sapling_disabled':changed,'role_commands_changed':roles,'root_units_different_under_ablation':root_diff})
    if not all(r['changed_roots_if_sapling_disabled'] for r in reports):
        raise ValueError('no independent Sapling contribution in a tested trajectory')
    retain(HERE/'contribution.json',{'package_sha256':mechanism['package_sha256'],'games':reports})
    print([(r['seed'],len(r['changed_roots_if_sapling_disabled']),r['steps_checked']) for r in reports])

if __name__=='__main__':
    audit()
