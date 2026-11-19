"""Fixed complete-game neural finite-difference study, before seeing any outcome."""
import hashlib,json,subprocess,sys
from pathlib import Path
H=Path(__file__).resolve().parent; A=H.parents[1]; root=A.parents[2];w=root/'data/weights/sapling_1200k'
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
src=lambda p:{'path':str(p.relative_to(root)),'sha256':digest(p)}
def main():
    train=[621896120,621896121,621896122];held=[621896220,621896221,621896222,621896223]
    p={'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'train':train,'holdout':held,'seeds':train+held,'seats':[0,1],
       'parent':src(w/'terminal_return_fit_v1.py'),'opponent':src(A/'actor_partner_dev/candidate.py'),'student_start':src(w/'master6_actor_v1.npz'),
       'perturbation':'Use original FP32 whead and bhead separately, append bias as last column, center across actions. Directions 0/1 are largest two worker singular components, direction 2 largest recruitment component (role-specific contextual rank-one weight axes). Scale by singular value. Antithetic radius .50; final radius .30. Only four original tensors can change. Exact current deployment, expert and return guards fixed.',
       'objective':'Maximize official terminal U = .5 WDL + .5 sigmoid(margin/1000) from turn zero to official time limit or genuine extinction, map and both seats equal; fixed greedy reacting it20 c5ad3f43, same physical terrain/seat/opponent for +/- and parent/candidate.',
       'rule':'In each neural direction use mean (Uplus-Uminus)/(2*.50) across the 3 maps and paired seats, giving g with 3 entries. If norm(g)>1e-8 choose sole update .30*g/norm(g), else retain exact parent. Freeze candidate before any held result. Log center/update on training plus fixed independent held result. No tuning.',
       'held_gate':'On four held map pairs require positive average U, at least 3 maps positive, neither seat negative on average and no additional extinction. Preserve wins/losses, score, margin and durations. No original Master3s2 games.',
       'behavior_gate':'Counterfactual same observed state with each actual policy public entry at 0,20,40,60,80,100,125,150,175,200,225,250,275,297 on all training center trajectories. At least one changed whole-team command per phase t0-99/100-199/200-298 for full-phase coverage. Complete held evaluation even if behavior gate fails.',
       'cost':'36 antithetic training games, 6 center + 6 updated train calibration and 16 held games = 64 full proxy games. CPU3 one thread; separate source interpreter checks every official terminal.',
       'neighbors':{'late_root':src(H.parent/'protocol.json'),'full_episode_handcrafted_not_neural':src(A/'terminal_return_fit/README.md'),'late_root_trainable_Sapling':src(A/'blended_policy_return_dev/README.md')}}
    with (H/'protocol.json').open('x') as f: json.dump(p,f,indent=2);f.write('\n')
    sys.path.insert(0,str(A/'first_win_dev'));import native
    native.PLAN=H/'protocol.json';native.REG=digest(native.PLAN);native.HERE=H
    clearance=native.clearance()
    assert set(map(int,clearance['physical_boards']))==set(p['seeds'])
    old=json.loads((H.parent/'protocol.json').read_bytes())
    anc=json.loads((A/'broader_elite_backbone_dev/panel.json').read_bytes())
    used=set(old['train_seeds']+old['development_seeds']); fields={native.map_identity.local_board(s) for s in used}
    for row in anc['games']:used.add(row['seed']);fields.add(row['terrain_sha256'])
    assert not (set(p['seeds'])&used or set(clearance['physical_boards'].values())&fields)
    with (H/'register.json').open('x') as f:json.dump(dict(protocol=digest(native.PLAN),additional_blocked_seeds=len(used),additional_blocked_terrains=len(fields),seeds_checked=p['seeds'],boards=clearance['physical_boards']),f,indent=2); f.write('\n')
if __name__=='__main__':main()
