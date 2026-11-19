from teacher import *
def student_cards():
    folder=ACTOR/'deployed_mix_policy_gradient_dev/full_episode';data=[];old={r['terrain'] for r in source_cards()}
    for split,sub,seeds in [('fit','train',[621896120,621896121,621896122]),('held','holdout',[621896220,621896221])]:
        for seed in seeds:
            for s in (0,1):
                f=folder/'games'/sub/f'selected_{seed}_{s}.json.gz';g=read(f);ident=g['identity'];assert ident['pack']==read(J/'binding.json')['parent'] and ident['terrain'] not in old
                data.append(dict(source=str(f.relative_to(ROOT)),sha=sha(f),terrain=ident['terrain'],seed=seed,seat=s,split=split))
    return data
def main():
    gate=read(HERE/'results.json.gz');assert gate['gate']
    cards=student_cards();plan=dict(gate=sha(HERE/'results.json.gz'),cards=cards,steps=[64,144,224],maximum_queries=90,
           design='Exactly 3 windows x 2 seats x 5 previously distinct maps. Use 09b4 traced action and its existing deployed self-rollout menu (A0+2 or 3 complete legal alternatives). Compare each alternative in an original A0/two-candidate search: native four-arm pruning would compare the full menu only at H1; require the same common H8 and rollout count, identical opponent roots and identical A0 score for all pairs. No dropped value imputation; skip entire incomplete root. Max one teacher search per alternative. Capture complete common rollout matrices. Fit three physical maps, hold two, never fit original 16 gate roots. Offline only.',
           scorer='One deterministic shared linear readout, 18 features: simulated H1 own-minus-opponent self-trained shared latent 16, simulated relative cash and old self leaf. Differences vs A0 only. Standardize by fit RMS with floor 0.001, ridge lambda 1.0 normalized by root count, equal candidate contribution within each root, no intercept, direct fit to native terminal-head mean difference. Send student top1 only if it beats always A0 in teacher-valued gain on BOTH held maps, improves total squared target error vs constant zero, and independently selecting on the 16 previously assessed historical roots has >8 correct and positive official mean gain. No new games/no threshold calibration.')
    bridge.save(HERE/'student_protocol.json',plan)
    cfg=read(ROOT/source_cards()[0]['source'])['game_configuration'];cold=read(ROOT/native.P['cases'][0]['warm_source'])['warm_cost']
    for card in cards:
        s=card['seat'];seed=card['seed'];f=HERE/f'labels_{seed}_{s}.json.gz'
        if f.exists():
            assert read(f)['protocol']==sha(HERE/'student_protocol.json');continue
        g=read(ROOT/card['source']);sim=bridge.FastCarbon(cfg).reset(seed=seed)
        assert hashlib.sha256(np.asarray(sim.carbon,np.uint8).tobytes()).hexdigest()==card['terrain']
        m=native.install(native.live.SOURCE,s,seed+s,cold,'restricted_original');ns,hist=instrument(m)
        from audit import module
        pkg=module(J/'candidate.py','student_menus');exec((HERE/'head_runtime.py').read_text(),vars(pkg))
        saved={};collect=pkg._head_rows
        def reader(cfg,obs,views,opp_x,commands,deadline):
            xx,en,term=collect(cfg,obs,views,opp_x,commands);saved.update(x=xx.tolist(),commands=commands,terminal=term);return 0,[],en
        pkg._joint_search=reader;recs=[];queries=0
        for t,a in enumerate(g['actions'][:225]):
            obs=sim.obs(s);obs={k:v for k,v in obs.items() if k in ('player','step','players','trees','carbon','remainingOverageTime')}
            if t in plan['steps']:
                saved.clear();base=pkg.agent(obs,cfg);assert base==a[s]
                row=dict(t=t,complete=False,public_sha=native.digest(obs))
                if len(saved.get('commands',[]))>=3:
                    tr=copy.deepcopy(m._ST);pairs=[];ok=True
                    for alt in saved['commands'][1:]:
                        m._ST.clear();m._ST.update(copy.deepcopy(tr));ns['forced_menu']=[base,alt];hist.clear()
                        wall=time.perf_counter();m.agent(obs,cfg);elapsed=time.perf_counter()-wall;queries+=1
                        snap=copy.deepcopy(hist[-1]) if hist else None
                        ok=bool(ok and snap and snap['depth']==8)
                        pairs.append(dict(final=snap,elapsed=elapsed))
                    ref=pairs[0]['final']
                    if ok:ok=all(z['final']['rolls']==ref['rolls'] and z['final']['enemy_by_stream']==ref['enemy_by_stream'] and np.allclose(z['final']['values'][0],ref['values'][0],rtol=0,atol=1e-6) for z in pairs)
                    row.update(**saved,pairs=pairs,complete=ok)
                    m._ST.clear();m._ST.update(tr)
                recs.append(row)
            elif t==0:pkg.agent(obs,cfg)
            else:pkg._FEAT.observe(obs);pkg._JB_FEAT.observe(pkg.public_enemy(obs,pkg._FEAT.est))
            native.historical(m,obs,a[s]);sim.step(*a)
        bridge.save(f,dict(protocol=sha(HERE/'student_protocol.json'),source=sha(__file__),root_data=recs,queries=queries,card=card))
if __name__=='__main__':main()
