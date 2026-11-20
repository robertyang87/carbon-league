"""Read all 7 physically verified economic extinctions in the frozen 1,324-game cohort."""
from pathlib import Path
import argparse,ast,csv,gzip,hashlib,json
from extinction import HERE,build
def sha(x):return hashlib.sha256(x).hexdigest()
def methods(src,mod,me,cfg):
    outer=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='_search')
    nodes=[s for s in outer.body if (isinstance(s,ast.FunctionDef) and s.name in {'extinct','over','result'})
           or (isinstance(s,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'recruit_floor','episode_steps'} for t in s.targets))]
    root=dict(configuration=cfg,me=me,op=1-me,LAST_TURN=mod.LAST_TURN,_terminal=mod._terminal)
    exec(compile(ast.Module(nodes,[]),'<native_terminal_functions>','exec'),root)
    return root['over'],root['result']
def main(cache):
    from probe import base,rr
    parent=HERE.parent/'cold_start/candidate.py';m=base.load('original_sim_end',parent)
    idx=HERE.parents[1]/'sapling_1200k/all_top5/index_games.tsv.gz'
    assert idx.exists()
    panel=list(csv.DictReader(gzip.open(idx,'rt'),delimiter='\t'));assert len(panel)==1324
    lst=[x for x in panel if x['validation_terminal']=='extinction'];assert len(lst)==7
    out=[]
    for entry in lst:
        raw=(cache/(entry['id']+'.json.gz')).read_bytes();assert sha(raw)==entry['raw_sha256']
        j=json.loads(gzip.decompress(raw));fr=j['steps'];cfg=j['configuration']
        act=[v['action'] or {} for v in fr[-1]];prior=fr[-2][0]['observation'];actual=fr[-1][0]['observation']
        # Feed true carbon solely to the rule replica for an equality check, never to a playing policy.
        simulator_root={k:v for k,v in prior.items() if k!='full_carbon'};simulator_root['carbon']=prior['full_carbon']
        child=m._turn(simulator_root,cfg,act)
        assert [p[0] for p in child['players']]==[p[0] for p in actual['players']]
        for s in (0,1):
            assert len(child['players'][s][2])==len(actual['players'][s][2]) and len(child['players'][s][3])==len(actual['players'][s][3])
        scores=[fr[-1][i]['reward'] for i in (0,1)];prior_rewards=[fr[-2][i]['reward'] for i in (0,1)]
        assert prior_rewards==[p[0] for p in prior['players']]
        b=[0,0,child,prior['players']];traces=[]
        for s in (0,1):
            old_over,old_result=methods(parent.read_bytes(),m,s,cfg);fixed_over,fixed_result=methods(build(),m,s,cfg)
            expected=m._terminal(scores[s]-scores[1-s])
            assert not old_over(b) and fixed_over(b) and fixed_result(b)==expected
            traces.append(dict(seat=s,stock_over=old_over(b),forced_stock_wallet_value=old_result(b),repaired_over=fixed_over(b),
                               repaired_value=fixed_result(b),actual_value=expected))
        k=entry['id'];out.append(dict(id=k,sha256=sha(raw),t=prior['step'],team=[entry['seat0_name'],entry['seat1_name']],
                prior_rewards=prior_rewards,post_wallet=[p[0] for p in child['players']],official_scores=scores,
                after_workers=[len(p[2]) for p in child['players']],after_trees=[len(p[3]) for p in child['players']],last_commands=act,calls=traces))
    rr.save(HERE/'extinction_check.json',dict(index_sha256=sha(idx.read_bytes()),registry_games=1324,extinctions=7,rows=out,
            source_parent=sha(parent.read_bytes()),repair=sha(build())))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('cache',type=Path)
    main(ap.parse_args().cache)
