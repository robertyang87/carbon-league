"""Check delivered dual-network packages on recorded public states, plus switch and structural contrasts."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import learning as L

ACTOR=L.STUDY.parent
ROOT=ACTOR.parents[2]
sys.path.insert(0,str(ROOT/"reports/sapling_1200k/root_recovery_branches"))
from branches import begin,public


def main():
    # Prior registered internal fitting terrain, disjoint from the eight new tickets.
    names=list((ACTOR/"tree_response_training").rglob("raw_621472120_0.json.gz"))
    L.check(len(names)==1,"Prior source is not unique")
    src=L.read(names[0])
    env=begin(src)
    parents=[L.module(p) for p in (L.BC,L.ELITE)]
    scenarios=[([1.,0.,0.,0.],256),([0.,1.,0.,0.],280),([-1.,0.,0.,0.],None)]
    observed={}
    with TemporaryDirectory() as temp:
        variants=[]
        for i,(coef,t) in enumerate(scenarios):
            p=Path(temp)/f"s{i}.py"
            L.build(coef,p)
            variants.append(L.module(p))
            observed[str(t)]=dict(local_package_bytes=p.stat().st_size,total=0,elite=0,old_new_disagreements_inside=0)
        for t,orders in enumerate(src["actions"]):
            q=public(env,0)
            baselines=[m.agent(q,src["game_configuration"]) for m in parents]
            for (coef,start),m in zip(scenarios,variants):
                using=int(start is not None and start<=t<start+8)
                L.check(m.agent(q,src["game_configuration"])==baselines[using],"Ordinary public gate order differs")
                L.check((m._NETS[1] is m._WIN_ELITE_NET)==bool(using),"Window stored wrong network")
                counts=observed[str(start)]
                counts["total"]+=1;counts["elite"]+=using
                counts["old_new_disagreements_inside"]+=int(using and baselines[0]!=baselines[1])
            env.step(orders)
        for (coef,start),m in zip(scenarios,variants):
            q=public(begin(src),0)
            L.check(m.agent(q,src["game_configuration"])==parents[0].agent(q,src["game_configuration"]) and m._WIN_AT is None,
                    "State does not reset in the next game")
    demo=[dict(physical=12.,margin=100.),dict(physical=98.,margin=-10.)]
    predictions=[{"elite":True},{"elite":True}]
    z=L.estimate(demo,predictions,256,"margin")
    L.check(z["single_vs_bc"]==100 and z["single_vs_elite_280"]==110 and z["both_roots_vs_elite"]==0,"Window effects were summed or crossed")
    L.save(L.HERE/"fixture.json",dict(source=names[0].relative_to(ROOT).as_posix(),source_sha256=L.sha(names[0]),runtime=L.sha(L.HERE/"runtime.py"),
            coefficient_fixture_only=True,observed=observed,new_games=0,runner_time_certificate=False))


if __name__=="__main__":
    main()
