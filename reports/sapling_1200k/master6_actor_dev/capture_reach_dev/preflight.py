"""Recover candidate and parent commands at the five previously recorded interventions."""
from match import HERE, PARENT, CAND, native
previous=HERE.parent/"planter_capture_full_dev"
refs=native.read(previous/"audit.json")
rows=[]
for seat in [0,1]:
    games=sorted((previous/"games/candidate").glob(f"raw_*_{seat}.json.gz"))
    raw=native.read(games[0])
    target={r["step"]:r for r in refs[seat]["changes"]}
    sim=native.exam.FastCarbon(native.exam.CFG).reset(seed=raw["seed"])
    cand=native.module(CAND,"reach_preflight_"+str(seat))
    for t,commands in enumerate(raw["actions"][:max(target)+1]):
        obs=sim.obs(seat)
        if t==0 or t in target:
            actual=cand.agent(obs,sim.config)
            if t in target:
                r=target[t];uid=r["unit"]
                assert commands[seat].get(uid)==r["actual"]
                rows.append(dict(seat=seat,step=t,unit=uid,parent=r["actual"],candidate=actual.get(uid)))
        else:
            cand._FEAT.observe(obs)
        sim.step(*commands)
assert any(r["parent"]!=r["candidate"] for r in rows)
native.save(HERE/"preflight.json",rows)
