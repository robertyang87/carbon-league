import json, glob, statistics as st

def side_of(md, target_name):
    slot = 'a' if md["strategies"]["a"]["name"]==target_name else 'b'
    return md["order"].index(slot), slot

def analyze(fname, target):
    d=json.load(open(fname)); md=d["metadata"]
    me, slot = side_of(md, target); op = 1-me
    opp_name = md["strategies"]['b' if slot=='a' else 'a']["name"]
    samples = d["samples"]  # [step, p0, p1, map_carbon]; player=[cash, collectors, planters, trees, carried]
    def traj(side):
        return {s[0]: s[1+side] for s in samples}
    tm, to = traj(me), traj(op)
    steps_sorted = sorted(tm)
    ev = d["events"]
    def evs(kind, side=None):
        return [e for e in ev if e[1]==kind and (side is None or e[2]==side)]
    rec_me = evs("recruit", me); rec_op = evs("recruit", op)
    plant_me = evs("plant", me); plant_op = evs("plant", op)
    seize_me = evs("seize", me); seize_op = evs("seize", op)
    dep_me = evs("deposit", me); dep_op = evs("deposit", op)
    death_me = evs("death", me); death_op = evs("death", op)
    def keypoints(tr):
        return {s: tr[s] for s in (0,10,30,50,100,150,200,250,270,290,299) if s in tr}
    tree_peak_me = max((v[3], s) for s,v in tm.items())
    tree_peak_op = max((v[3], s) for s,v in to.items())
    last30_dep_me = [e for e in dep_me if e[0]>=270]
    last30_dep_op = [e for e in dep_op if e[0]>=270]
    cross_dep_me = [e for e in dep_me if e[5]!=me]  # credited to other side
    def death_stats(ds):
        by={}
        carb=0
        for e in ds:
            by[f"{e[3]}|{e[6]}"]=by.get(f"{e[3]}|{e[6]}",0)+1
            if e[4] is not None: carb+=e[4]
        return by, round(carb,1)
    out = dict(
        file=fname, target=target, opp=opp_name, match=md["id"][:8],
        winner_slot=md["winner"], target_slot=slot, target_replay_side=me,
        target_win = md["winner"]==slot,
        score_me=md["scores"][slot], score_op=md["scores"]['b' if slot=='a' else 'a'],
        steps=md["steps"],
        timing_ms_per_step={k: round(v["total_seconds"]/md["steps"]*1000,2) for k,v in md["action_timing"].items()},
        recruit_seq_me=[(e[0],e[3]) for e in rec_me],
        recruit_seq_op=[(e[0],e[3]) for e in rec_op],
        n_recruit_me=(sum(1 for e in rec_me if e[3]=='collector'), sum(1 for e in rec_me if e[3]=='planter')),
        n_recruit_op=(sum(1 for e in rec_op if e[3]=='collector'), sum(1 for e in rec_op if e[3]=='planter')),
        first_plant_me=plant_me[0][0] if plant_me else None,
        first_plant_op=plant_op[0][0] if plant_op else None,
        n_plant_me=len(plant_me), n_plant_op=len(plant_op),
        last_plant_me=plant_me[-1][0] if plant_me else None,
        plant_steps_me=[e[0] for e in plant_me],
        n_seize_me=len(seize_me), n_seize_op=len(seize_op),
        seize_steps_me=[e[0] for e in seize_me], seize_steps_op=[e[0] for e in seize_op],
        dep_me=(len(dep_me), round(sum(e[3] for e in dep_me),1)),
        dep_op=(len(dep_op), round(sum(e[3] for e in dep_op),1)),
        dep_last30_me=(len(last30_dep_me), round(sum(e[3] for e in last30_dep_me),1)),
        dep_last30_op=(len(last30_dep_op), round(sum(e[3] for e in last30_dep_op),1)),
        last_dep_step_me=dep_me[-1][0] if dep_me else None,
        cross_dep_me=(len(cross_dep_me), round(sum(e[3] for e in cross_dep_me),1)),
        deaths_me=death_stats(death_me), deaths_op=death_stats(death_op),
        deaths_last30_me=sum(1 for e in death_me if e[0]>=270),
        tree_peak_me=tree_peak_me, tree_peak_op=tree_peak_op,
        kp_me=keypoints(tm), kp_op=keypoints(to),
        carried_me={s: round(tm[s][4],1) for s in (200,250,270,280,290,299) if s in tm},
        cash_gain_270_299=(round(tm[299][0]-tm[270][0],1) if 270 in tm else None),
        cash_gain_270_299_op=(round(to[299][0]-to[270][0],1) if 270 in to else None),
        cash_diff_by_step={s: round(tm[s][0]-to[s][0],1) for s in (10,30,50,100,150,200,250,299) if s in tm},
        map_carbon_final=samples[-1][3],
    )
    return out

results=[]
for f in sorted(glob.glob("summaries/a2_*.json")):
    results.append(analyze(f, "claude-AlphaCarbon-2"))
for f in sorted(glob.glob("summaries/a1_*.json")):
    results.append(analyze(f, "claude-AlphaCarbon"))
json.dump(results, open("alpha_analysis_data.json","w"), indent=1, ensure_ascii=False)
for r in results:
    print("="*100)
    print(f"{r['file']} vs {r['opp']} | {'WIN' if r['target_win'] else 'LOSS'} {r['score_me']:.0f}:{r['score_op']:.0f} | target replay side {r['target_replay_side']} timing {r['timing_ms_per_step']}")
    print(" recruit ME (C,P) total:", r['n_recruit_me'], " OP:", r['n_recruit_op'])
    print(" recruit seq ME first12:", r['recruit_seq_me'][:12])
    print(" first plant ME/OP:", r['first_plant_me'], r['first_plant_op'], "| plants ME/OP:", r['n_plant_me'], r['n_plant_op'], "| last plant ME:", r['last_plant_me'])
    print(" seize ME/OP:", r['n_seize_me'], r['n_seize_op'], "seize steps ME:", r['seize_steps_me'][:10])
    print(" deposits ME:", r['dep_me'], " OP:", r['dep_op'], "| last30 ME:", r['dep_last30_me'], " OP:", r['dep_last30_op'], "| last dep step ME:", r['last_dep_step_me'], "| cross-credit ME:", r['cross_dep_me'])
    print(" deaths ME:", r['deaths_me'], " last30:", r['deaths_last30_me'], "| deaths OP:", r['deaths_op'])
    print(" tree peak ME:", r['tree_peak_me'], " OP:", r['tree_peak_op'])
    print(" kp ME  step:[cash,C,P,T,carried]:", {k:[round(x) if isinstance(x,float) else x for x in v] for k,v in r['kp_me'].items()})
    print(" kp OP :", {k:[round(x) if isinstance(x,float) else x for x in v] for k,v in r['kp_op'].items()})
    print(" cash diff by step:", r['cash_diff_by_step'])
    print(" carried ME late:", r['carried_me'], "| cash gain 270->299 ME/OP:", r['cash_gain_270_299'], r['cash_gain_270_299_op'])
