"""After roots are frozen, independently check one-step interventions for the recorded enemy root action."""
from study import HERE,ROOT,read,sha,save,load_case,begin,clean,PACK_HASH,live


def event(obs,tree):
    owner=next((s for s,p in enumerate(obs.players) if tree in p[3]),None)
    return {"owner":owner,"income":obs.trees[tree][1] if owner is not None else 0.}


def main():
    rows=[]
    for r in read(HERE/"roots.json")["selected"]:
        folder=HERE/r["family"];reg=read(folder/"live_protocol.json")
        assert sha(folder/"live_protocol.json")==r["case_sha256"]
        seed,seat,t=r["seed"],r["seat"],r["t"]
        raw,g=load_case(ROOT/reg["source_dir"],seed,seat)
        if g!=reg["record"] or raw["pack_sha256"]!=PACK_HASH:
            raise ValueError("parent game identity")
        result={}
        for which in ("parent","alternative"):
            env=begin(raw)
            for a in raw["actions"][:t]:env.step(a)
            pre=env.states[0].observation
            joint=[dict(x) for x in raw["actions"][t]];commands=joint[seat]
            if which=="alternative":
                if r["alternative"]=="NONE":commands.pop(r["worker"],None)
                else:commands[r["worker"]]=r["alternative"]
            env.step(joint);post=env.states[0].observation
            result[which]={"own_root_commands":clean(commands),"opponent_root_commands":clean(joint[1-seat]),
                           "tree_before":event(pre,r["tree"]),"tree_after":event(post,r["tree"]),
                           "worker_after":post.players[seat][2].get(r["worker"]),
                           "cash_at_t_plus_one":[p[0] for p in post.players]}
        a,b=result.values();assert a["opponent_root_commands"]==b["opponent_root_commands"]
        assert {uid for uid in a["own_root_commands"].keys()|b["own_root_commands"].keys()
                if a["own_root_commands"].get(uid)!=b["own_root_commands"].get(uid)}=={r["worker"]}
        rows.append({"case_sha256":r["case_sha256"],"seed":seed,"t":t,"seat":seat,"family":r["family"],"arms":result})
    save(HERE/"root_checks.json",rows)
    print([(v["family"],[(k,x["tree_after"]) for k,x in v["arms"].items()]) for v in rows])


if __name__=="__main__":main()
