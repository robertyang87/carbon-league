"""One antithetic policy-search update on terminal U, with map-paired controls."""
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import sys
import types

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE))
from tree_audited_game import exam, prior

ACTORS = ROOT / "data/weights/sapling_1200k"
PARENT = ACTORS / "hybrid_it50_v3.py"
PARENT_SHA = "b54cc51f4109f19f4a7b508eafc9cf04a28b7e295a9f442c5208f0d33f10a9ca"
OPP = ROOT / "code/packs/ppou2_m_it45.py"
OPP_SHA = "4675f0ca7e1d1db523f06b351c8cf0419051f255345635abe67135053a90fc97"
# Collector, planter and base, before and from step 180. Endpoints are opposite perturbations of 0.25.
ARMS = {"parent": [[.25, .25, .25], [.25, .25, .25]],
        "plus": [[.15, .36, .37], [.35, .17, .16]],
        "minus": [[.35, .14, .13], [.15, .33, .34]]}
TRAIN = [621453700, 621453701]
HOLD = [621453800, 621453801]


def packed(table):
    if prior.digest(PARENT) != PARENT_SHA:
        raise ValueError("hybrid changed")
    old = PARENT.read_text()
    key = '        combined = _HYBRID["expert_weight"] * scores[0] + _HYBRID["student_weight"] * scores[1]'
    new = ('        role_index = 2 if slot >= 10 else int(x["tok_kind"][slot])\n'
           '        alpha = _TERMINAL_BLEND[int(x["left"] <= 119)][role_index]\n'
           '        combined = (1. - alpha) * scores[0] + alpha * scores[1]')
    if old.count(key) != 1:
        raise ValueError("ambiguous decoder splice")
    return (old.replace(key,new)+"\n_TERMINAL_BLEND = "+repr(table)+"\n").encode()


def load_agent(path, name, content=None):
    if content is None:
        mod = prior.shared.module(path, name)
    else:
        mod = types.ModuleType(name)
        mod.__file__ = str(path)
        exec(compile(content,str(path),"exec"), mod.__dict__)
    # Both packs contain standalone modules with overlapping names. Restore each actor's
    # own embedded namespace before its call, including lazy first-call construction.
    deps = dict(mod._P) if hasattr(mod, "_P") else {key:sys.modules[key] for key in mod._SRC}
    return mod, deps


def act(actor, obs, config):
    mod, deps = actor
    sys.modules.update(deps)
    result = mod.agent(obs, config)
    if not isinstance(result, dict):
        raise ValueError("invalid action")
    return result


def utility(margin):
    wdl = 1 if margin > 0 else .5 if margin == 0 else 0
    return .5*wdl + .5/(1.+math.exp(-max(-80.,min(80.,margin/1000.))))


def play(job):
    stage, seed, seat, name = job
    os.sched_setaffinity(0, {2+seat})
    import numpy as np
    rng = seed + 997 * seat
    random.seed(rng)
    np.random.seed(rng)
    sim = exam.FastCarbon(exam.CFG).reset(seed=seed)
    if prior.digest(PARENT) != PARENT_SHA or prior.digest(OPP) != OPP_SHA:
        raise ValueError("weight identity mismatch")
    candidate = load_agent(PARENT,"blend_train_actor",None if name=="parent" else packed(ARMS[name]))
    opponent = load_agent(OPP,"fixed_it45")
    if opponent[0].CONF["SEARCH"] or not opponent[0].CONF["GUARD"]:
        raise ValueError("unexpected training opponent")
    # Exact public-actor parity of the parameterized control on the first 20 actual
    # roots of each control game separately in both seats. This includes step zero.
    neutral = load_agent(PARENT,"neutral_actor",packed(ARMS["parent"])) if name=="parent" else None
    players = [candidate,opponent] if seat==0 else [opponent,candidate]
    actions = []
    for t in range(exam.EPISODE_STEPS-1):
        before = list(sim.cash)
        joint = [act(players[s],sim.obs(s),sim.config) for s in (0,1)]
        if neutral and t < 20 and act(neutral,sim.obs(seat),sim.config) != joint[seat]:
            raise ValueError("neutral blend differs from exact parent")
        for s,commands in enumerate(joint):
            ids = set(sim.player_workers[s]) | {bid for bid,_ in sim.bases[s]}
            if set(commands)-ids or any(a not in ("UP","RIGHT","DOWN","LEFT","RECCOLLECTOR","RECPLANTER",None) for a in commands.values()):
                raise ValueError("invalid joint move")
        actions.append(joint)
        sim.step(*joint)
        state = sim.get_state()
        extinct = [bool(exam._extinct(state,p)) for p in (0,1)]
        if any(extinct): break
    if any(opponent[0].agent.last_stats.get(k,0) for k in ("errors","root_errors")):
        raise ValueError("training opponent swallowed error")
    score = [sim.step_count-exam.EPISODE_STEPS-1 if extinct[p] else (before if any(extinct) else sim.cash)[p] for p in (0,1)]
    margin = score[seat]-score[1-seat]
    result = "win" if margin>0 else "draw" if margin==0 else "loss"
    raw={"seed":seed,"candidate_seat":seat,"actions":actions,"official_score_by_seat":score,
         "steps":sim.step_count,"game_configuration":exam.CFG,"margin":margin,"result":result,
         "extinct_by_seat":extinct,"rng_seed":rng}
    path=HERE/f"{stage}_{seed}_{seat}_{name}.json.gz"
    tmp=path.with_suffix(".tmp")
    blob=gzip.compress(json.dumps(raw,sort_keys=True,separators=(",",":"),allow_nan=False).encode(),mtime=0)
    tmp.write_bytes(blob); os.replace(tmp,path)
    return {"stage":stage,"seed":seed,"candidate_seat":seat,"arm":name,"U":utility(margin),
            "margin":margin,"result":result,"official_score_by_seat":score,"actual_transitions":sim.step_count,
            "raw_file":path.name,"raw_sha256":hashlib.sha256(blob).hexdigest(),
            "neutral_parity_calls":min(20,len(actions)) if neutral else 0,
            "student_slots_selected_in_disagreement":sum(d["selected_student_in_disagreement"] for d in candidate[0]._COUNTS.values())}


def means(rows):
    return {k:sum(r[k] for r in rows)/len(rows) for k in ("U","margin")}


def main():
    import fcntl
    from official_game_replay import check
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise ValueError("one BLAS thread is required")
    lock=(HERE/".lock").open("a"); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    journal=HERE/"results.json"
    recipe={"parent":PARENT_SHA,"opponent":OPP_SHA,"train":TRAIN,"hold":HOLD,"arms":ARMS,
            "source_sha256":prior.digest(__file__), "phase_boundary":180,
            "selection":"Among two symmetric perturbations choose highest mean training U then mean margin. "
                        "Compare the winner to the parent on independent proxy holdout maps. "
                        "To expose a frozen candidate to original-examiner development, require positive U and margin "
                        "relative to parent on both training and holdout means, with actual Sapling decisions."}
    doc=json.loads(journal.read_bytes()) if journal.exists() else {"recipe":recipe,"games":[]}
    if doc["recipe"]!=recipe: raise ValueError("run identity changed")
    prior.shared.atomic_json(journal,doc)
    def stage(pool, phase, seeds, arms):
        for seed in seeds:
            for arm in arms:
                jobs=[]
                for seat in (0,1):
                    known=[r for r in doc["games"] if (r["stage"],r["seed"],r["candidate_seat"],r["arm"])==(phase,seed,seat,arm)]
                    if known:
                        if len(known)!=1 or prior.digest(HERE/known[0]["raw_file"])!=known[0]["raw_sha256"]: raise ValueError("invalid saved training game")
                    else: jobs.append((phase,seed,seat,arm))
                for row in pool.map(play,jobs):
                    row["original_interpreter"]=check(HERE,row)
                    doc["games"].append(row); prior.shared.atomic_json(journal,doc)
                    print(phase,seed,row["candidate_seat"],arm,row["result"],round(row["margin"],2),flush=True)
        return {arm:means([r for r in doc["games"] if r["stage"]==phase and r["arm"]==arm]) for arm in arms}
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context("spawn")) as pool:
        train=stage(pool,"train",TRAIN,list(ARMS))
        best=max(("plus","minus"),key=lambda a:(train[a]["U"],train[a]["margin"]))
        hold=stage(pool,"holdout",HOLD,["parent",best])
    positive=all(s[best][k]>s["parent"][k] for s in (train,hold) for k in ("U","margin"))
    supported=all(r["student_slots_selected_in_disagreement"]>0 for r in doc["games"] if r["arm"]==best)
    payload=packed(ARMS[best]); pack_sha=hashlib.sha256(payload).hexdigest()
    if positive and supported:
        output=ACTORS/"terminal_blend_agent.py"
        if output.exists() and output.read_bytes()!=payload: raise FileExistsError(output)
        if not output.exists():
            with output.open("xb") as sink: sink.write(payload)
    doc.update(status="complete",train_means=train,holdout_means=hold,chosen=best,
               admitted_to_original_development=positive and supported,chosen_pack_sha256=pack_sha)
    prior.shared.atomic_json(journal,doc)
    print(doc["chosen"],positive,supported,train,hold,flush=True)


if __name__=="__main__":
    main()
