"""Train a public-state conditional mixture by paired, full-game terminal rewards."""
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import fcntl
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path.insert(0,str(BASE/"terminal_blend"))
import train as arena
from official_game_replay import check
from tree_win_gate import RESERVED_RANGE

TRAIN=[621454710,621454711,621454712,621454713]
HOLD=[621454810,621454811,621454812]
REAL=621454920
ARMS={"parent":[.25,.25], "risk_plant":[.43,.11], "risk_freight":[.07,.39]}


def packed(params):
    if arena.prior.digest(arena.PARENT)!=arena.PARENT_SHA:
        raise ValueError("incorrect hybrid parent")
    source=arena.PARENT.read_text()
    old='        combined = _HYBRID["expert_weight"] * scores[0] + _HYBRID["student_weight"] * scores[1]'
    start='    x = _FEAT.observe(observation)'
    if source.count(old)!=1 or source.count(start)!=1:
        raise ValueError("ambiguous actor implant")
    source=source.replace(old,'        alpha = _student_fraction(F, x, aux, slot)\n        combined = (1. - alpha) * scores[0] + alpha * scores[1]')
    source=source.replace(start,'    _update_economy(observation)\n'+start)
    return (source+'\n_ECON_FRACTIONS = '+repr(params)+'\n'+(HERE/'runtime.py').read_text()).encode()


@contextmanager
def protocol():
    # Reuse the original fully audited game and scorer. Its three experiment
    # parameters are local to each worker process and restored on exit.
    saved=(arena.HERE,arena.packed,arena.ARMS)
    arena.HERE,arena.packed,arena.ARMS=HERE,packed,ARMS
    try: yield
    finally: arena.HERE,arena.packed,arena.ARMS=saved


def game(job):
    with protocol(): return arena.play(job)


def aggregate(rows):
    maps=[]
    for seed in sorted({g["seed"] for g in rows}):
        pair=sorted((g for g in rows if g["seed"]==seed),key=lambda x:x["candidate_seat"])
        if [g["candidate_seat"] for g in pair]!=[0,1]: raise ValueError("incomplete map pair")
        mean=arena.means(pair)
        maps.append({"seed":seed,**mean,"seat_margins":[g["margin"] for g in pair],
                     "seat_U":[g["U"] for g in pair],"wins":sum(g["result"]=="win" for g in pair)})
    return {"by_map":maps,**arena.means(maps)}


def changes(candidate,parent):
    rows=[]
    for c,p in zip(candidate["by_map"],parent["by_map"],strict=True):
        if c["seed"]!=p["seed"]: raise ValueError("mismatched maps")
        rows.append({"seed":c["seed"],"U":c["U"]-p["U"],"margin":c["margin"]-p["margin"]})
    return {"by_map":rows,"maps_positive_U":sum(r["U"]>0 for r in rows),
            "maps_negative_U":sum(r["U"]<0 for r in rows),"worst_map_U":min(r["U"] for r in rows),
            "worst_map_margin":min(r["margin"] for r in rows),**arena.means(rows)}


def phase(pool,doc,phase_name,seeds,arms):
    for seed in seeds:
        for a in arms:
            jobs=[]
            for s in (0,1):
                done=[g for g in doc["games"] if (g["stage"],g["seed"],g["candidate_seat"],g["arm"])==(phase_name,seed,s,a)]
                if not done: jobs.append((phase_name,seed,s,a))
                elif len(done)!=1 or arena.prior.digest(HERE/done[0]["raw_file"])!=done[0]["raw_sha256"]: raise ValueError("incorrect checkpoint")
            for row in pool.map(game,jobs):
                row["original_interpreter"]=check(HERE,row)
                doc["games"].append(row)
                arena.prior.shared.atomic_json(HERE/"results.json",doc)
                print(phase_name,seed,row["candidate_seat"],a,row["result"],round(row["margin"],2),flush=True)
    return {a:aggregate([g for g in doc["games"] if g["stage"]==phase_name and g["arm"]==a]) for a in arms}


def run():
    lock=(HERE/".lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    corpus=BASE/'courier_train_labels.json.gz'
    corpus_games=json.loads(gzip.decompress(corpus.read_bytes()))['games']
    exclude=sorted({int(g['seed']) for g in corpus_games})
    train26=sorted({int(g['seed']) for g in corpus_games if g['split']=='train'})
    maps=TRAIN+HOLD+[REAL]
    if (len(train26)!=26 or len(exclude)!=36 or len(set(maps))!=len(maps) or set(exclude)&set(maps)
            or set(maps)&set(arena.TRAIN+arena.HOLD)
            or any(RESERVED_RANGE[0]<=s<RESERVED_RANGE[1] for s in maps)):
        raise ValueError("overlapping train, diagnostic or formal maps")
    recipe={"parent_sha256":arena.PARENT_SHA,"opponent_sha256":arena.OPP_SHA,"arms":ARMS,
            "train":TRAIN,"hold":HOLD,"original_development":REAL,"excluded_26":train26,"excluded_train_and_val":exclude,
            "excluded_labels_sha256":arena.prior.digest(corpus),"formal_reserved":RESERVED_RANGE,
            "source_sha256":{p.name:arena.prior.digest(p) for p in (Path(__file__),HERE/"runtime.py",Path(arena.__file__))},
            "criterion":"Select the candidate maximizing the number of four paired training maps with positive U gain, "
                        "then mean U, then mean margin. Freeze selected arm before separate three-map proxy holdout. "
                        "Run one diagnostic original-examiner pair for that same candidate if it improves U on >=2 training maps "
                        "and its Sapling branch actually selects actions. The heldout proxy does not decide which candidate "
                        "receives original feedback. Report mean, worst map, number of flips, both seats, and reachability."}
    # JSON preserves all recorded experiment parameters including the reserved bounds.
    recipe=json.loads(json.dumps(recipe))
    path=HERE/"results.json"
    doc=json.loads(path.read_bytes()) if path.exists() else {"recipe":recipe,"games":[]}
    if doc["recipe"]!=recipe: raise ValueError("frozen recipe differs")
    arena.prior.shared.atomic_json(path,doc)
    with ProcessPoolExecutor(2,mp_context=multiprocessing.get_context("spawn")) as pool:
        train=phase(pool,doc,"train",TRAIN,list(ARMS))
        deltas={a:changes(train[a],train["parent"]) for a in ("risk_plant","risk_freight")}
        best=max(deltas,key=lambda a:(deltas[a]["maps_positive_U"],deltas[a]["U"],deltas[a]["margin"]))
        admit=(deltas[best]["maps_positive_U"]>=2 and all(g["student_slots_selected_in_disagreement"]>0
              for g in doc["games"] if g["stage"]=="train" and g["arm"]==best))
        payload=packed(ARMS[best]); sha=hashlib.sha256(payload).hexdigest()
        selection={"chosen":best,"chosen_pack_sha256":sha,"training":train,"training_deltas":deltas,
                   "admit_original_diagnostic":admit}
        freeze=HERE/"selection.json"
        if freeze.exists() and json.loads(freeze.read_bytes())!=selection: raise ValueError("train selection changed")
        arena.prior.shared.atomic_json(freeze,selection)
        if admit:
            output=arena.ACTORS/"terminal_economy_agent.py"
            if output.exists() and output.read_bytes()!=payload: raise ValueError("different frozen package")
            if not output.exists():
                with output.open("xb") as stream: stream.write(payload)
        hold=phase(pool,doc,"holdout",HOLD,["parent",best])
    doc.update(status="complete",selection=selection,holdout=hold,holdout_delta=changes(hold[best],hold["parent"]))
    arena.prior.shared.atomic_json(path,doc)
    print(best,admit,doc["holdout_delta"],flush=True)


if __name__=="__main__": run()
