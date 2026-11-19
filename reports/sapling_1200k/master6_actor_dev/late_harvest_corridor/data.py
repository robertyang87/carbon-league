"""Frozen paired native replays. Public policies and offline counterfactual auditors use separate files."""
from copy import deepcopy
from functools import lru_cache
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
ACTOR=HERE.parent
ROOT=ACTOR.parents[2]
STUDY=ACTOR/"endgame_queue_dev/study.py"
PACK="9c54df17eba7cfecf6ac2698eccbbadc2607665e16ab613925188131ce5514e0"
EXAM="65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"
WEIGHTS="8a5544ce41cf7fef04424eb735d43c7a659411f872bbd8d90f0046843bad33a2"
SOURCES=(
    ("dev","tree_response_training/planter_survival/native_dev/parent",(621481220,621481221)),
    ("dev","symmetry_dev/parent",(621495120,621495121)),
    ("eval","economy_aux_dev/native_dev/parent",(621497120,621497121)),
    ("eval","tree_response_training/supplement_01/batches/intake_08/source",(621478120,621478121)))
WINDOW=12


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    raw=p.read_bytes()
    return json.loads(gzip.decompress(raw) if p.suffix==".gz" else raw)


def must(ok,why):
    if not ok:raise ValueError(why)


@lru_cache(maxsize=1)
def reference():
    spec=importlib.util.spec_from_file_location("corridor_reference_economy",STUDY)
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    return ref


def registry():
    ref=reference();games=[];srcs={};version={}
    alias=ROOT/"data/master3s2_source.py"
    canonical=subprocess.check_output(["git","-C",str(ROOT),"show","HEAD:sources/master3s2/master3s2.py"])
    must(sha(alias)==hashlib.sha256(canonical).hexdigest()==EXAM,"official master source")
    for split,folder,seeds in SOURCES:
        d=ACTOR/folder;native=read(d/"results.json");official=read(d/"official_replay.json")
        head=native["header"];run=official["runs"][d.name]
        must(head["source_sha256"]["data/master3s2_source.py"]==EXAM and head["pack_sha256"]==run["pack_sha256"]==PACK and head["weight_sha256"]==WEIGHTS,"subject/examiner identity")
        must(native["status"]=="complete" and set(head["seeds"])==set(seeds),"wrong allocation")
        for f,h in official["engine_source_sha256"].items():
            must(sha(ref.ENGINE/f)==h,"official engine drift")
            version[f]=h
        rawrows={(v["seed"],v["candidate_seat"]):v for v in native["games"]}
        replay={(v["seed"],v["seat"]):v for v in run["games"]}
        keys={(s,m) for s in seeds for m in (0,1)}
        must(len(rawrows)==len(native["games"])==len(replay)==len(run["games"])==4 and keys==set(rawrows)==set(replay),"missing map or seat")
        srcs[folder]={f:sha(d/f) for f in ("results.json","official_replay.json")}
        for seed,seat in sorted(keys):
            row=rawrows[seed,seat];auth=replay[seed,seat];path=d/row["raw_file"]
            ref.validate_game(row)
            ref.verify_raw(d,row,PACK,WEIGHTS,expected_terrain=auth["map_sha256"])
            must(sha(path)==row["raw_sha256"]==auth["transcript_sha256"] and row["official_score_by_seat"]==auth["official_reward_by_seat"]
                 and row["actual_transitions"]==auth["steps"] and auth["statuses"]==["DONE","DONE"],"archive/certificate mismatch")
            games.append(dict(split=split,seed=seed,seat=seat,path=str(path.relative_to(ROOT)),raw_sha256=sha(path),
                              terrain_sha256=auth["map_sha256"],steps=auth["steps"]))
    own={k:{v["terrain_sha256"] for v in games if v["split"]==k} for k in ("dev","eval")}
    must(len(own["dev"])==len(own["eval"])==4 and own["dev"].isdisjoint(own["eval"]),"terrain collision")
    src={"schema":1,"pack_sha256":PACK,"weight_manifest_sha256":WEIGHTS,"examiner_sha256":EXAM,"inputs":srcs,
         "reference_replay_sha256":sha(STUDY),"engine_sha256":version,"games":games,"development_maps":4,"evaluation_maps":4}
    locked=HERE/"protocol.json"
    digest=sha(locked)
    must(digest=="1d6069ed2af8a3afb7df2c3d826ff50fe43031b02519ac32ef9a3ca0b4139c1f","registered protocol changed")
    exact={(g["split"],v["seed"],v["seat"],v["terrain"],v["raw_sha256"]) for g in read(locked)["groups"] for v in g["maps"]}
    must(exact=={(v["split"],v["seed"],v["seat"],v["terrain_sha256"],v["raw_sha256"]) for v in games},"registered maps differ")
    for group in read(locked)["groups"]:
        files=srcs[group["source_dir"]]
        must(files["results.json"]==group["results_sha256"] and files["official_replay.json"]==group["official_sha256"],"frozen exams differ")
    src["protocol_sha256"]=digest
    ref.write(HERE/"source_registry.json",src)
    return src


def chosen_prefix(feat, obs_features, own_command, F):
    y=feat.labels(own_command);base=F.Prefix(obs_features,feat.aux)
    policy={}
    for i in range(len(y)):
        if y[i]<0:continue
        _,_,legal=base.feats(i)
        act=int(y[i])
        policy[feat.own_uid[i]]={"slot":i,"chosen":act,"legal":legal.astype(bool).tolist(),
                                 "selected_legal":bool(legal[act])}
        base.advance(i,act)
    original={k:v for k,v in own_command.items() if v and v!="NONE"}
    must(feat.to_actions(y)==original,"unrepresentable focal command")
    return policy


def build():
    ref=reference(); import cnet2_feat as F
    src=registry()
    public=[];audit=[];counts={k:0 for k in ("dev","eval")};bad=0;total=0
    for game in src["games"]:
        raw=read(ROOT/game["path"]);s=game["seat"];seed=game["seed"];steps=game["steps"]
        cfg=raw["game_configuration"];ref.validate_examiner_configuration(raw["configuration"])
        must(not raw["audit_messages"] and not raw["errors"] and steps==len(raw["actions"]),"source execution")
        sim=ref.FastCarbon(cfg).reset(seed=seed)
        must(list(sim.carbon)==raw["initial_full_carbon"] and hashlib.sha256(bytes(map(int,sim.carbon))).hexdigest()==game["terrain_sha256"],"generated terrain changed")
        f=F.Feat(s);observations=[];own={};states={};pairs={};flows={};prefix={}
        from_t=max(0,steps-WINDOW)
        aggregate=[{},{}]
        for t,j in enumerate(raw["actions"]):
            obs=deepcopy(sim.obs(s));x=f.observe(obs)
            must(set(obs)=={"carbon","players","player","step","trees","remainingOverageTime"} and obs["step"]==t,"public interface")
            observations.append(obs)
            if t>=from_t:
                own[str(t)]=deepcopy(j[s]);states[str(t)]=sim.get_state();pairs[str(t)]=j
                prefix[str(t)]=chosen_prefix(f,x,j[s],F)
                bad+=sum(not v["selected_legal"] for v in prefix[str(t)].values())
                total+=len(prefix[str(t)])
            row=ref.twin_step(sim,j,events=True)
            for side in (0,1):
                for k,v in row["flow"][side].items():aggregate[side][k]=aggregate[side].get(k,0)+v
            must(row["status"]==(["DONE","DONE"] if t==steps-1 else ["ACTIVE","ACTIVE"]),"invalid official transition")
            if t>=from_t:flows[str(t)]=row
        must(row["score"]==raw["official_score_by_seat"] and list(sim.cash)==row["cash"],"terminal score mismatch")
        initial=[p[0] for p in observations[0]["players"]]
        must(all(abs(row["cash"][i]-initial[i]-sum(aggregate[i].values()))<1e-6 for i in (0,1)),"unreconciled whole game economy")
        identity={k:game[k] for k in ("split","seed","seat")}
        public.append(dict(**identity,cfg={k:v for k,v in cfg.items() if k!="randomSeed"},first=from_t,
                           observed=observations,own=own,prefix=prefix))
        audit.append(dict(**identity,config=cfg,first=from_t,states=states,orders=pairs,accounts=flows,
                          original={k:row[k] for k in ("cash","score","status")},whole_game_cash_flows=aggregate))
        counts[game["split"]]+=len(own)
        print(game["split"],seed,s,len(own),flush=True)
    ref.write(HERE/"data.json.gz",{"schema":1,"games":public})
    ref.write(HERE/"audit.json.gz",{"schema":1,"games":audit})
    ref.write(HERE/"data_receipt.json",{"source_registry_sha256":sha(HERE/"source_registry.json"),
        "public_sha256":sha(HERE/"data.json.gz"),"audit_sha256":sha(HERE/"audit.json.gz"),"source_sha256":sha(Path(__file__)),
        "full_transitions":sum(g["steps"] for g in src["games"]),"retained_roots":counts,
        "recorded_slots":total,"recorded_slots_outside_prefix_mask":bad,
        "preaction_visibility":"opening onward only focal official observation; sparse roots retain own baseline and sequential public legality",
        "evaluation_only":"exact board, same-step enemy order, future actions, receipts and scores are physically separate in audit.json.gz"})


@lru_cache(maxsize=1)
def _load_public():
    receipt=read(HERE/"data_receipt.json")
    must(sha(HERE/"data.json.gz")==receipt["public_sha256"],"public hash mismatch")
    return read(HERE/"data.json.gz")


@lru_cache(maxsize=1)
def _load_audit():
    receipt=read(HERE/"data_receipt.json")
    must(sha(HERE/"audit.json.gz")==receipt["audit_sha256"],"evaluation evidence changed")
    return read(HERE/"audit.json.gz")


def load_public():
    return deepcopy(_load_public())


def load_audit():
    return deepcopy(_load_audit())


def public_roots(split):
    must(split in ("dev","eval"),"split must be dev or eval")
    for g in _load_public()["games"]:
        if g["split"]!=split:continue
        for t in sorted(map(int,g["own"])):
            yield ((g["seed"],g["seat"],t),deepcopy(g["observed"][t]),
                   {i:deepcopy(g["observed"][i]) for i in range(max(0,t-12),t)},
                   deepcopy(g["own"][str(t)]),dict(g["cfg"]))


def audit_root(key):
    seed,s,t=key
    for g in _load_audit()["games"]:
        if (g["seed"],g["seat"])!=(seed,s):continue
        times=sorted(k for k in map(int,g["orders"]) if k>=t)
        if not times or times[0]!=t:raise KeyError(key)
        return deepcopy({"state":g["states"][str(t)],"joint":g["orders"][str(t)],
                "remaining_joint":[g["orders"][str(i)] for i in times],"original":g["original"],
                "step_accounts":[g["accounts"][str(i)] for i in times],"config":g["config"]})
    raise KeyError(key)


if __name__=="__main__":
    build()
