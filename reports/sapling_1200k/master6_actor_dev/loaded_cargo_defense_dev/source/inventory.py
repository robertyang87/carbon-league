"""Lock cargo defence development maps by byte-verified source identity, before measuring losses."""
from collections import Counter
import gzip,hashlib,json,subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
ACTOR=HERE.parents[1];BASE=ACTOR.parent;ROOT=BASE.parents[1]
INVENTORY=ACTOR/"stratified_action_value_dev/source/inventory.json"
EXPECTED="7bb88ab8aff37d04726358328c42eaed35123c324d450ec08084bb531c405657"
DOMAIN=b"loaded-cargo-defence/top-5-training-4-held-2-reserve/v1\0"


def read(p):
    b=p.read_bytes();return json.loads(gzip.decompress(b) if p.suffix==".gz" else b)


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def checked(x,m):
    if not x:raise ValueError(m)


def main():
    current=HERE/"inventory.json"
    if current.exists():
        frozen=read(current)
        for row in frozen["games"]:
            checked(sha(ROOT/row["raw"])==row["raw_sha256"] and sha(ROOT/row["official"])==row["official_sha256"],"frozen inputs changed")
        return frozen
    checked(sha(INVENTORY)==EXPECTED,"wrong preceding inventory")
    inv=read(INVENTORY);old1=read(ACTOR/"late_local_residual_dev/protocol.json");old2=read(ACTOR/"stratified_action_value_dev/protocol.json")
    idmap={r['seed']:r['map'] for r in inv['v1_inventory']}
    excluded1={r['map'] for r in old1['source_groups']};excluded2={idmap[s] for s in old2["train_maps"]+old2["held_maps"]}
    checked(len(excluded1)==8 and len(excluded2)==6 and excluded1.isdisjoint(excluded2),"previous map definitions")
    candidates=set(idmap.values())-excluded1-excluded2
    checked(len(candidates)==11,"unexpected count of available original v1 maps")
    ranking=sorted((hashlib.sha256(DOMAIN+bytes.fromhex(b)).hexdigest(),b) for b in candidates)
    groups={b:("train" if i<5 else "held" if i<9 else "reserve") for i,(_,b) in enumerate(ranking)}
    rank={b:i for i,(_,b) in enumerate(ranking,1)}
    queue=read(ACTOR/"endgame_queue_dev/native_protocol.json")["cases"]
    ex={ (e['seed'],e['seat'],e['t']):e for e in queue}
    oldmix=read(ACTOR/"economy_aux_dev/native_dev/paired_edits/native_t288/protocol.json")
    gates=read(ACTOR/"late_harvest_corridor/protocol.json")
    proxy_maps={r["terrain"] for d in gates["groups"] for r in d["maps"]}
    directions=read(ACTOR/"return_value_dev/protocol.json")
    rows=[]
    for ref in inv["v1_inventory"]:
        seed,b,s=ref["seed"],ref["map"],ref["seat"]
        if b not in candidates:continue
        path=ROOT/ref["raw"];cert=ROOT/ref["official"]; d=read(path)
        checked((sha(path),sha(cert))==(ref["raw_sha256"],ref["official_sha256"]),"artifact changed")
        checked((d['seed'],d['candidate_seat'],d['pack_sha256'],d['examiner_md5'])==(seed,s,inv["accepted_raw_pack_sha256"],inv["original_md5"]),
                "pack/seat/opponent identity")
        checked(not d["errors"] and len(d["actions"])==d["steps"] and [a[0] for a in d["search_calls"]]==list(range(d["steps"]))
                and hashlib.sha256(bytes(map(int,d["initial_full_carbon"]))).hexdigest()==b,"incomplete source")
        verified=[x for a in read(cert)["runs"].values() if a["pack_sha256"]==d["pack_sha256"] for x in a["games"] if (x["seed"],x["seat"])==(seed,s)]
        checked(len(verified)==1 and (verified[0]["map_sha256"],verified[0]["transcript_sha256"],verified[0]["statuses"])==(b,sha(path),["DONE","DONE"]),"official record")
        uses=list(ref["other_label_root_sources"])
        for (u,v,t),x in ex.items():
            if (u,v)==(seed,s):
                uses.append({"t":t,"family":"old one-step terminal queue (whole/partial joint)","source":str((ACTOR/"endgame_queue_dev/native_protocol.json").relative_to(ROOT))})
        if (oldmix["seed"],oldmix["seat"])==(seed,s):
            uses.append({"t":oldmix["turn"],"family":"two-collector auxiliary advance-home bundle","source":str((ACTOR/"economy_aux_dev/native_dev/paired_edits/native_t288/protocol.json").relative_to(ROOT))})
        exclude=set(ref["excluded_exact_physical_steps"])|{u["t"] for u in uses}
        history=[ref["known_raw_source_purpose"]]
        if b in proxy_maps:
            history.append("late_harvest_corridor considered every physical t287..294 using proxy opponent policies; conservatively exclude entire eight-step range")
            exclude.update(range(287,295))
        if ref["used_for_terminal_queue_t297_298"]:history.append("endgame_queue considered t297 and t298")
        if seed in (621475120,621475121):history.append("lot_settlement_dev held-map prediction of factual principal; previous stratified archive also listed this map as unused optional hold")
        if seed==621461220:history.append("early development of full-game terminal_return_fit v1")
        if seed in (621497120,621497121):history.append("older economy_aux full-package A/B/C development games")
        if seed in (621495120,621495121):history.append("symmetry_dev full-package rotation development")
        if seed in (621481220,621481221):history.append("planter_survival whole-match development")
        rows.append({"split":groups[b],"rank":rank[b],"seed":seed,"seat":s,"map":b,"raw":ref["raw"],"raw_sha256":ref["raw_sha256"],
                     "official":ref["official"],"official_sha256":ref["official_sha256"],"config":ref["config"],"physical_steps":ref["steps"],
                     "original_pack_sha256":d["pack_sha256"],"examiner_md5":d["examiner_md5"],"prior_uses":history,
                     "older_intervention_roots":uses,"excluded_exact_physical_steps":sorted(exclude)})
    rows.sort(key=lambda r:(r["rank"],r["seat"]))
    checked(Counter(r["split"] for r in rows)=={"train":10,"held":8,"reserve":4} and all({r["seat"] for r in rows if r["map"]==b}=={0,1} for b in candidates),"unpaired source")
    out={"schema":"v1-certified-loaded-cargo-defence-identity-v1","pack_sha256":inv["accepted_raw_pack_sha256"],
         "original_source_sha256":"65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5",
         "cutoff_revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
         "input_inventory_sha256":EXPECTED,"historical_split_proofs":{str(p.relative_to(ROOT)):sha(p) for p in (ACTOR/"late_local_residual_dev/protocol.json",ACTOR/"stratified_action_value_dev/protocol.json")},
         "selection":"sort SHA-256(UTF8('loaded-cargo-defence/top-5-training-4-held-2-reserve/v1') + NUL + raw 32-byte terrain digest); first five train, next four held, last two reserve. No score or cargo outcome used.",
         "identity_universe":{"distinct_current_parent_terrains":25,"first_local_terrains":8,"second_stratified_terrains":6,
                             "overlap_between_prior_sets":0,"remaining":11,"earlier_whole_game_evaluation_maps":True},
         "excluded_previous_map_hashes":sorted(excluded1|excluded2),
         "maps":[{"seed":next(r["seed"] for r in rows if r["map"]==b),"map":b,"rank_hash":h,"split":groups[b]} for h,b in ranking],
         "configurations":inv["configurations"],"games":rows}
    current.write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n")
    return out


if __name__=="__main__":
    result=main();print({k:[x["seed"] for x in result["maps"] if x["split"]==k] for k in ("train","held","reserve")})
