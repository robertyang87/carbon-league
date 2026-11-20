"""Cross-check this cold public-entry evaluation against the separately archived BC release."""
from collections import Counter,defaultdict
import gzip
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
def read(path):
    raw=path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix==".gz" else raw)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    old_path=HERE.parent/"free_commands.json.gz"
    old={(r["tape"],r["step"],r["seat"]):r for r in read(old_path) if r["split"]=="val"}
    raw=read(HERE/"held_commands.json.gz")
    assert len(old)==len(raw)==len({(r["tape"],r["t"],r["seat"]) for r in raw})==192
    counts=defaultdict(Counter); actions=Counter(); by_seat=defaultdict(Counter)
    for r in raw:
        archive=old[r["tape"],r["t"],r["seat"]]
        assert all(archive[k]==r[k] for k in ("roles","expert","terrain"))
        assert archive["commands"]["bc"]==r["parent"]
        A=[r["parent"].get(uid,"NONE")==r["expert"].get(uid,"NONE") for uid in r["roles"]]
        B=[r["student"].get(uid,"NONE")==r["expert"].get(uid,"NONE") for uid in r["roles"]]
        assert all(A)==(r["parent_prefix"]["slot"] is None) and all(B)==(r["student_prefix"]["slot"] is None)
        assert r["slots"]==len(A)==len(B)
        tag=f"{int(all(A))}_to_{int(all(B))}"
        counts[r["seed"]][tag]+=1
        by_seat[r["seat"]][tag]+=1
        for before,after in zip(A,B):actions[f"{int(before)}_to_{int(after)}"]+=1
    fit=read(HERE/"fit.json");result=read(HERE/"result.json")
    assert sha(HERE/"run.py")==fit["script"] and sha(HERE/"protocol.json")==fit["protocol"]
    assert sha(HERE/"trained.py")==fit["package"]==result["package"]
    assert sha(HERE/"weights.npz")==fit["weights"] and sha(HERE/"held_commands.json.gz")==result["held_evidence"]
    train=read(HERE/"train_roots.json.gz")
    geo=set(r["key"][0] for r in train); assert len(geo)==12 and not geo.intersection(counts)
    all_slots=result["pooled"]["roles"]
    assert sum(v["gained"] for v in all_slots.values())==actions["0_to_1"]
    assert sum(v["lost"] for v in all_slots.values())==actions["1_to_0"]
    assert sum(v["student_correct"] for v in all_slots.values())==actions["0_to_1"]+actions["1_to_1"]
    result=dict(original_untouched_parent_log_sha256=sha(old_path),identical_baseline_and_targets=len(raw),
                previously_correct_versus_new_team_actions_by_map=counts,by_seat=by_seat,
                entities_before_and_after=actions,training_terrains=len(geo),held_terrains=len(counts),
                training_rows=len(train),fit_sha256=sha(HERE/"fit.json"),result_sha256=sha(HERE/"result.json"))
    (HERE/"audit.json").write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+"\n")
if __name__=="__main__":
    run()
