import gzip,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ACTOR=HERE.parent
TREE=ACTOR/"tree_response_training"
ROOT=ACTOR.parents[2]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_bytes())
def main():
    batches=[TREE,*[TREE/f"batches/intake_{i:02}" for i in (2,3,4)]]
    registry=[]
    for i,b in enumerate(batches):
        path=b/"source_receipt.json";r=load(path)
        if digest(b/"source/official_replay.json")!=r["official_sha256"]:raise ValueError("official receipt")
        for g in r["games"]:
            if digest(ROOT/g["raw_path"])!=g["raw_sha256"]:raise ValueError("recorded source")
            registry.append(dict(split="train" if i<2 else "held",raw=g["raw_path"],raw_sha256=g["raw_sha256"],
                      map=g["map_sha256"],seed=g["seed"],seat=g["seat"],official_sha256=r["official_sha256"]))
    if len(registry)!=16 or len({r["map"] for r in registry})!=8:raise ValueError("source membership")
    old=set(r["map"] for r in registry if r["split"]=="train")
    if any(r["map"] in old for r in registry if r["split"]=="held"):raise ValueError("map leakage")
    proto={"source_groups":registry,"proposal_sha256":digest(HERE/"runtime.py"),
       "t_range":[286,296],"selection":"First public returning vs staying alternative per registered map/seat from t286..296, replay old own public prefix, keep all other own commands, do not rank by future outcomes. Original parent and changed joints must be legal including recruitment and friendly collision protection. No opportunity or rejecting all interventions gives zero paired difference for that registered seat.",
       "optimization":"For the preregistered four training maps, fit a clipped-outcome, map-equal ridge residual with a conservative zero prior to independent observed difference in official terminal margin, original focal principal direct delivery and all own direct delivery; no global network updates. Fixed 8 public features in runtime.py, L2=0.5 including intercept, labels clipped to +/-400 cash. Entry requires predicted margin>10 and both predicted focal principal and predicted own delivery >=-5. At least four paired labels on three different training maps, else withhold.",
       "held_gate":"Freeze weights before inspecting any intervention outcome on four remaining maps. Count all 8 registered seats, unchanged decisions as zero. Admit to a two-fresh-map swapped-seat full-game pilot only if exact mechanism guards pass, original intervention independently reproduces, intervention triggers on at least two different held maps, by-map mean official margin gain >0, both physical seats' aggregate gain >=0, combined original principal does not decrease, all own direct deposit aggregate increases. This is a screen, never formal proof.",
       "native":"Same source board, real full original Master3s2 with official temperature; hold root enemy commands fixed as simultaneously registered. Opponent continues independently on each resulting board. Self continues with single identical frozen v1 after intervention. Separate terminal official rewards (incl extinction), focal starting principal, fresh direct and whole-team and opponent deliveries. Any incompatible or missing certificate is ineligible; other strata may be recorded but not pooled.",
       "cost":"Train at most eight paired roots; held at most eight, only if the frozen residual actually changes a legal command. Existing exact same pair may be reused. Native jobs require the established shared native lock, no extra formal/evaluation maps selected during this phase."}
    (HERE/"protocol.json").write_text(json.dumps(proto,ensure_ascii=False,indent=2)+"\n")
if __name__=="__main__":main()
