from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent
TR=HERE.parent/"tree_response_training"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def split(paths):
    return [{"receipt":str(p.relative_to(HERE.parents[3])),"sha256":sha(p),
             "games":[{k:g[k] for k in ("seed","seat","raw_sha256","raw_path","map_sha256","steps")} for g in json.loads(p.read_bytes())["games"]]} for p in paths]
if __name__=="__main__":
    p={"parent_pack_sha256":"9c54df17eba7cfecf6ac2698eccbbadc2607665e16ab613925188131ce5514e0","train":split([TR/"source_receipt.json", TR/"batches/intake_03/source_receipt.json"]),"validation":split([TR/"batches/intake_04/source_receipt.json"]),
       "question":"For already-returning workers holding >=100, does switching only the first step of an equally short route to the same base protect carried principal against enemy tree siphoning without foregoing an extra move of harvest?",
       "release":"v1 + route.ranked, full prefix check, maximum 1 modified worker / turn; no training labels for future-vs-unchanged actions; published probabilities fit P(current enemy tree still same owner at h=1,2,3) using training maps only; known expiries exact; conditional approacher distance bucket 0/1/2/3/4+, beta-smoothed by 20 pooled samples / horizon (maps weighted equally). Opponent never sees first action. Alternative and old cannot plausibly hit a lighter collector in one move, both must advance 1 toward same nearest own base; actual v1 harvest, actual banking, all other orders and total number of shortest steps preserved. Light enemy next-turn safety guard is no guarantee of future safety. Compare old and new each with its own best continuation over the fixed 3-turn rollout and same treasury value. First step cannot increase modeled tax; predicted three-step cargo benefit >=max(8,2% of current cargo).",
       "validation":"all eligible v1 public transitions both seats. Overall and by map: genuine first-step legal changed commands, worker acquisition/spending/cargo traffic, collected new carbon, cash deliveries and exact enemy tree uptake, focal survival. The old path is independently replayed to terminal. Modified future is not fabricated. Candidate mechanism must improve actual focal carried principal after one step on each map, no additional focal death on either map and no forgone first-step new harvest. Statistical unit map. If met select first authentic trigger in chronological order, earliest seat t then seat 0 per map, for genuine two-sided responsive continuation, even if first-step gain is zero. No official new map pending root schedule. Never infer increased actual banking from carrying.",
       "fit":"zero/one label for each existing tree whose age allows survival to h, actual h owner by ID. Each map contributes same average weight per horizon. Conditional shrinkage 20 pseudo observations/h (rescale map weights to equivalent unweighted observation count).",
       "model_sha256":sha(HERE/"route.py")}
    path=HERE/"protocol.json"
    b=(json.dumps(p,indent=2,ensure_ascii=False)+"\n").encode()
    if path.exists() and path.read_bytes()!=b:raise ValueError("frozen protocol exists")
    path.write_bytes(b)
