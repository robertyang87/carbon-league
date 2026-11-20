"""Freeze old map identities and the single training/holdout trajectory per map before collection."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
D=HERE.parent
sha=lambda b: hashlib.sha256(b).hexdigest()
raw=(D/"opponent_policy/protocol.json").read_bytes()
geometry=(D/"symmetry_value/search_protocol.json").read_bytes()
assert sha(raw)=="16e77a290cfe1c22503981f34cff934aa3fccd6fc69b0ba41d9383e1e078e502"
assert sha(geometry)=="1a2d96babdbacccf41724a830c5fd3563801559d73793cc0a5fb4cdf80834cc9"
panel=json.loads(raw)["games"]
train_maps=sorted({r["terrain"] for r in panel if r["split"]=="train"})
held_maps=sorted({r["terrain"] for r in panel if r["split"]=="val"})
exclude={r["terrain"] for r in json.loads(geometry)["roots"]}
assert len(train_maps)==24 and len(held_maps)==4 and len(exclude)==4 and exclude<set(train_maps)
chosen=[("train",m) for m in train_maps if m not in exclude][:8]+[("held",m) for m in held_maps]
tickets=[]
for i,(split,m) in enumerate(chosen):
    seat=(i//2)%2
    step=(279,285)[i%2]
    row=next(r for r in panel if r["terrain"]==m and r["expert_seat"]==seat)
    rng=int.from_bytes(hashlib.sha256(f"native-rank-20261120/{m}/{seat}/{step}".encode()).digest()[:4],"big")
    tickets.append(dict(ordinal=i,split=split,seed=row["seed"],terrain=m,seat=seat,step=step,path=row["path"],
                        source_sha256=row["sha256"],official_cash_path_sha256=row["official_cash_path_sha256"],
                        actions_sha256=row["actions_sha256"],policy_rng=rng,
                        arm_order=["first","second"] if i%2==0 else ["second","first"]))
p=dict(schema="native-original-response-action-pairs/1",base="7cf5c6c91fc59b46b2a38924c4c5ab56143fb1242d082f9426303af7336ac3ed",
       examiner="65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5",source_ledger=sha(raw),
       original_train_terrains=train_maps,original_holdout_terrains=held_maps,excluded_symmetry_terrains=sorted(exclude),tickets=tickets,
       selection="All 12 tickets are fixed once. At each only the full unmodified V2 public root search is allowed, .58 native deadline, original R=M=MMAX=1. Use distinct surviving joint orders with complete H=8 on the same roll set. Sort by native score descending, tie by native alive order. First and second distinct qualified orders are the arms. Keep exact ties; fewer than two or incomplete depth => no pair, no substitute. A root without near-terminal relevance remains eligible.",
       features="Capture the finalists' actual _search look inputs of the original frozen V2 under the same root belief, after quantization, before looking at suffix outcomes. NPZ compact uint8 X reconstructed exactly as float32*0.01, per-row float32 native _scalars, source critic value, final depth/step/alive arm indices, exact-terminal mask and roll ids. Root JSON per arm contains issued orders, full simulated players as supplied to network at each selected leaf, native leaf X hash. Full hidden referee carbon, future orders and outcomes excluded from inputs.",
       labels="Both one-action interventions share the same referee and the same fresh native original opponent root reply with no knowledge of our move. Preserve the private state/real root cost of V2; correct its committed root intent per intervention. Both then continue with identical V2 and original native public policy to official score. Store per-arm score and return via native _terminal(margin); target is first minus second. Record all errors and branch timings. Archived physical root and public history checked independently with official referee and FastCarbon. Private historical entropy and per-tick cost unrecorded; use common reconstructed private streams and local cost calibration; local public-call timing is diagnostic.",
       chronology="Collect TRAIN ordinals 0–7 first, in order. Retain every eligible/excluded root. Hand their complete receipts to model trainer, wait for blind head digest before opening held 8–11. Held tapes, private RNG, score and labels are separately located; avoid pulling any held outcome into fit or hyperparameter choices. No independent final boards. No map substitution if any trial is unavailable.",
       budget="6 roots at step 279 (20 moves) and 6 at 285 (14 moves). If all qualify and last to time limit: 408 joint suffix steps, 816 per-seat records, 792 actual public calls including each shared root exactly once per physical seat. Initialization, warm prefix commands and archive reconstruction timed separately. Process serial per pinned CPU; regulator and clock recorded.")
raw=(json.dumps(p,sort_keys=True,indent=2)+"\n").encode()
dest=HERE/"protocol.json"
if dest.exists() and dest.read_bytes()!=raw:raise ValueError("Different registration already present")
dest.write_bytes(raw)
print(sha(raw),[(r["seed"],r["seat"],r["step"]) for r in tickets])
