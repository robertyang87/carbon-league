"""Match only complete live candidate orders to the two signed arms of a fixed historical fork."""
import argparse
from collections import Counter, defaultdict
from importlib.util import module_from_spec, spec_from_file_location
import os
from pathlib import Path
import sys

for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"

HERE = Path(__file__).resolve().parent
CORPUS = HERE.parent
EXP = CORPUS.parent
OLD = CORPUS / "terminal_direction"
LATE = CORPUS / "return_action_labels"
spec = spec_from_file_location("fixed_terminal_pairs", OLD/"compare.py")
C = module_from_spec(spec)
spec.loader.exec_module(C)
ROOT = C.ROOT
PARENT = "8da4b465f7fa6e601e51bd7c2e81f25825cb59b9b5ab285f7f56d0f6bbfe1cac"
EXAMINER = "65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"
INPUTS = {
    OLD/"commands.json.gz":"5ffd11e7b9aea4094b0d0fb575281483abce315591bd76fb05844f4d383c673a",
    LATE/"roots.json":"7ad088e15fd22875dbcd191a94d11ec8e25e3a0a5ecfdf00113afc4bdba03686",
    LATE/"labels.json":"7146e46a06c1d8e1de84687808145a64d25930baad360823e6ad4552aa4ec6a7",
}

def require(test, message):
    if not test:
        raise ValueError(message)

def by_id(rows, name, identity):
    matches = [x for x in rows if x[name] == identity]
    require(len(matches) == 1, f"{identity} has {len(matches)} signed records")
    return matches[0]

def earlier_economy(r):
    """Return historical arm-1 minus arm-0 net deposit and tree credits; bind its RNG pair."""
    name, group = r["case"],r["group"]
    h = EXP / "learned_carbon_observation"
    if group == "active_single":
        d = C.read(h/"native_training/labels.json.gz")
        x = by_id(d["roots"],"root_id",name)
        require(x["shared_receipt_sha"] and x["shared_snapshot_sha"],name)
        components = {k:v["delta_cash_margin"] for k,v in x["cash_components_proposal_minus_incumbent"].items()}
        scores = [x["arms"][a]["official_score_by_seat"] for a in ("parent","student")]
        binding = x["shared_snapshot_sha"]
    elif group == "decoder_single":
        x = by_id(C.read(h/"decoder/real_roots/verification.json")["cases"],"id",name)
        binding = x["verified_common_prefix"]["snapshot_sha"]
        components = {k:v["margin"] for k,v in x["economic_components"].items()}
        scores = [x["arms"][a]["official_scores"] for a in ("parent","student")]
    elif group == "encoder_single":
        x = by_id(C.read(h/"native_response/result.json")["maps"],"seed",r["seed"])
        require(x["changed"] and x["opposing_rng_at_root_identical"] and x["seat"] == r["seat"], name)
        binding = x["arms"]["parent"]["shared"]
        require(binding == x["arms"]["student"]["shared"], name)
        scores = [x["arms"][a]["score"] for a in ("parent","student")]
        components = x["component_net_own_minus_rival_student_minus_parent"]
    elif group == "onpolicy":
        base = EXP/"onpolicy_native/response"
        ledger = C.read(base/"economy.json.gz")
        require(ledger["roots"] == C.sha(base/"roots.json") and ledger["signed"] == C.sha(base/"result.json"), name)
        x = by_id(ledger["rows"],"id",name)
        tickets = [C.read(base/f"{name}_{a}.json.gz") for a in ("parent","student")]
        require(tickets[0]["rng"] == tickets[1]["rng"] and tickets[0]["binding"]["warm"] == tickets[1]["binding"]["warm"],name)
        require(x["root_identical_enemy_order"] and all(z["binding"]["roots"] == ledger["roots"] for z in tickets),name)
        scores = [w["score"] for w in tickets]
        binding = {"rng":tickets[0]["rng"],"warm":tickets[0]["binding"]["warm"],"roots":ledger["roots"]}
        components = {k:v["relative"] for k,v in x["cash_credits_student_minus_parent"].items()}
    else:
        raise ValueError(group)
    require(scores == r["scores"] and binding is not None,name)
    trees = [k for k in ("ground_rent","enemy_cargo_tax","tree_absorption") if k in components]
    require(trees and "own_delivery" in components, name)
    return dict(deposit_margin=components["own_delivery"], tree_income_margin=sum(components[k] for k in trees),
                ledger_tree_keys=trees, common_draw=binding)

def cases():
    for path,expected in INPUTS.items():
        require(C.sha(path) == expected,f"Changed frozen historical file {path.name}")
    historical = [r for r in C.read(OLD/"commands.json.gz") if r["continuation"] == PARENT]
    audited = [r for r in C.load_cases()[0] if r["continuation"] == PARENT]
    require(len(historical) == len(audited) == 9,"Nine same-continuation early pairs required")
    for r in historical:
        src = by_id(audited,"case",r["case"])
        require(all(src[k] == r[k] for k in ("group","source","source_sha256","archived_commands","scores","seed","seat","t")),r["case"])
        r["econ"] = earlier_economy(r)
        r["parent_expected"] = r["free_commands"]["center"]
    registry,ledger = C.read(LATE/"roots.json"),C.read(LATE/"labels.json")
    require(ledger["original_sha256"] == EXAMINER and ledger["packages"]["parent"]["sha256"] == PARENT
            and ledger["roots_sha256"] == C.sha(LATE/"roots.json"),"Late receipt not bound to original and parent")
    new = []
    for r in registry["roots"]:
        name = r["id"]
        x = by_id(ledger["rows"],"id",name)
        require(all(x[k] == r[k] for k in ("seed","seat","terrain","t","source_sha256","prefix_sha256","public_sha256")),name)
        folder = LATE/"native"/f"{r['seed']}_{r['seat']}"/name
        arm_keys = ("parent","student")
        net_deposit, net_tree = [],[]
        for arm in arm_keys:
            a = x["arms"][arm]
            path = folder/(arm+".json.gz")
            require(C.sha(path) == a["receipt_sha256"],name)
            ticket = C.read(path)
            require(ticket["identity"]["packages"]["parent"] == PARENT and ticket["identity"]["original"] == EXAMINER
                    and ticket["snapshot_sha"] == x["snapshot_sha256"] and ticket["shared_sha"] == x["shared_file_sha256"]
                    and ticket["root_sha"] == x["root_file_sha256"] and ticket["official_scores"] == a["official_scores"]
                    and C.clean(ticket["tape"][0][r["seat"]]) == C.clean(r[arm])
                    and a["subsequent_parent_deployments_verified"] == len(ticket["tape"])-1, name)
            terms=a["horizons"]["terminal"]
            credits=terms["by_physical_seat"]
            tree=[side.get("ground_rent",0)+side.get("cargo_tax",0) for side in credits]
            net_tree.append(tree[r["seat"]]-tree[1-r["seat"]])
            net_deposit.append(terms["deposit_margin"])
        new.append(dict(case=name,group="late_return",continuation=PARENT,archived_commands=[C.clean(r[a]) for a in arm_keys],
                        scores=[x["arms"][a]["official_scores"] for a in arm_keys],parent_expected=C.clean(r["parent"]),
                        public_sha256=r["short_public_sha256"],prefix_sha256=r["prefix_sha256"],
                        econ=dict(deposit_margin=net_deposit[1]-net_deposit[0],tree_income_margin=net_tree[1]-net_tree[0],
                                  ledger_tree_keys=["ground_rent","cargo_tax"],common_draw=x["snapshot_sha256"]),
                        **{k:r[k] for k in ("source","seed","seat","source_sha256","t","terrain")}))
    rows = historical+new
    require(len(new) == 7 and len({(r["source_sha256"],r["seat"],r["t"],r["prefix_sha256"]) for r in rows}) == 16,"Wrong fixed root count")
    return rows

def compare_commands(r, before, after):
    frozen = r["archived_commands"]
    require(before == r["parent_expected"] and before in frozen and frozen[0] != frozen[1],r["case"])
    i = frozen.index(before)
    j = frozen.index(after) if after in frozen else None
    scores = r["scores"]
    s = r["seat"]
    margin = [round(v[s]-v[1-s],2) for v in scores]
    uid = set(before) | set(after)
    changes = {k:[before.get(k,"NONE"),after.get(k,"NONE")] for k in sorted(uid) if before.get(k) != after.get(k)}
    direction = j-i if j is not None else None
    def delta(field):
        return None if direction is None else round(direction*r["econ"][field],6)
    return dict(case=r["case"],group=r["group"],seed=r["seed"],seat=s,t=r["t"],terrain=r["terrain"],
                source=r["source"], source_sha256=r["source_sha256"], prefix_sha256=r["prefix_sha256"],
                public_sha256=r["public_sha256"],parent=before,candidate=after,unit_changes=changes,
                measured_actions=frozen, measured_scores_by_physical_seat=scores,exact_archived_arm=dict(parent=i,candidate=j),
                status="unchanged" if i==j else ("supported_change" if j is not None else "unmeasured_change"),
                parent_official_margin=margin[i], candidate_official_margin=margin[j] if j is not None else None,
                delta_official_margin=round(margin[j]-margin[i],2) if j is not None else None,
                delta_net_normal_deposits=delta("deposit_margin"),delta_net_tree_income=delta("tree_income_margin"),
                tree_ledger_categories=r["econ"]["ledger_tree_keys"],common_draw=r["econ"]["common_draw"])

def measure(rows, candidate):
    groups = defaultdict(list)
    for r in rows:groups[(r["source"],r["seat"],r.get("custom_prefix"))].append(r)
    models = [C.module(C.EXP/"candidate.py","parent"),C.module(candidate,"league")]
    out=[]
    for (source,seat,override),batch in groups.items():
        path = ROOT/source
        require(C.sha(path) == batch[0]["source_sha256"],source)
        raw = C.read(path)
        C.validate_examiner_configuration(raw["configuration"])
        require(raw["examiner_md5"] == "57f9a594cce4f2ec9e3573d8d9d26ca1",source)
        history=C.read(ROOT/override)["prefix"] if override else raw["actions"]
        cfg={k:v for k,v in raw["game_configuration"].items() if k!="randomSeed"}
        env=C.begin(raw)
        at={r["t"]:r for r in batch}
        require(len(at)==len(batch),source)
        for t in range(max(at)+1):
            obs=C.public(env,seat)
            actions=[C.clean(m.agent(obs,cfg)) for m in models]
            if t in at:
                r=at[t]
                require(C.digest(history[:t])==r["prefix_sha256"] and
                        C.digest({k:v for k,v in obs.items() if k!="remainingOverageTime"})==r["public_sha256"],r["case"])
                out.append(compare_commands(r,*actions))
            if t<max(at):env.step(history[t])
    return out

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--verify-inputs",action="store_true")
    parser.add_argument("--self-check", action="store_true",help="Check complete deployed histories with parent on both sides; output no result file.")
    args=parser.parse_args()
    rows=cases()
    require(C.sha(EXP/"candidate.py")==PARENT,"Wrong parent")
    if args.verify_inputs:
        print("Locked early/late original observations",Counter(r["group"] for r in rows),flush=True)
        return
    if args.self_check:
        controls=measure(rows,EXP/"candidate.py")
        require(len(controls)==16 and all(x["status"]=="unchanged" and x["delta_official_margin"]==0 for x in controls),"Unexpected parent mismatch")
        print("Parent replay verified at 16/16 fixed roots", flush=True)
        return
    fit=C.read(HERE/"fit.json")
    candidate=HERE/"candidate.py"
    require(fit["parent_sha256"]==PARENT and C.sha(candidate)==fit["package_sha256"]
            and C.sha(HERE/"decision_audit.json.gz")==fit["audit_sha256"],"Unbound quantized candidate")
    scored=measure(rows,candidate)
    summary=Counter(r["status"] for r in scored)
    measured=[r for r in scored if r["status"]=="supported_change"]
    out=dict(parent_sha256=PARENT,candidate_sha256=C.sha(candidate),fit_sha256=C.sha(HERE/"fit.json"),
             probe_source_sha256=C.sha(Path(__file__)),fixed_sources={str(k.relative_to(ROOT)):v for k,v in INPUTS.items()},
             scope="Sixteen historical paired forks, the corresponding observed native draw per fork and common 8da4 continuation. Only the first complete joint action is replaced. Missing outcomes remain null. These exposed positions do not estimate full-game wins or determine candidate selection.",
             counts=dict(summary),covered_changed_maps=len({r["terrain"] for r in measured}),rows=scored)
    tmp=HERE/"transfer_probe.tmp.gz"
    C.write(tmp,out)
    tmp.replace(HERE/"transfer_probe.json.gz")
    print(dict(summary),flush=True)

if __name__=="__main__":
    main()
