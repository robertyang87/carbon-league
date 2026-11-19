"""Check the semantic selector only against four already-recorded training trajectories."""
import collections
import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np
import select_root as selector

HERE = Path(__file__).resolve().parent
MENU = HERE.parent
spec = importlib.util.spec_from_file_location("sapling_economy_catalog_audit",MENU/"audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
require = audit.need
CATEGORIES = ("plant_or_capture","freight")


def frozen(path, sha):
    blob = path.read_bytes()
    require(audit.sha(blob)==sha, f"Changed input {path.name}")
    return json.loads(gzip.decompress(blob)) if path.suffix==".gz" else json.loads(blob)


def named_payment(row, choice, uid, kind):
    event = {"plant":"plant","capture":"seize","deposit":"own_delivery"}[kind]
    account = choice["account"]["cash_events"]
    amounts = [float(x[2]) for x in account if x[0]==event and x[1]==row["seat"] and x[3]==uid]
    require(len(amounts)<=1 and all((p>0 if kind=="deposit" else p<0) for p in amounts),
            "Unexpected payment sign or duplicate focal event")
    return sum(amounts)


def historical(catalog, record, ledger, cfg, times):
    """Apply the selector to the historical natural order. Reveal effects only after selection."""
    comp = catalog["comparison_after_freeze"]
    public = record["public_observation"]
    order = record["issued"]
    require(public["step"]==catalog["step"] and order==ledger["recorded_own_order"]==comp["issued"] and
            record["public_observation_sha256"]==catalog["public_observation_sha256"],"Archived native choice")
    old = {"p":record["probabilities"],"pick":record["choice"]} if record["supported"] else None
    choices = {audit.key(c["order"]):c for c in ledger["choices"]}
    out = []
    for category in CATEGORIES:
        event = selector.pick(catalog["catalog"], public, cfg, order, category, old)
        row = dict(seed=catalog["seed"],seat=catalog["seat"],step=catalog["step"],category=category,
                   on_registered_turn=catalog["step"] in times,native_A_differs_from_greedy=order!=comp["original_options"][0],
                   selected=event is not None)
        if event:
            arm = event["effect_arm"]
            require(event["options"][0]==order and event["options"][1]!=order and
                    all(audit.key(o) in choices for o in event["options"]), "Different physical menu")
            payments=[named_payment(row,choices[audit.key(o)],event["focal_uid"],event["kind"]) for o in event["options"]]
            row.update(kind=event["kind"],uid=event["focal_uid"],orders_sha256=[selector.digest(o) for o in event["options"]],
                       named_payments=payments,effect_arm=arm,verified_focal_effect=payments[arm]!=0 and payments[1-arm]==0,
                       native_A_probability=float(record["probabilities"][record["choice"]]),
                       selection_sha256=selector.digest(event))
        out.append(row)
    return out


def live_callbacks(student,obs,cfg,catalog):
    """Compare ordinary sampling to both issue callbacks, starting from exactly the same warm actor state."""
    ss,run = selector.student_state,selector.runner
    saved = ss.snapshot(student)
    ordinary,native=run.action(student,obs,cfg)
    natural_post=ss.fingerprint(student)
    rng_after=selector.play.student_history.behavior_rng_snapshot(student)
    expected_options=catalog["comparison_after_freeze"]["original_options"]
    require((native["options"] if native else [ordinary["order"]])==expected_options,"Warm native support differs")
    rows=[]
    for category in CATEGORIES:
        ss.restore(student,saved)
        own,selection=selector.probe(student,obs,cfg,category)
        require(own["order"]==ordinary["order"] and ss.fingerprint(student)==natural_post and
                selector.play.student_history.behavior_rng_snapshot(student)==rng_after,"Probe changed natural A, state or RNG")
        expected=selector.pick(catalog["catalog"],obs,cfg,ordinary["order"],category,native)
        require((selection is None)==(expected is None),"Live and archived catalog selection disagree")
        if selection:
            require(all(selection[k]==v for k,v in expected.items() if k!="native_A_probability") and
                    selection["native_A_probability"]==(native["p"][native["pick"]] if native else 1.),
                    "Different callback distribution or witness")
            for i in (0,1):
                ss.restore(student,saved)
                returned,event=selector.issue_student(student,obs,cfg,i,category,selection)
                require(returned["order"]==selection["options"][i] and event["pick"]==i and
                        ss.fingerprint(student)==natural_post and selector.play.student_history.behavior_rng_snapshot(student)==rng_after,
                        "Fork did not reissue the exact selection with the natural hidden state and stream")
        rows.append(dict(step=obs["step"],category=category,selected=selection is not None,
                         diagnostic_draw_differs_from_greedy=ordinary["order"]!=expected_options[0]))
    return rows


def main():
    plan=json.loads((HERE/"protocol.json").read_bytes())
    protocol,games=audit.inventory()
    roots=frozen(MENU/"roots.json.gz",plan["static_inputs"]["reported_48_roots"])
    effects_sha="3b832cba36e79f49bef59fa611734c1d7fee9a693cc5fa801950ce6a4df9d7f9"
    effects=frozen(MENU/"verification/official_orders.json.gz",effects_sha)
    data={(r["seed"],r["seat"],r["step"]):r for t in roots["tickets"] for r in t["rows"]}
    effects={(r["seed"],r["seat"],r["step"]):r for r in effects["roots"]}
    history=[];callbacks=[];counts=[]
    for ep in games:
        seat,seed=ep["seat"],ep["seed"];events={r["step"]:r for r in ep["events"]}
        bank=[5.,5.];n=0
        with audit.trajectory(ep,audit.LEAGUE/protocol["behavior"]["package"],plan["student_sha256"]) as (live,env,cfg,tape,student):
            for t,blob in enumerate(tape):
                fr=json.loads(blob)
                obs=[live.observe(env,s,bank[s]) for s in (0,1)]
                require(t==fr["step"]==env.states[0].observation.step and
                        [audit.sha(live.encoded(o)) for o in obs]==fr["public_observation_sha256"],"Wrong public replay")
                if t in events:
                    key=(seed,seat,t)
                    require(events[t]["public_observation"]==obs[seat],"Archived acting observation")
                    history.extend(historical(data[key],events[t],effects[key],cfg,plan["probe_turns"]))
                    callbacks.extend(dict(seed=seed,seat=seat,**r) for r in live_callbacks(student,obs[seat],cfg,data[key]))
                else:
                    student.act(obs[seat],cfg)
                    if t==0:
                        # A reproducible diagnostics-only draw stream. The archive preserves its actual issued A separately.
                        student.mod._behavior_rng=np.random.default_rng(280061)
                env.step([c["response"]["action"] for c in fr["calls"]],[])
                bank=fr["overage_remaining_seconds"];n+=1
            require(env.done and n==ep["official_transitions"] and [s.reward for s in env.states]==ep["official_scores_by_seat"],
                    "Official terminal or physical score differs")
        counts.append(n)
        print(seed,seat,n,flush=True)
    require(len(data)==len(effects)==48 and len(callbacks)==len(history)==96,"Incomplete four-game frozen census")
    eligible={}
    for cat in CATEGORIES:
        h=[r for r in history if r["category"]==cat and r["on_registered_turn"]]
        valid=[r for r in h if r.get("verified_focal_effect")]
        eligible[cat]=dict(scheduled_roots=len(h),physically_selected=sum(r["selected"] for r in h),
                           verified_named_payment_roots=len(valid),named_payments_by_kind=dict(collections.Counter(r["kind"] for r in valid)))
    receipt=dict(scope="Archived 460/461 draw_a, two seats. Only recorded commands advanced the official environment. Diagnostic reconstructions use a fixed RNG; historical qualifying checks use the actual archived natural A.",
                 selector_sha256=audit.sha((HERE/"select_root.py").read_bytes()),fixture_sha256=audit.sha(Path(__file__).read_bytes()),
                 protocol_sha256=audit.sha((HERE/"protocol.json").read_bytes()),
                 catalog_sha256=plan["static_inputs"]["reported_48_roots"],branches_sha256=effects_sha,
                 official_transitions=sum(counts),public_hashes=2*sum(counts),scheduled_categories=eligible,
                 callback_forks_verified=2*sum(r["selected"] for r in callbacks),tested_natural_calls=48,
                 historical=history,diagnostic=callbacks,new_games=0)
    audit.publish(HERE/"selector_fixture.json",json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False).encode()+b"\n")
    print(eligible,flush=True)


if __name__=="__main__":
    main()
