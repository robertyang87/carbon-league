"""Independently settle fixed native tapes. This process never calls the examiner."""
import argparse
from collections import Counter
import gzip
import os
from pathlib import Path

import numpy as np
from collect import HERE, CARBON, EXP, PLAN_SHA, K, R


def digest(value):
    return K.digest(K.compact(value))


def summaries(calls):
    depth = R.native_clock.summarize(calls, .58, 8)
    zero = [v["step"] for v in calls if v["kind"] == "depth_zero_return_base"]
    wall = [v["public_wall_seconds"] for v in calls]
    return dict(calls=len(calls), depth=depth, zero_fallback_steps=zero,
                max_public_wall=max(wall, default=None),
                mean_public_wall=sum(wall)/len(wall) if wall else None)


def validate_clock(clock, start, stop):
    calls = clock["calls"]
    assert [c["step"] for c in calls] == list(range(start, stop))
    for c in calls:
        assert c["process_cpu_seconds"] >= 0 and c["public_wall_seconds"] >= 0
        assert c["kind"] in ("searched","single","depth_zero_return_base")
        assert (c["kind"] == "searched") == (c["max_depth"] > 0)
        assert (c["kind"] == "single") == (c["candidates"] == 1)
    result = summaries(calls)
    assert result["depth"] == clock["summary"]
    return result


def verify_identities(case):
    plan=R.load(HERE/"native_protocol.json"); assert R.sha(HERE/"native_protocol.json")==PLAN_SHA
    assert R.sha(HERE/"roots.json")==plan["roots_sha256"] and case in R.load(HERE/"roots.json")["cases"]
    assert R.sha(HERE/"protocol.json")==plan["registration_sha256"]
    assert R.sha(HERE/"source_roles.json")==plan["source_roles_sha256"]
    folder = HERE/"native"/case["id"]
    bank = R.load(folder/"shared.json.gz")
    identity = bank["identity"]
    witness = R.load(HERE/case["prefix_file"])
    assert R.sha(HERE/case["prefix_file"])==case["prefix_file_sha256"]
    assert digest(witness["both_normal_commands_at_each_public_state"]) == case["seq_sha256"]
    assert bank["case"]==case and bank["prefix_sha"]==case["prefix_sha256"]
    assert bank["prefix"]==witness["physical_fixed_history"] and digest(bank["prefix"])==bank["prefix_sha"]
    assert identity["plan"]==PLAN_SHA and identity["roots"]==plan["roots_sha256"]
    assert identity["kernel"]==R.sha(Path(K.__file__)) and identity["runner"]==R.sha(HERE/"collect.py")
    assert identity["case"]==case["identity_sha256"] and identity["source"]==case["source_sha256"]
    assert identity["packs"]==dict(parent=R.sha(CARBON/"candidate.py"),student=R.sha(CARBON/"decoder/candidate.py"),
                                   continuation=R.sha(EXP/"candidate.py"))
    assert identity["packs"]["continuation"]==plan["continuation_sha256"]
    assert (identity["packs"]["parent"], identity["packs"]["student"])==tuple(plan["root_policies"][k] for k in ("parent_encoder_only","alternative_encoder_decoder"))
    assert identity["original"]==plan["examiner"]["sha256"]==R.sha(K.ORIGINAL)
    assert R.sha(R.ROOT/case["source"]) == case["source_sha256"]
    raw=R.load(R.ROOT/case["source"])
    assert raw["seed"]==case["seed"] and raw["initial_terrain_u8_sha256"]==case["terrain"]
    assert raw["actions"][:case["t"]]==bank["prefix"]
    assert len(bank["trace"])==case["t"]+1
    assert (bank["root"]["parent"],bank["root"]["student"])==(case["parent"],case["student"])
    assert K.digest(gzip.decompress((folder/"original_root.pickle.gz").read_bytes()))==bank["snapshot_sha"]
    assert bank["host"]["affinity"]==[identity["cpu"]]
    R.native_clock.validate_host(bank["host"],[identity["cpu"]])
    s=case["seat"]
    assert R.live.clean(raw["actions"][case["t"]][1-s])==case["root_enemy"]==bank["root"]["archived_enemy"]
    assert bank["root"]["enemy"]==bank["trace"][-1]["original_proposed"]==bank["trace"][-1]["original_executed"]
    return raw, folder, bank, witness


def court_state(env,sim):
    obs = env.states[0].observation
    assert obs.step == sim.step_count
    np.testing.assert_allclose(obs.full_carbon,sim.carbon,atol=.011,rtol=0)
    alt=sim.obs(0)["players"]
    for team,second in zip(obs.players,alt,strict=True):
        assert abs(team[0]-second[0])<=.011
        assert team[1]==second[1] and team[3]==second[3]
        assert team[2].keys()==second[2].keys()
        for uid,(pos,cargo,role) in team[2].items():
            independent=second[2][uid]
            assert (pos,role)==(independent[0],independent[2]) and abs(cargo-independent[1])<=.011


def category_events(events):
    for kind,who,amount,*fields in events:
        if kind=="tree_absorption":
            yield who,"ground_rent",amount-fields[2]
            yield who,"enemy_cargo_tax",fields[2]
        else:
            yield who,kind,amount


def delta_components(ledgers,s):
    p,a=(ledgers[k] for k in ("parent","student"))
    out={}
    for cat in sorted(set().union(*(x.keys() for x in p+a))):
        own=a[s].get(cat,0)-p[s].get(cat,0)
        enemy=a[1-s].get(cat,0)-p[1-s].get(cat,0)
        out[cat]=dict(own=round(own,6),enemy=round(enemy,6),margin=round(own-enemy,6))
    return out


def replay(case, arm=None):
    raw,folder,bank,witness=verify_identities(case)
    s=case["seat"];op=1-s;t0=case["t"];cfg=raw["game_configuration"]
    root=bank["root"]
    prefix_clock = validate_clock(bank["clock"],0,t0+1)
    seq=witness["both_normal_commands_at_each_public_state"]
    ticket=None
    if arm:
        path=folder/(arm+".json.gz");ticket=R.load(path)
        assert ticket["identity"]==bank["identity"] and ticket["arm"]==arm and ticket["root_t"]==t0
        assert ticket["shared_sha"]==R.sha(folder/"shared.json.gz")
        assert ticket["state_sha"]==bank["state_sha"] and ticket["snapshot_sha"]==bank["snapshot_sha"]
        assert ticket["case_terrain"]==case["terrain"] and (ticket["seat"],ticket["seed"])==(s,case["seed"])
        assert ticket["tape"][0]==ticket["root_joint"]==K.physical(root[arm],root["enemy"],s)
        assert digest(ticket["tape"])==ticket["tape_sha"]
        actions=bank["prefix"]+ticket["tape"]
        suffix_clock=validate_clock(ticket["clock"],t0+1,len(actions))
        for host in (ticket["host_at_start"],ticket["host_at_end"]):
            R.native_clock.validate_host(host,[bank["cpu"]])
        assert len(ticket["own_public_seconds"])==len(ticket["suffix_observations"])==len(actions)-t0-1
    else:
        # Root observation only, no transition on either prospective arm.
        actions=bank["prefix"]+[None]
    paths={"continuation":EXP/"candidate.py"}
    if arm is None:
        paths.update(parent=CARBON/"candidate.py",student=CARBON/"decoder/candidate.py")
    models={k:R.live.module(p,"settle_decoder_"+k) for k,p in paths.items()}
    env=K.checked_env(raw)
    sim=R.FastCarbon(cfg).reset(seed=case["seed"])
    assert K.digest(bytes(map(int,sim.carbon)))==case["terrain"]
    sim.cash=R.CashEvents(sim.cash);initial=list(sim.cash)
    ledger=[Counter(),Counter()];suffix=[Counter(),Counter()]
    events=[];own_root_calls={};verified_post=0
    phases={};focal=case["all_changed_uids"][0]
    for t,joint in enumerate(actions):
        assert not env.done and t==sim.step_count
        court_state(env,sim)
        my,them=(K.public(env,who) for who in (s,op))
        hashes=[digest(my),digest(them)]
        if t<=t0:
            r=bank["trace"][t]; signed=seq[t]
            assert r["t"]==signed["t"]==t and [r["own_sha"],r["enemy_sha"]]==hashes
            assert digest({k:v for k,v in my.items() if k!="remainingOverageTime"})==signed["public_obs_sha256"]
            assert (r["parent"],r["student"])==(signed["parent"],signed["student"])
            if t<t0:assert r["original_executed"]==R.live.clean(joint[op])
            else:
                assert hashes==[root["own_sha"],root["enemy_sha"]]
                assert digest({k:v for k,v in my.items() if k!="remainingOverageTime"})==case["public_obs_sha256"]
            for k,mdl in models.items():
                cmd=R.live.clean(mdl.agent(my,cfg))
                assert cmd==r[k],(case["id"],arm,t,k)
                if t==t0:own_root_calls[k]=cmd
        else:
            assert hashes==ticket["suffix_observations"][t-t0-1]
            cmd=R.live.clean(models["continuation"].agent(my,cfg))
            assert cmd==R.live.clean(joint[s]),(case["id"],arm,t,"8da4 suffix")
            verified_post+=1
        if joint is None:break
        sim.cash.events.clear()
        sim.step(*joint);env.step(joint)
        post=env.states[0].observation.players
        cash=[float(p[0]) for p in post]
        if t>=t0:
            phase=t//50*50
            phases.setdefault(phase,[Counter(),Counter()])
            unit=sim.workers.get(focal)
            events.append(dict(t=t,joint=joint,cash=cash,root_worker=None if unit is None else [int(unit.pos),float(unit.carbon)],
                               workers=[len(team[2]) for team in post],trees=[len(team[3]) for team in post],
                               cargo=[sum(v[1] for v in team[2].values() if v[2]=="COLLECTOR") for team in post],
                               cash_events=[list(e) for e in sim.cash.events]))
        for who,kind,amount in category_events(sim.cash.events):
            ledger[who][kind]+=amount
            if t>=t0:
                suffix[who][kind]+=amount
                phases[phase][who][kind]+=amount
    if not arm:
        return dict(root=case["id"],sha256=R.sha(folder/"shared.json.gz"),parent_register=PLAN_SHA,
                    verified_prefix_steps=t0,normal_root_and_continuation_calls_each=t0+1,
                    root_normal_calls=own_root_calls,prefix_clock=prefix_clock,original_root_call=bank["clock"]["calls"][-1],
                    shared_state_sha=bank["state_sha"],snapshot_sha=bank["snapshot_sha"],
                    original_proposals_differ_from_forced_history=sum(z["original_proposed"]!=z["original_executed"] for z in bank["trace"][:t0]),
                    archived_enemy_same_as_live=root["archived_enemy"]==root["enemy"])
    court_state(env,sim)
    scores,cash,status=K.score(env)
    assert scores==ticket["official_scores"] and cash==ticket["physical_cash"] and status==ticket["status"]
    assert scores[s]-scores[op]==ticket["official_margin"]
    np.testing.assert_allclose([sum(c.values()) for c in ledger],np.subtract(sim.cash,initial),atol=1e-6,rtol=0)
    root_w=witness["root_public_observation"]["players"][s][2][focal]
    return dict(arm=arm,seed=case["seed"],seat=s,t=t0,focal_uid=focal,root_position_cargo_role=root_w,verified_transitions=len(actions),
                verified_suffix_8da4_calls=verified_post,stored_ticket_sha256=R.sha(path),
                official_scores=scores,physical_cash=cash,status=status,margin=ticket["official_margin"],ledger_by_physical_seat=ledger,
                root_to_terminal_ledger_by_physical_seat=suffix,by_50turn_ledger=phases,original_suffix=suffix_clock,
                root_call=bank["clock"]["calls"][-1],original_by_50turn={q:summaries([c for c in ticket["clock"]["calls"] if q<=c["step"]<q+50]) for q in phases},
                events=events)


def paired(case,results):
    p,a=results["parent"],results["student"]
    s=case["seat"];e=1-s
    ev={k:v.pop("events") for k,v in results.items()}
    delta=a["margin"]-p["margin"]
    accounting=HERE/"native"/case["id"]/"verified_events.json.gz"
    R.write(accounting,ev)
    horizons={}
    for h in (1,3,5,10,24,50,100):
        if min(len(v) for v in ev.values())<h:continue
        old,new=[ev[arm][h-1]["cash"] for arm in ("parent","student")]
        horizons[h]=dict(own=round(new[s]-old[s],2),enemy=round(new[e]-old[e],2),margin=round((new[s]-old[s])-(new[e]-old[e]),2))
    components=delta_components({k:v["root_to_terminal_ledger_by_physical_seat"] for k,v in results.items()},s)
    assert abs(sum(x["margin"] for x in components.values())-delta)<=.011 or any(v["official_scores"]!=v["physical_cash"] for v in results.values())
    phases=sorted(set(p["by_50turn_ledger"])|set(a["by_50turn_ledger"]))
    staged={q:delta_components({k:v["by_50turn_ledger"].get(q,[{},{}]) for k,v in results.items()},s) for q in phases}
    first_enemy=next((l["t"] for l,r in zip(ev["parent"][1:],ev["student"][1:]) if l["joint"][e]!=r["joint"][e]),None)
    focal={}
    for arm,tape in ev.items():
        died=next((x["t"] for x in tape if x["root_worker"] is None),None)
        uid=case["all_changed_uids"][0]
        deposits=[[x["t"],round(v[2],6)] for x in tape for v in x["cash_events"]
                  if v[0]=="own_delivery" and v[1]==s and len(v)>3 and v[3]==uid]
        focal[arm]=dict(position_cargo_after_root=tape[0]["root_worker"],
                        first_absent_poststep=died, later_own_deliveries=deposits)
    return dict(id=case["id"],seat=s,root=case["t"],physical_commands=case["physical_changed_workers"],arms=results,
                root_worker_trajectories=focal,
                first_later_enemy_difference=first_enemy,official_margin_student_minus_parent=round(delta,2),
                economic_components=components,horizons=horizons,component_delta_by_50turn=staged,
                economic_events=str(accounting.relative_to(HERE)),economic_events_sha=R.sha(accounting))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--case", choices=("0","1","all"), default="all")
    ap.add_argument("--stage", choices=("prefix","parent","student","report","all"), default="all")
    args=ap.parse_args()
    assert os.sched_getaffinity(0)=={0},"run pinned to idle/candidate-independent CPU0"
    cases=R.load(HERE/"roots.json")["cases"]
    if args.case!="all": cases=[cases[int(args.case)]]
    for case in cases:
        folder=HERE/"native"/case["id"]
        if args.stage in ("prefix","all"):
            d=replay(case)
            R.write(folder/"verified_prefix.json",d)
        for arm in ("parent","student"):
            if args.stage in (arm,"all"):
                d=replay(case,arm)
                R.write(folder/("verified_"+arm+".json.gz"),d)
    if args.stage in ("report","all"):
        assert len(cases)==2
        details=[]
        for case in cases:
            folder=HERE/"native"/case["id"]
            res={k:R.load(folder/("verified_"+k+".json.gz")) for k in ("parent","student")}
            # JSON serializes integer phase keys as strings; both branches have the same key type.
            result=paired(case,res)
            result["verified_common_prefix"]=R.load(folder/"verified_prefix.json")
            details.append(result)
        R.write(HERE/"verification.json",dict(plan=PLAN_SHA,roots=R.sha(HERE/"roots.json"),settlement_sha=R.sha(Path(__file__)),
            method="Independent reconstructed official referee, FastCarbon full board (cash, workers, trees, ground carbon), instrumented transaction ledger and separately invoked real 8da4 for every post-root own action. Original per-step calls are verified from signed raw tape/clock and snapshot bindings without a second stochastic examiner draw.",
            cases=details,full_game_wins_claimed=0))

if __name__=="__main__":main()
