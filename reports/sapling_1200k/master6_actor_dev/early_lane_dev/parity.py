"""Exercise the immutable single-file entry point against all independently generated held-out orders."""
from collections import Counter
import hashlib
import importlib.util
import json
import time

from train import HERE, PROTOCOL, batch, check, extract


def load_pack(path):
    spec = importlib.util.spec_from_file_location("portable_early_lane_candidate",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clean(commands):
    return {uid:cmd for uid,cmd in commands.items() if cmd and cmd != 'NONE'}


def game(spec, signed_result, pack):
    agent = load_pack(pack)
    recorded = extract.ROOT/spec['raw_path']
    check(extract.sha(recorded)==spec['raw_sha256'], 'original game changed')
    raw=extract.load(recorded); cfg=raw['game_configuration']; seat=spec['seat']
    actual_config = dict(cfg,randomSeed=None)
    sim=extract.FastCarbon(cfg).reset(seed=spec['seed'])
    check(hashlib.sha256(bytes(int(v) for v in sim.carbon)).hexdigest()==spec['map_sha256'], 'terrain mismatch')
    src=HERE/f"validation_{spec['seed']}_{seat}.json.gz"
    check(extract.sha(src)==signed_result, 'mechanism evidence changed')
    v=extract.load(src); decisions={r['t']:r for r in v['decisions']}
    tally=Counter(); diff=[]; elapsed=[]
    for t,joint in enumerate(raw['actions']):
        obs=sim.obs(seat)
        agent._EARLY_TRACE = None
        start=time.perf_counter()
        answer=clean(agent.agent(obs,actual_config))
        elapsed.append(time.perf_counter()-start)
        trace=agent._EARLY_TRACE
        check(trace['step']==t and trace['seat']==seat, 'stale or missing pack telemetry')
        unchanged=clean(joint[seat]); expected=unchanged.copy()
        check(clean(trace['v1_commands'])==unchanged, f"v1 differs from actual source at {t}")
        row=decisions.get(t); candidate=row['candidate'] if row else None
        dec=trace['decision']; fire=bool(row and row['issued'])
        check(dec['candidate']==candidate, f"candidate, cells or scores differ at {spec['seed']}/{seat}/{t}")
        check(dec['triggered']==fire, f"trigger differs at {t}")
        if fire:
            uid=candidate['own_uid']; changed=candidate['order']
            if changed=='NONE': expected.pop(uid,None)
            else: expected[uid]=changed
            diff.append(dict(t=t,uid=uid,order=changed,old=candidate['old'],signature=row['signature']))
        check(clean(dec['commands'])==expected==answer, f"emitted orders differ at {t}")
        tally['v1_order_agreements']+=1
        tally['all_order_agreements']+=1
        tally['candidate_rows_and_scores_agree']+=candidate is not None
        tally['overrides_agree']+=fire
        # Advance the signed factual transcript at every root. Divergent games were not played.
        extract.step(sim,sim.full_observation(),cfg,joint)
    check([round(v,2) for v in sim.cash]==spec['scores'], 'terminal score changed')
    check(dict(tally)['all_order_agreements']==spec['steps'] and len(diff)==v['counts']['issued'], 'missing or extra root')
    return dict(seed=spec['seed'],seat=seat,source_sha256=spec['raw_sha256'],map_sha256=spec['map_sha256'],
                outcome=spec['scores'],counts=dict(tally),changes=diff,
                locally_observed_single_core_wall=dict(first=elapsed[0],p50=sorted(elapsed)[len(elapsed)//2],max=max(elapsed)))


def main():
    pack=HERE/'candidate.py'; sh=extract.sha(pack); gate=extract.load(HERE/'gate.json')
    check(gate['decision']=='pass_for_portable_parity' and all(gate['gates'].values()),
          'construction not eligible for packaging')
    plan=extract.load(HERE/'protocol.json'); prefix=(extract.ROOT/plan['parent_pack']).read_bytes()
    check(hashlib.sha256(prefix).hexdigest()==plan['parent_sha256'] and pack.read_bytes().startswith(prefix)
          and pack.stat().st_size<=10_000_000, 'package identity, parent, or size limit')
    root=HERE/'parity_rows';root.mkdir(exist_ok=True)
    evidence=[]
    for n in (4,5):
        for s in batch(n):
            p=root/f"{s['seed']}_{s['seat']}.json"
            resultid=gate['covered_artifacts'][f"{s['seed']}:{s['seat']}"]
            ident=dict(candidate_sha256=sh, validation_sha256=resultid)
            if p.exists():
                old=extract.load(p)
                check(all(old[k]==v for k,v in ident.items()), 'stale partial parity file')
            else:
                old={**game(s,resultid,pack), **ident};extract.write(p,old)
            evidence.append(old)
            print(p.name,old['counts'],flush=True)
    total=Counter()
    for row in evidence: total.update(row['counts'])
    check(total['overrides_agree']==gate['overall']['issued'] and total['all_order_agreements']==gate['overall']['source_turns']
          and total['candidate_rows_and_scores_agree']==gate['overall']['candidate'], 'incomplete heldout parity')
    extract.write(HERE/'parity.json',dict(protocol_sha256=PROTOCOL,candidate_sha256=sh,parent_sha256=plan['parent_sha256'],
                  gate_sha256=extract.sha(HERE/'gate.json'),pack_bytes=pack.stat().st_size,
                  weights_sha256=extract.sha(HERE/'weights.json'),evidence={f"{g['seed']}:{g['seat']}":extract.sha(root/f"{g['seed']}_{g['seat']}.json") for g in evidence},
                  total=dict(total),result='pass',scope='Exact selected candidate, scores, original inference and emitted orders on signed v1 histories; original enemy did not observe alternative subsequent turns.'))
    print('Portable parity PASS',total,flush=True)


if __name__=='__main__':
    main()
