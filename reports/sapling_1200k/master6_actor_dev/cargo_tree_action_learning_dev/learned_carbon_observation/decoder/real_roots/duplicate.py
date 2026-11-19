"""Audit registered root identities using prefix material only. No new result files are inputs."""
from collections import Counter, defaultdict
from functools import lru_cache
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
CARBON=HERE.parents[1]
CARGO=HERE.parents[2]
ACTOR=CARGO.parent
ROOT=HERE.parents[6]
sys.path.insert(0,str(CARBON/'active_roots'))
from select_roots import streaming_archive, begin, public

SNAPSHOT="3d3068425b066042345d3f242c3c5f968129d310"
def sha(b):return hashlib.sha256(b).hexdigest()
def digest(x):return sha(json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode())
def read(p):
    assert "native_training" not in p.parts
    b=p.read_bytes()
    return json.loads(gzip.decompress(b) if p.suffix==".gz" else b)
def rel(p):return str(p.relative_to(ROOT))
def clean(team):return {k:v for k,v in team.items() if v and v!="NONE"}


def main():
    current=read(HERE/"roots.json")["cases"]
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only',SNAPSHOT,'reports/sapling_1200k'],cwd=ROOT).decode().splitlines()
    historic=[ROOT/x for x in paths if '/native_training/' not in x and '/decoder/' not in x]
    terrains={c["seed"]:c["terrain"] for c in current}
    known_src={}
    for p in historic:
        if any(p.name.startswith(f"raw_{seed}_") for seed in terrains) and p.name.endswith(".json.gz"):
            known_src[sha(p.read_bytes())]=p
    sources={}
    by_group=defaultdict(list)
    def manifest(p):
        sources[rel(p)]=sha(p.read_bytes())
        return read(p)
    def add(group, **r):
        by_group[group].append(r)

    signed=manifest(ACTOR/'joint_response_critic_dev/signed_roots.json.gz')["rows"]
    idx={r["case"]:r for r in signed}
    for r in signed:
        add("signed108",**{k:r[k] for k in ('seed','seat','t','case','source','source_sha256','terrain','public_root_sha256')},arms=[r["parent"],r["alternative"]])
    for f in ("signed","earlier","windows"):
        for r in manifest(CARGO/'paired_response_prefix'/f'{f}.json.gz')["rows"]:
            if f=="signed":
                ref=idx[r['case']]
                assert all(r[k]==ref[k] for k in ('seed','seat','terrain','source_sha256'))
                assert r['start']==ref['t']
                continue
            add('paired_'+f,seed=r['seed'],seat=r['seat'],t=r['start'],case=r['case'],source=r['source'],source_sha256=r['source_sha256'],terrain=r['terrain'],prefix_sha256=r['prior_archived_joint_sha256'],first_exposure=r['first_actual_intervention'])
    for r in manifest(CARBON/'active_roots/roots.json')['cases']:
        add("preselected_four",**{k:r[k] for k in ('seed','seat','t','source','source_sha256','terrain','prefix_sha256','public_obs_sha256')},arms=[r['incumbent'],r['alternative']])
    for r in manifest(CARBON/'action_preference/extra_sources.json')['rows']:
        add("additional_early_live16",**{k:r[k] for k in ('seed','seat','t','source','source_sha256','terrain')},case=r['path'],arms=r['root_commands'])
    for r in manifest(CARBON/'native_response/protocol.json')['cases']:
        shared=manifest(CARBON/'native_response'/f"{r['seed']}_{r['seat']}"/'shared.json.gz')
        add('prior_carbon_native',seed=r['seed'],seat=r['seat'],terrain=r['terrain'],t=shared['root']['t'] if shared['root'] else None,
            case=rel(CARBON/'native_response'/f"{r['seed']}_{r['seat']}"),custom_prefix=True,prefix_sha256=digest(shared['prefix']),prefix_length=len(shared['prefix']),
            arms=[shared['root']['parent'],shared['root']['student']] if shared['root'] else None)
    for p in historic:
        if p.name=='live_protocol.json':
            r=manifest(p)
            assert 'root_t' in r
            rec=r.get('record') or {}
            add("live_protocol_census",seed=r['seed'],seat=r['seat'],t=r['root_t'],case=rel(p),source=rec.get('source'),source_sha256=rec.get('source_sha256'),family=r.get('family'))
    safety=manifest(CARGO/'counterfactual_safety/roots.json.gz')
    for r in safety:
        add("single_worker_proxy2614",seed=r['seed'],seat=r['own_seat'],t=r['t'],terrain=r['terrain'],source_sha256=r['raw'],case=r['uid']+'/'+r['tag'])

    # Source registration files for study-specific formats, when the two seeds occur in their metadata.
    # Some have no completed native outcomes, so store them as registrations, never measured paired payoffs.
    seeds={str(c['seed']).encode() for c in current}
    def selected(row,p):
        if not isinstance(row,dict):return
        if isinstance(row.get('registered_case'),dict):
            selected(row['registered_case'],p)
            return
        rec=row.get('source',{})
        info=rec if isinstance(rec,dict) else {}
        body=row.get('root',row)
        if not isinstance(body,dict):return
        seed=row.get('seed',info.get('seed'));seat=row.get('seat',info.get('seat'))
        t=body.get('t',body.get('step'))
        if seed is None or seat is None or not isinstance(t,int):return
        source=row.get('raw', rec if isinstance(rec,str) else info.get('raw'))
        sc=row.get('source_sha256',row.get('raw_sha256',info.get('raw_sha256')))
        add("other_selection_registrations",seed=seed,seat=seat,t=t,source=source,source_sha256=sc,case=rel(p),terrain=row.get('terrain',row.get('map',info.get('map'))))
    selector_files=[]
    for p in historic:
        if ('learned_carbon_observation' in p.parts or p.suffix!='.json' or
            not (p.name=='roots.json' or p.name.startswith('selection') and p.name.endswith('.json'))):
            continue
        if not any(s in p.read_bytes() for s in seeds):continue
        selector_files.append(rel(p))
        doc=manifest(p)
        if isinstance(doc,list):
            for r in doc:selected(r,p)
        elif 'root' in doc:
            selected(doc,p)
        else:
            for group in ('cases','roots','selected'):
                for r in doc.get(group,[]):selected(r,p)

    @lru_cache(None)
    def history(path,n):
        meta,it=streaming_archive(path)
        try:
            seq=list(itertools.islice(it,n))
            assert len(seq)==n
            return meta,seq
        finally:it.close()
    output=[]
    prior_eval={rel(p):sha(p.read_bytes()) for p in (CARBON/"normal_summary.json",CARBON/"other_held_actions.json",HERE.parent/"held_turns.json.gz")}
    for c in current:
        prefix=read(HERE/c['prefix_file'])['physical_fixed_history']
        assert len(prefix)==c['t'] and digest(prefix)==c['prefix_sha256']
        own_source=ROOT/c['source'];assert sha(own_source.read_bytes())==c['source_sha256']
        meta,seq=history(own_source,c['t'])
        assert seq==prefix and meta['initial_terrain_u8_sha256']==c['terrain']
        results={}
        for group,ls in by_group.items():
            count=Counter(); details=[]
            for r in ls:
                if r.get('terrain') and r['terrain']!=c['terrain']:continue
                if r['seed']!=c['seed']:continue
                count['same_terrain']+=1
                s=r.get('source_sha256')
                if s==c['source_sha256']:count['same_source_archive_any_seat']+=1
                if r["seat"]!=c["seat"]:continue
                count['same_terrain_seat']+=1
                if r['t'] is None:continue
                n=min(r['t'],c['t'])
                record={k:r[k] for k in ('case','t','source_sha256','first_exposure','family') if r.get(k) is not None}
                record['same_archive_bytes']=s==c['source_sha256'] if s else None
                if s==c['source_sha256'] and r['t']!=c['t']:count['same_archive_and_seat_other_turn']+=1
                if r['t']==c['t']:count['same_terrain_seat_t']+=1
                path=ROOT/r['source'] if r.get('source') else known_src.get(s)
                if path and path.is_file() and not r.get('custom_prefix'):
                    if s:assert sha(path.read_bytes())==s
                    record['source']=rel(path)
                    m,prev=history(path,n)
                    assert m['initial_terrain_u8_sha256']==c['terrain']
                    oldhash=digest(prev);newhash=digest(prefix[:n])
                    record['compared_prefix_steps']=n
                    record['common_until_first_differing_step']=next((i for i,(a,b) in enumerate(zip(prev,prefix)) if a!=b),n)
                    record.update(old_prefix_through_comparison_sha256=oldhash,new_prefix_through_comparison_sha256=newhash)
                    if oldhash==newhash:count['same_actions_through_earlier_registered_turn']+=1
                    if r['t']==c['t']:
                        if r.get('prefix_sha256'):assert oldhash==r['prefix_sha256']
                        env=begin(m)
                        for pair in prev:env.step(pair)
                        obs=public(env,c['seat'])
                        h=digest({k:v for k,v in obs.items() if k!='remainingOverageTime'})
                        if r.get('public_root_sha256'):assert h==r['public_root_sha256']
                        record['old_public_obs_sha256']=h
                        record['same_public_obs']=h==c['public_obs_sha256']
                        if h==c['public_obs_sha256']:count['same_public_root']+=1
                        if oldhash==newhash:
                            count['same_complete_joint_prefix_at_root']+=1
                            if r.get('arms'):
                                a,b=r['arms']; cur=[clean(c['parent']),clean(c['student'])]
                                record['same_pair_forward']=[clean(a),clean(b)]==cur
                                record['same_pair_swapped']=[clean(b),clean(a)]==cur
                                if record['same_pair_forward'] or record['same_pair_swapped']:count['same_measured_command_pair']+=1
                            else:count['same_complete_prefix_pair_unavailable']+=1
                if r['t']==c['t'] or group not in ('single_worker_proxy2614',):details.append(record)
            results[group]=dict(counts=dict(count),overlaps=details)
        output.append(dict(seed=c['seed'],seat=c['seat'],t=c['t'],terrain=c['terrain'],source=c['source'],source_sha256=c['source_sha256'],prefix_sha256=c['prefix_sha256'],public_obs_sha256=c['public_obs_sha256'],source_full_game_previously_used=True,
            any_prior_matching_public_root=any(v["counts"].get("same_public_root",0)>0 for v in results.values()),
            prior_pair_identity_matches=sum(v["counts"].get("same_measured_command_pair",0) for v in results.values()),
            prior_exact_prefix_pairs_unresolved=sum(v["counts"].get("same_complete_prefix_pair_unavailable",0) for v in results.values()),
            prior_fixed_tape_evaluation_sha256=prior_eval,archive_exposure="This exact trajectory already supplied both seats' complete public observation evaluation. The source game and the archived first-divergence root are exposed. Running a policy on this fixed tape does not supply the alternative continuation's original-opponent terminal outcome.",registries=results))
    out=dict(snapshot=SNAPSHOT,protocol_sha256=sha((HERE/"protocol.json").read_bytes()),roots_sha256=sha((HERE/"roots.json").read_bytes()),
             coverage={g:len(ls) for g,ls in by_group.items()},selector_files=selector_files,source_sha256=sources,cases=output,
             scope="Named source registries under this pinned repository snapshot. Counts within groups can overlap, including repeats, and are not summed. Additional old registration files are searched under roots.json and selection*.json for these two seeds. Counterfactual-safety probes are one-worker proxy opportunities, not measured native terminal pairs. The two known previous bc69 native archives are checked as custom histories. Window comparisons use the registered beginning and report first actual intervention separately. Other schemas and unpublished experiments are outside coverage. No downstream native_training or these roots' response scores were inputs.")
    dest=HERE/'duplicates.json'
    dest.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(sha(dest.read_bytes()),out['coverage'])
    for c in output:
        print(c['seed'],{g:r['counts'] for g,r in c['registries'].items() if r['counts'].get('same_terrain_seat_t')})

if __name__=='__main__':main()
