"""Clear four independent replication development maps against repository, platform and other development panels."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import types

HERE=Path(__file__).resolve().parent
FAMILY=HERE.parent
SCANNER=FAMILY/'ladder/cargo_defense/development/clearance.py'
SOURCE='91d1e13719060a50efd36cfc443510f0456c2b0c'
SEEDS=list(range(621782460,621782464))
TOKEN=re.compile(rb'(?<![.\da-zA-Z])62178246[0-3](?![.\da-zA-Z])')

def main():
    spec=importlib.util.spec_from_file_location('official_and_published_maps',SCANNER)
    shared=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    tip=shared.prior.checked_remote()
    old=types.ModuleType('numeric_prefilter')
    old.__file__=shared.old.__file__
    source=Path(old.__file__).read_text()
    test="if b'62175859' in b and CANDIDATE.search(b)"
    assert source.count(test)==1
    exec(compile(source.replace(test,"if b'62178246' in b and CANDIDATE.search(b)"),old.__file__,'exec'),old.__dict__)
    old.SEEDS,old.TOKEN,old.BASE=SEEDS,TOKEN,SOURCE
    shared.SEEDS,shared.TOKEN,shared.SOURCE=SEEDS,TOKEN,SOURCE
    # Many public games have already been saved below later audit directories.
    # Bind offline bytes to this frozen main. Fetch the live replay only for a new game.
    fetch=shared.get
    main_files=shared.prior.git('ls-tree','-r','--name-only',tip).decode().splitlines()
    archived={Path(p).name:p for p in main_files if '/replays/' in p and p.endswith('-1.json.gz')}
    def cached_get(url):
        m=re.fullmatch(re.escape(shared.SERVER)+r'/replay/([0-9a-f]{32})/([1-9][0-9]*)',url)
        if m:
            path=archived.get(f'{m[1]}-{m[2]}.json.gz')
            if path:
                return shared.prior.frozen(shared.prior.ROOT/path,tip)
        return fetch(url)
    shared.get=cached_get
    census=shared.complete(old.generate(),tip)
    boards=census['by_seed']
    panels={}
    for folder in ('paired_improvement_dev','calibration_improvement_dev','identity_replication_dev','v3_release/development','nn_policy_development','v4_replication_dev','native_critic_dev','two_response_dev'):
        record=json.loads(shared.prior.frozen(FAMILY/folder/'protocol.json',tip))
        historical=record['physical_boards']
        assert not set(historical)&set(boards) and not set(historical.values())&set(boards.values())
        panels[folder]=historical
    def api(path):
        return json.loads(subprocess.check_output(['gh','api','repos/robertyang87/carbon-league/'+path]))
    active={}
    for status in ('in_progress','queued','waiting','requested','pending'):
        runs=api('actions/runs?per_page=100&status='+status)
        assert runs['total_count']==len(runs['workflow_runs'])
        for r in runs['workflow_runs']:
            if r['id'] in active:continue
            jobs=api(f"actions/runs/{r['id']}/jobs?per_page=100")
            assert jobs['total_count']==len(jobs['jobs'])
            names=[j['name'] for j in jobs['jobs']]
            assert not any(TOKEN.search(n.encode()) for n in names)
            active[r['id']]={'head':r['head_sha'],'jobs':names}
    # Main and the public platform cannot identify unpushed local reservations.
    # Check local, unstaged/untracked research documents for exact candidate IDs.
    changed=shared.prior.git('diff','--name-only','HEAD').decode().splitlines()
    untracked=shared.prior.git('ls-files','--others','--exclude-standard').decode().splitlines()
    local_checked=[]
    for name in sorted(set(changed+untracked)):
        if name.startswith(HERE.relative_to(shared.prior.ROOT).as_posix()+'/'):
            continue
        path=shared.prior.ROOT/name
        if name=='.github/workflows/master3s2-native-speed-dev.yml':
            actual=[s.strip() for s in path.read_bytes().splitlines() if TOKEN.search(s)]
            expected=b'seed: ['+b', '.join(str(s).encode() for s in SEEDS)+b']'
            assert actual==[expected], (name,actual)
            continue
        if path.is_file() and path.suffix in ('.md','.json','.py','.yaml','.yml','.txt','.jsonl'):
            raw=path.read_bytes()
            assert not TOKEN.search(raw) and not any(h.encode() in raw for h in boards.values()),name
            local_checked.append(name)
    census['unpublished_local_files_checked']=local_checked
    platform_board=json.loads(fetch(shared.SERVER+'/api/leaderboard'))
    names={'ds-pro-v3','ds-pro-v4'}
    entrants={r['id']:r['name'] for field in ('strategies','extra') for r in platform_board[field]
              if r.get('author')=='ds-pro' and r.get('name') in names}
    registered=json.loads(shared.prior.frozen(FAMILY/'ladder/ds-pro-v4.json',tip))['platform']
    assert registered['name']=='ds-pro-v4'
    entrants[registered['id']]=registered['name']
    for uid,version in entrants.items():
        rows=shared.live_cards(uid);completed=[];pending={}
        for ident,r in rows.items():
            for g in r.get('result') or []:
                key=f"{ident}-{g['game']}"
                if key not in census['ladder_openings']:
                    raw=cached_get(shared.SERVER+f"/replay/{ident}/{g['game']}")
                    env,_,_=shared.public.extract(raw);shared.public.validate(env,{**r,**g})
                    initial=env['steps'][0][0]['observation']
                    layout=shared.prior.old.engine.map_identity.terrain_sha256(initial['full_carbon'])
                    assert {g['seed'],env['configuration']['randomSeed']}.isdisjoint(SEEDS)
                    assert {g['map_sha256'],layout}.isdisjoint(boards.values())
                    census['ladder_openings'][key]=dict(match=ident,game=g['game'],platform_map_seed=g['seed'],platform_map_sha256=g['map_sha256'],
                        replay_random_seed=env['configuration']['randomSeed'],opening_u8_sha256=layout,source_sha256=shared.prior.sha(raw),source='public-or-frozen')
                completed.append(key)
            if r['state']!='done' or not r.get('result'):pending[ident]=r['state']
        census['ladder_scorecards'][version]=dict(strategy_id=uid,completed=sorted(completed),pending_without_complete_result=pending)
    census['public_v3_discovery']={'endpoint':shared.SERVER+'/api/leaderboard','entries_checked':{k:len(v) for k,v in platform_board.items()},'registered_v4_id':registered['id'],'matched_names':entrants}
    census['ladder_as_of_utc']=shared.datetime.now(shared.timezone.utc).isoformat()
    assert set(map(int,boards))==set(SEEDS) and len(set(boards.values()))==4
    census.update(schema='native-speed-development-clearance/1',purpose='Four development boards only, separate from formal and training.',
                  groups={'native_speed_development_only':SEEDS},older_comparison_boards=panels,active_hosted_jobs=active,
                  verifier_sha256=shared.prior.sha(Path(__file__).read_bytes()),legacy_scanner_sha256=shared.prior.sha(SCANNER.read_bytes()))
    print(old.shared.save(HERE/'clearance.json',census),boards,tip,flush=True)


if __name__=='__main__':main()
