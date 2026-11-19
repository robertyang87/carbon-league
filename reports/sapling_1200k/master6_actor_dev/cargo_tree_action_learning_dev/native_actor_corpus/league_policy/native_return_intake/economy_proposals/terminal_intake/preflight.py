"""Check the registered category gates on the sixteen historical turns, without model oracles."""
import gzip
import json
from pathlib import Path
import zipfile
from select_root import pick, play, runner

HERE = Path(__file__).resolve().parent
ECON = HERE.parent
SHARED = ECON.parent/'live_curriculum/shared_enemy_root'
hfile = runner.fixture.hfile
require = runner.require
PLAN_SHA = '02374a25676aeb83ff9eb1a9fbe7184b3e63eafd49c720dd1baef637dc3c1c77'


def bindings():
    plan = play.read(HERE/'protocol.json')
    require(hfile(HERE/'protocol.json') == PLAN_SHA, 'Registration changed')
    expected = plan['static_inputs']
    signed = [(ECON/'protocol.json','proposal_protocol'),(ECON/'proposals.py','proposal_code'),
              (ECON/'roots.json.gz','reported_48_roots'),(ECON/'coverage.json','coverage'),
              (SHARED/'receipt.json','shared_enemy_mechanism'),(SHARED/'audit.json','shared_enemy_independent_audit')]
    for f,key in signed:
        require(hfile(f) == expected[key], f'Different {key}')
    v = ECON/'verification'
    summary = play.read(v/'summary.json')
    require(summary['official_orders_sha256']==hfile(v/'official_orders.json.gz') and
            summary['catalogs_sha256']==expected['reported_48_roots'] and
            summary['program_sha256']==hfile(v/'verify.py') and summary['census']['roots']==48,
            'Wrong official one-step corpus')
    ext = play.read(SHARED/'external_audit.json')
    require(ext['real_root_calls']==1 and len(ext['rejected'])==4 and ext['strict_parent_after_state_and_rng']
            and ext['implementation_sha256']==hfile(SHARED/'external_menu.py')
            and ext['auditor_sha256']==hfile(SHARED/'check_external.py')
            and ext['source_sha256']==expected['shared_enemy_mechanism'], 'Failed external shared-root verification')
    files = [f for f,_ in signed]+[v/'official_orders.json.gz',v/'summary.json',v/'verify.py',
            SHARED/'external_menu.py',SHARED/'external_audit.json',SHARED/'check_external.py',
            HERE/'protocol.json',HERE/'select_root.py',Path(__file__).resolve()]
    return {str(f.relative_to(play.ROOT)):hfile(f) for f in files}


def loadgz(f):
    with gzip.open(f) as fp:
        return json.load(fp)


def compare(menu, official, seat):
    choices = official['choices']
    kind = {'plant':'plant','capture':'seize','deposit':'own_delivery'}[menu['kind']]
    uid = menu['focal_uid']
    events=[]
    for cmd in menu['options']:
        [choice] = [row for row in choices if row['order']==cmd]
        events.append([v for v in choice['account']['cash_events']
                       if v[0] in ('plant','seize','own_delivery') and v[1]==seat and v[3]==uid])
    on = menu['effect_arm']
    relevant = ('own_delivery',) if kind=='own_delivery' else ('plant','seize')
    hit = any(v[0]==kind and (v[2]>0 if kind=='own_delivery' else v[2]<0) for v in events[on])
    other = any(v[0] in relevant for v in events[1-on])
    return dict(focal_uid=uid,kind=kind,effect_arm=on,events=events,options=menu['options'],
                verified_positive_focal_contrast=hit and not other)


def build():
    sources=bindings()
    plan=play.read(HERE/'protocol.json')
    source=play.read(ECON/'protocol.json')['evidence']['archives']
    cats={f['category']:0 for f in plan['fixtures']}
    catalog={(r['seed'],r['seat'],r['step']):r for t in loadgz(ECON/'roots.json.gz')['tickets'] for r in t['rows']}
    physical={(r['seed'],r['seat'],r['step']):r for r in loadgz(ECON/'verification/official_orders.json.gz')['roots']}
    require(catalog.keys()==physical.keys() and len(catalog)==48,'Different historical roots')
    out=[]
    for ep in source:
        seat,seed=ep['seat'],ep['seed']
        package=play.ROOT/ep['archive_path']
        require(hfile(package)==ep['archive_sha256'],'Source archive changed')
        env,cfg,live=play.official(seed,ep['terrain'],seat)
        bank=[5.,5.]
        with zipfile.ZipFile(package) as archive:
            [member]=[n for n in archive.namelist() if n.endswith('draw_a/turns.jsonl')]
            with archive.open(member) as stream:
                for line in stream:
                    rec=json.loads(line)
                    t=rec['step']
                    require(t==env.states[0].observation.step,'Historical episode misaligned')
                    orders=[v['response']['action'] for v in rec['calls']]
                    if t in plan['probe_turns']:
                        c=catalog[seed,seat,t]
                        p=physical[seed,seat,t]
                        obs=live.observe(env,seat,bank[seat])
                        h=play.sha(live.encoded(obs))
                        require(h==c['public_observation_sha256']==p['pre_visible_sha256'][seat]==rec['public_observation_sha256'][seat]
                                and p['recorded_own_order']==orders[seat] and p['fixed_opponent_order']==orders[1-seat]
                                and c['catalog_sha256']==p['pre_catalog_sha256'], 'One-step or catalog identity changed')
                        for cat in cats:
                            sel=pick(c['catalog'],obs,cfg,orders[seat],cat)
                            details=compare(sel,p,seat) if sel is not None else {'verified_positive_focal_contrast':False}
                            cats[cat]+=int(details['verified_positive_focal_contrast'])
                            out.append(dict(seed=seed,seat=seat,step=t,category=cat,supported=sel is not None,**details))
                    if t==max(plan['probe_turns']):
                        break
                    env.step(orders)
                    bank=rec['overage_remaining_seconds']
    require(len(out)==32,'Wrong historical probe denominator')
    return dict(schema='sapling-semantic-intake-gates/1',sources=sources,archives=source,
                observed_historical_roots=48,scheduled_history_turns=16,verified_focal_contrasts_by_category=cats,
                permitted={k:v>=1 for k,v in cats.items()},rows=out)


if __name__=='__main__':
    play.save(HERE/'preflight.json',build())
