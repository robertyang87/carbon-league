"""Official one-transition checks of all positive-mass physical orders under the recorded foe action."""
import os
for var in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[var]='1'
import collections
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
MENU=HERE.parent
INTAKE=MENU.parent
sys.path.insert(0,str(INTAKE/'menu_coverage'));import coverage as mechanical
sys.path.insert(0,str(MENU));import audit
need=mechanical.need
ROOTS_SHA='936cb513038ad6d79e17517aa84a6aec66a6c01a17a5f5ce39d88240c6a50036'
KEYS=list(mechanical.KEYS)
CLASSES=('recruit','plant_or_capture','freight','ordinary')
EXTRA=('relative_cash_now','relative_warehouse_now','relative_ground_rent','relative_loot_now','relative_carried_cargo','own_cash','own_carried_cargo',
       'own_hires','own_plants','own_seizures','own_loaded_deaths','own_load_lost_to_enemy','own_warehouse_now','own_loot_now','enemy_warehouse_now')
FIELDS=EXTRA+tuple(f'{side}_{name}' for side in ('focal','foe') for name in KEYS)
EPS=1e-7


def metrics(balance,seat):
    a,b=balance['by_seat'][seat],balance['by_seat'][1-seat]
    result=dict(relative_cash_now=a['cash_delta']-b['cash_delta'],
                relative_warehouse_now=a['warehouse_deposit']-b['warehouse_deposit'],
                relative_ground_rent=a['tree_ground_income']-b['tree_ground_income'],
                relative_loot_now=a['opponent_cargo_looted']-b['opponent_cargo_looted'],
                relative_carried_cargo=a['closing_worker_cargo']-b['closing_worker_cargo'],
                own_cash=a['cash_delta'],own_carried_cargo=a['closing_worker_cargo'],own_hires=a['collector_hires']+a['planter_hires'],
                own_plants=a['plants'],own_seizures=a['seizures'],own_loaded_deaths=a['own_root_loaded_deaths'],
                own_load_lost_to_enemy=a['own_root_load_lost_to_enemy_collector'],own_warehouse_now=a['warehouse_deposit'],
                own_loot_now=a['opponent_cargo_looted'],enemy_warehouse_now=b['warehouse_deposit'])
    result.update({f'{relative}_{kind}':side[kind] for relative,side in [('focal',a),('foe',b)] for kind in KEYS})
    return result


def branch(env,state,orders):
    from zerosum_env.envs.carbon import idgen
    ids=dict(idgen._id_dict)
    try:
        alt=copy.copy(env);alt.states=copy.deepcopy(env.states)
        alt.steps=env.steps[:-1]+[alt.states];alt.logs=[];alt.error=env.error.copy()
        alt.step(orders,[])
        sim=mechanical.FastCarbon(env.configuration).set_state(state)
        account=mechanical.account(sim,orders,state)
        mechanical.check_board(sim,alt)
        return dict(account=account,official_post_cash_by_seat=[v[0] for v in alt.states[0].observation.players],
                    official_terminal=bool(alt.done),official_rewards=[v.reward for v in alt.states] if alt.done else None)
    finally:
        idgen._id_dict.clear();idgen._id_dict.update(ids)


def root_check(row,env,sim,both,hashes):
    seat=row['seat']
    sealed=row['catalog'];before=row['comparison_after_freeze']
    need(audit.sha(audit.canonical(sealed))==row['catalog_sha256'],'Pre-action catalog digest')
    old={audit.key(c):p for c,p in zip(before['original_options'],before['original_probabilities'])}
    new={audit.key(x['command']):x for x in sealed['orders']}
    need(len(old)==len(before['original_probabilities']) and len(new)==len(sealed['orders']) and set(old)<=set(new),'Missing/duplicated support')
    need(before['issued']==both[seat] and before['original_options'][before['actual_old_choice']]==both[seat] and
         row['public_observation_sha256']==hashes[seat],'Issued physical identity')
    need(all(q>0 for q in old.values()) and all(x['probability']>0 for x in new.values()) and math.isclose(sum(old.values()),1,abs_tol=1e-12)
         and math.isclose(sum(x['probability'] for x in new.values()),1,abs_tol=1e-12),'Not a distribution')
    state=sim.get_state();actions=[];idx={}
    for key,src in new.items():
        order=src['command'];joint=both.copy();joint[seat]=order
        account=branch(env,state,joint)
        ratios=collections.Counter()
        for p in src['paths']:
            need(p['class'] in CLASSES and math.isfinite(p['mass']) and p['mass']>=0,'Wrong class mass')
            ratios[p['class']]+=p['mass']
        need(abs(sum(ratios.values())-src['probability'])<=1e-12,'Physical order marginal does not match class paths')
        info=dict(order=order,old_probability=old.get(key,0.),new_probability=src['probability'],
                  class_mass=dict(ratios),metrics=metrics(account['account'],seat),**account)
        idx[key]=len(actions);actions.append(info)
    default=idx[audit.key(sealed['baseline'])];base=actions[default]['metrics']
    for opt in actions:
        opt['minus_default']={k:opt['metrics'][k]-base[k] for k in FIELDS}
    masses={c:sum(q['class_mass'].get(c,0.) for q in actions) for c in CLASSES}
    need(all(abs(p-0.25)<1e-12 for p in masses.values()),'Wrong class sampling prior')
    return dict(seed=row['seed'],seat=seat,arm=row['arm'],step=row['step'],terrain=row['terrain'],
                pre_visible_sha256=hashes,pre_catalog_sha256=row['catalog_sha256'],
                fixed_opponent_order=both[1-seat],recorded_own_order=both[seat],default_index=default,
                choices=actions)


def process(ticket,ep):
    need(all(ticket[k]==ep[k] for k in ('seed','seat','arm','terrain','archive_sha256','registration_sha256','tape_sha256')),'Training ticket changed')
    selected={r['step']:r for r in ticket['rows']};need(set(selected)==set(range(12,277,24)),'Incomplete registered roots')
    out=[];sha=hashlib.sha256()
    ref=dict(path=ep['archive_path'],sha256=ep['archive_sha256'],seed=ep['seed'],seat=ep['seat'])
    with mechanical.roots.open_episode(ref,ep['arm']) as (env,live,tape,reg,reg_bytes):
        need(mechanical.sha(reg_bytes)==ep['registration_sha256'],'Wrong official registration')
        sim=mechanical.FastCarbon(env.configuration).reset(seed=ep['seed']);bank=[5.,5.]
        mechanical.check_board(sim,env)
        for t,b in enumerate(tape):
            row=json.loads(b);sha.update(b)
            need(t==row['step']==env.states[0].observation.step and all(s['status']=='ok' for s in row['calls']),'Invalid source transition')
            hashes=[mechanical.sha(live.encoded(live.observe(env,s,bank[s]))) for s in (0,1)]
            need(hashes==row['public_observation_sha256'],'Incorrect physical prefix')
            both=[s['response']['action'] for s in row['calls']]
            if t in selected:
                mechanical.check_board(sim,env);out.append(root_check(selected[t],env,sim,both,hashes))
            sim.step(*both);env.step(both,[]);bank=row['overage_remaining_seconds']
        need(env.done and [s.reward for s in env.states]==ep['official_scores_by_seat'] and sha.hexdigest()==ep['tape_sha256'],'Source final rules differ')
        mechanical.check_board(sim,env)
    return out


def aggregate(rows,kind):
    # Each physical root gets equal weight. Within a root a physical full order may
    # belong to several sampling paths; use summed mass once for total-menu results.
    def mass(o):
        if kind=='new':return o['new_probability']
        if kind=='old':return o['old_probability']
        if kind=='new_only_unconditional':return o['new_probability'] if not o['old_probability'] else 0.
        return o['class_mass'].get(kind,0.)/0.25
    fields={}
    for name in FIELDS:
        moments=[];upp=[];down=[];n_up=n_down=any_change=0
        for r in rows:
            supported=[o for o in r['choices'] if mass(o)>0]
            dv=[(mass(o),o['minus_default'][name]) for o in supported]
            moments.append(math.fsum(w*v for w,v in dv))
            up=math.fsum(w for w,v in dv if v>EPS);dn=math.fsum(w for w,v in dv if v< -EPS)
            any_change+=any(abs(v)>EPS for w,v in dv)
            upp.append(up);down.append(dn);n_up+=any(v>EPS for w,v in dv);n_down+=any(v< -EPS for w,v in dv)
        fields[name]=dict(roots_with_changed_choice=any_change,avg_shift_from_default=math.fsum(moments)/len(rows),mean_mass_increase=math.fsum(upp)/len(rows),
            mean_mass_decrease=math.fsum(down)/len(rows),roots_allow_increase=n_up,roots_allow_decrease=n_down,
            positive_mean_roots=sum(v>EPS for v in moments),negative_mean_roots=sum(v< -EPS for v in moments))
    return fields


def run():
    tree=(MENU/'roots.json.gz').read_bytes()
    need(audit.sha(tree)==ROOTS_SHA,'Unsigned proposed roots')
    doc=json.loads(gzip.decompress(tree));summary=json.loads((MENU/'coverage.json').read_bytes());plan,eps=audit.inventory()
    need(summary['roots_file_sha256']==ROOTS_SHA and doc['protocol_sha256']==audit.PROTOCOL and len(doc['tickets'])==len(eps)==4,'Wrong source evidence')
    source={(e['seed'],e['seat'],e['arm']):e for e in eps}
    out=[]
    for t in doc['tickets']:
        out+=process(t,source[(t['seed'],t['seat'],t['arm'])])
    need(len(out)==48==len({(r['seed'],r['seat'],r['step']) for r in out}),'Invalid denominator')
    nnew=sum(len(r['choices']) for r in out)
    nold=sum(sum(v['old_probability']>0 for v in r['choices']) for r in out)
    need(nnew==315 and nold==92,'Expected signed proposal census')
    allkinds=('new','old','new_only_unconditional',*CLASSES)
    report=dict(schema='sapling-public-menu-official-single-transition/1',catalogs_sha256=ROOTS_SHA,
                program_sha256=mechanical.sha(Path(__file__).read_bytes()),mechanical_ledger_sha256=mechanical.sha(Path(mechanical.__file__).read_bytes()),
                source_games=len(eps),physical_maps=len(set(e['terrain'] for e in eps)),census=dict(roots=48,physical_orders=315,old_positive=92,new_only=223),
                by_menu={k:aggregate(out,k) for k in allkinds},by_seat={str(s):{k:aggregate([r for r in out if r['seat']==s],k) for k in ('new','old')} for s in (0,1)},
                conditional_terminal_labels=0,new_games=0,scope='One simultaneous official transition under the logged opponent order for each already-sealed student proposal. Counterfactual enemy responses, lasting tree income, future freight realization and terminal game advantage are unknown. Signs of cash outlays and cargo carried separately are not welfare labels. Groups have 1/4 mass. New-only figures retain unconditional new-menu probabilities.')
    dump=lambda x:(json.dumps(x,ensure_ascii=False,sort_keys=True,allow_nan=False,separators=(',',':'))+'\n').encode()
    rows_path=HERE/'official_orders.json.gz';payload=gzip.compress(dump(dict(catalogs_sha256=ROOTS_SHA,roots=out)),mtime=0);rows_path.write_bytes(payload)
    report['official_orders_sha256']=mechanical.sha(payload);(HERE/'summary.json').write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')
    return report


if __name__=='__main__':
    print(run()['census'])
