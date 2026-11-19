"""Disjoint and overlapping explanations of the public opportunity denominator, by physical source game."""
from collections import Counter, defaultdict
from statistics import mean

import policy
from action_audit import HERE, sources, native_game, read, save


def reasons(st, u):
    own=st['own'];p=own[2][u][0]
    if p in st['trees'] or p in st['bases']:
        return ['tree_or_base']
    checks={'cash_reserve':st['available']+st['pay'].get(u,0)<st['price']+st['min_hire'],
            'friendly_hire_or_landing': p in st['hires'] or any(q==p and v!=u for v,q in st['dest'].items()),
            'opposing_worker_within_one':policy.near(p,st['ep']+st['ec'])<=1,
            'opposing_base_within_one':policy.near(p,st['nbases'])<=1,
            'neighbor_carbon_unavailable':any(st['carbon'][q] is None or st['carbon'][q]<0 for q in policy.DEST[p][1:])}
    return [k for k,v in checks.items() if v]


def scan(source,g):
    rows={k:defaultdict(Counter) for k in ('v1','opponent')}
    for views,joint,_,cfg in native_game(source):
        for s,o in enumerate(views):
            role='v1' if s==source['seat'] else 'opponent'
            phase=policy.stage(o['step']); r=rows[role][phase]
            st=policy.context(o,cfg,joint[s])
            for u,(p,_,kind) in st['own'][2].items():
                if kind!='PLANTER':continue
                r['worker_turns']+=1
                why=reasons(st,u)
                for k in why: r['present_'+k]+=1
                r['first_'+(why[0] if why else 'eligible')]+=1
                if not why:
                    x=policy.proposal(st,u)
                    if x is None:raise ValueError('inconsistent opportunity filter')
                    held=st['dest'][u]==p
                    r['paid']+=int(held)
                    if not held:
                        lo,hi=policy.bounds(st,u)
                        r['move_lower_gt_quote']+=int(lo>st['price']+.02)
                        r['move_lower_gt_zero']+=int(lo>0)
                    if held:
                        _,hi=policy.bounds(st,u)
                        r['hold_upper_lt_quote']+=int(hi is not None and hi+.02<st['price'])
    return dict(set=g,seed=source['seed'],seat=source['seat'],groups=rows)


def main():
    games=[scan(s,g) for g,s in sources()]
    groups={}
    for group in ('training_source','held_development'):
        sums={role:defaultdict(Counter) for role in ('v1','opponent')}
        for game in games:
            if game['set']!=group:continue
            for role,phases in game['groups'].items():
                for phase,counts in phases.items():sums[role][phase].update(counts)
        by_phase={}
        for phase in ('opening','middle','late'):
            a,b=sums['v1'][phase],sums['opponent'][phase]
            m,n=a['first_eligible'],b['first_eligible']
            p,q=a['paid']/m,b['paid']/n
            by_phase[phase]=dict(sapling_clean=m,opponent_clean=n,sapling_rate=p,opponent_rate=q,
                                 total_paid_difference=a['paid']-b['paid'],
                                 opportunity_count_contribution=(m-n)*(p+q)/2,
                                 conditional_rate_contribution=(p-q)*(m+n)/2)
        groups[group]=dict(counts=sums,arithmetic_decomposition=by_phase)
    save(HERE/'spatial.json',dict(groups=groups,games=games,
        meanings='first_ bins partition every planter turn. present_ exclusions co-occur and cannot be added. The two-factor split is a descriptive arithmetic identity. A contested cell can reflect an intentional attack.'))
    print(groups['training_source']['arithmetic_decomposition']['middle'],groups['training_source']['counts']['v1']['middle'],flush=True)


if __name__=='__main__':
    main()
