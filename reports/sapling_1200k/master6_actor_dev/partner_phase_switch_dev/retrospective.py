"""Audit each completed historical partner fork's actual commands, cash writes and official endpoint."""
import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ACTOR=HERE.parent
SAP=ACTOR.parent
ROOT=SAP.parents[1]
RESP=HERE/'responses'
OUT=HERE/'retrospective'
sys.path[:0]=[str(SAP),str(SAP/'root_recovery_branches')]
from branches import begin, public
from turning_point_ledger import FastCarbon,CashEvents

ARMS=('it20','it50')


def need(ok,msg):
    if not ok: raise ValueError(msg)


def read(p):
    with (gzip.open(p,'rb') if p.suffix=='.gz' else p.open('rb')) as f:return json.load(f)


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def digest(d):
    return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write(p,v):
    data=json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
    if p.suffix=='.gz':data=gzip.compress(data,mtime=0)
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.is_file() and p.read_bytes()==data:return
    q=p.with_name(p.name+'.new');q.write_bytes(data);q.replace(p)


def sign(x):
    return int(x>0)-int(x<0)


def net_cash(cash,s):
    return cash[s]-cash[1-s]


def sides(events):
    counters=[Counter(),Counter()]
    for tag,s,amount,*_ in events:counters[s][tag]+=amount
    return [dict(x) for x in counters]


def _trace(raw,card,a,name):
    s=card['seat'];t=card['t'];cfg=raw['game_configuration']
    binding={'id':card['id'],'card_sha256':digest(card),'arm':name,'frozen_sha256':sha(RESP/'frozen.json'),
             'paired_seed':int(digest(['sapling-partner-continuation-v1',card['raw_sha256'],t,card['id'],card['stream']])[:8],16)}
    need(a['binding']==binding and a['initial_public']==[card['own_public'],card['enemy_public']],'signed root binding')
    need(a['joint0'][s]==card[name] and a['joint0'][1-s]==card['root_enemy']==raw['actions'][t][1-s],'submitted simultaneous root')
    hist=a['history'];N=a['terminal_step']; oc=a['original_config']
    need(oc['_SEED'] is None and oc['_TEMP']==1.0 and oc['_SEARCH_OK'] and oc['_RX'] and oc['_S_HOMEFIX'],'original configuration')
    need([v['t'] for v in hist]==list(range(t+1,N)) and a['completed_original_calls']==len(hist) and
         a['statuses']==['DONE','DONE'],'incomplete or truncated branch')
    full=list(raw['actions'][:t])+[a['joint0']]+[x['joint'] for x in hist]
    env=begin(raw);sim=FastCarbon(cfg).reset(seed=card['seed']);sim.cash=CashEvents(sim.cash)
    frames=[];cash_from_root=[Counter(),Counter()];cash_at_root=None
    for i,joint in enumerate(full):
        need(sim.step_count==i and not env.done,'command after terminal')
        if i>=t:
            us=public(env,s);them=public(env,1-s)
            if i==t:
                need([digest(us),digest(them)]==a['initial_public'],'initial history mismatch')
                cash_at_root=list(sim.cash)
            else:
                h=hist[i-t-1];meter=h['original_search_meter']
                need(h['issue']['binding']==binding and h['issue']['t']==i and digest(us)==h['own_public_sha256']
                     and digest(them)==h['op_public_sha256'] and meter['wall_time_ns']>0
                     and math.isclose(h['wall_seconds'],meter['wall_time_ns']/1e9,rel_tol=0,abs_tol=1e-14),'live meter or public view mismatch')
        keep=i>=max(0,t-10)
        if keep:
            pre=sim.full_observation();start=list(sim.cash)
        sim.cash.events.clear()
        env.step(joint);sim.step(*joint)
        if not keep:continue
        post=sim.full_observation(); official=env.states[0].observation
        need(post['players']==official.players and post['trees']==official.trees and post['carbon']==official.full_carbon,'official board mismatch')
        events=[list(v) for v in sim.cash.events];flow=sides(events)
        need(all(math.isclose(start[k]+sum(flow[k].values()),sim.cash[k],abs_tol=1e-6) for k in (0,1)),'step cash ledger')
        if i>=t:
            for k in (0,1):cash_from_root[k].update(flow[k])
        frames.append({'action_t':i,'source':'common_observed_prefix' if i<t else 'native_root_or_response',
                       'pre_players':pre['players'],'pre_tree_history':pre['trees'],'joint_physical_seats':joint,
                       'cash_after_physical':[pl[0] for pl in post['players']],
                       'physical_cash_writes':events,'flow_by_physical_seat':flow,
                       'collision_losses_post_move':sim.last_collisions.copy()})
    scores=[a0.reward for a0 in env.states];postcash=[pl[0] for pl in env.states[0].observation.players]
    need(env.done and scores==a['official_scores'] and postcash==a['cash'] and a['official_margin']==net_cash(scores,s)
         and sim.step_count==N and all(math.isclose(cash_at_root[k]+sum(cash_from_root[k].values()),sim.cash[k],abs_tol=1e-6)
                                     for k in (0,1)),'complete official cash/score mismatch')
    floor=min(cfg['recCollectorCost'],cfg['recPlanterCost'])
    terminal=env.states[0].observation.players
    extinct=[not(work or trees) and (not base or money<floor) for money,base,work,trees in terminal]
    if not any(extinct):need(scores==postcash and N==cfg['episodeSteps']-1,'normal terminal score')
    else:
        pre=frames[-1]['pre_players'];expected=[N-cfg['episodeSteps']-1 if extinct[k] else pre[k][0] for k in (0,1)]
        need(scores==expected,'extinction rule')
    return {'frames':frames,'terminal_step':N,'official_scores':scores,'official_margin':net_cash(scores,s),'terminal_cash':postcash,
            'terminal_cash_margin':net_cash(postcash,s),'official_minus_terminal_cash_margin':net_cash(scores,s)-net_cash(postcash,s),
            'extinct':extinct,'cash_from_root_by_physical_seat':cash_from_root,'native_followup_calls':len(hist),
            'paired_seed':binding['paired_seed'],'root_observation_t':t,'source_raw_sha256':card['raw_sha256']}


def turning(series,root,first,flows):
    """series indexed by pre-action observation time. Compare candidate windows ending in the observed fork."""
    end=max(series)
    def mark(t,delta,start):
        channels=Counter()
        for j in range(start,t):
            channels.update(flows[j])
        need(abs(sum(channels.values())-delta)<.041,'signed cash-channel allocation at turning point')
        return {'observed_t':t,'turning_action_t':t-1,'change_since_observation_t':start,'change':delta,
                'signed_cash_channel_contributions':dict(channels),
                'lookback_ranges_of_actual_actions':{str(n):[max(first,t-n),t-1] for n in (3,5,10)},
                'lookback_before_turning_action_t':{str(n):max(first,t-1-n) for n in (3,5,10)}}
    peaks={}
    for w in (1,3,5,10):
        # Only actual branch earnings within these windows. Context references can extend into the common prefix.
        intervals=[(t,series[t]-series[t-w]) for t in range(root+w,end+1)]
        if not intervals:continue
        up=max(intervals,key=lambda x:x[1]);down=min(intervals,key=lambda x:x[1])
        peaks[str(w)]={'largest_increase':mark(up[0],up[1],up[0]-w) if up[1]>0 else None,
                       'largest_decrease':mark(down[0],down[1],down[0]-w) if down[1]<0 else None}
    flips=[mark(t,series[t]-series[t-1],t-1) for t in range(root+1,end+1)
           if sign(round(series[t],7))!=sign(round(series[t-1],7))]
    persistent=next((t for t in range(root,end+1) if sign(series[t])==sign(series[end])
                     and all(sign(series[i])==sign(series[end]) for i in range(t,end+1))),None)
    return {'largest_changes_by_window':peaks,'sign_changes_including_ties':flips,'earliest_persistent_terminal_sign_observation_t':persistent}


def first_difference(byarm,side):
    for t in sorted(set(byarm['it20']) & set(byarm['it50'])):
        a,b=[byarm[k][t] for k in ARMS];x,y=a['joint_physical_seats'][side],b['joint_physical_seats'][side]
        keys=sorted(k for k in x.keys()|y.keys() if (x.get(k) or 'NONE')!=(y.get(k) or 'NONE'))
        if keys:
            return {'action_t':t,'workers_or_bases':{
                k:{arm:{'order':f['joint_physical_seats'][side].get(k,'NONE'),
                        'worker_pre_state':f['pre_players'][side][2].get(k),'base_pre_position':f['pre_players'][side][1].get(k)}
                   for arm,f in zip(ARMS,(a,b))} for k in keys}}
    return None


def audit_pair(card):
    folder=RESP/'cases'/card['stream']/card['id']
    paths={k:folder/(k+'.json.gz') for k in ARMS}
    rawfile=ROOT/card['raw'];need(sha(rawfile)==card['raw_sha256'],'source tape changed')
    raw=read(rawfile);need(raw['seed']==card['seed'] and raw['candidate_seat']==card['seat'] and
                           raw['initial_terrain_u8_sha256']==card['map'] and raw['pack_sha256']==
                           read(HERE/'protocol.json')['partners']['it50'],'wrong map/seat or historical parent')
    traces={k:_trace(raw,card,read(p),k) for k,p in paths.items()}
    need(traces['it20']['paired_seed']==traces['it50']['paired_seed'],'unpaired response draw')
    s,t=card['seat'],card['t'];byarm={};gaps={};flow_diffs={}
    for k,r in traces.items():
        f={a['action_t']:a for a in r['frames']};byarm[k]=f
        first=min(f);prefix=max(0,t-10);need(first==prefix,'missing requested ten-step context')
        v={first:net_cash([z[0] for z in f[first]['pre_players']],s)}
        v.update({i+1:net_cash(a['cash_after_physical'],s) for i,a in f.items()})
        gaps[k]=v
        r['own_minus_enemy_cash_by_observation']=[[i,v[i]] for i in sorted(v)]
        flows={}
        for i,a in f.items():
            net=Counter(a['flow_by_physical_seat'][s]);net.subtract(a['flow_by_physical_seat'][1-s]);flows[i]=net
        flow_diffs[k]=flows
        r['cash_margin_turns']=turning(v,t,first,flows)
    both=sorted(set(gaps['it20']) & set(gaps['it50']))
    diffs={i:gaps['it20'][i]-gaps['it50'][i] for i in both}
    need(all(abs(diffs[i])<1e-8 for i in both if i<=t),'incompatible pre-root histories')
    first={side:first_difference(byarm,s if side=='own' else 1-s) for side in ('own','opponent')}
    need(first['opponent'] is None or first['opponent']['action_t']>t,'opponent could see root action')
    compared={}
    for i in both[:-1]:
        net=flow_diffs['it20'][i].copy();net.subtract(flow_diffs['it50'][i]);compared[i]=net
    delta=traces['it20']['official_margin']-traces['it50']['official_margin']
    return {'id':card['id'],'seed':card['seed'],'map':card['map'],'seat':s,'t':t,'stream':card['stream'],'split':card['split'],
            'arm_sha256':{k:sha(p) for k,p in paths.items()},'card_sha256':sha(folder/'card.json'),
            'first_command_difference':first,'arms':traces,'official_margin_it20_minus_it50':delta,
            'aligned_last_observation':max(both),'cash_gap_it20_minus_it50_by_observation':[[i,diffs[i]] for i in both],
            'difference_turns':turning(diffs,t,min(both),compared),
            'credit':'Each transfer or treasury write is observed. Earlier moves leading up to a swing provide context, not an isolated causal estimate. The two policies and opponent responses can diverge over the entire suffix.'}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('split',choices=['train','held'])
    ap.add_argument('--id',help='Audit this one completed preregistered pair before aggregate labels (also requires --stream)')
    ap.add_argument('--stream',choices=['primary','repeat'])
    args=ap.parse_args();need(bool(args.id)==bool(args.stream),'--id and --stream must be supplied together')
    roots=read(HERE/'roots.json');prot=read(HERE/'protocol.json')
    need(roots['protocol_sha256']==sha(HERE/'protocol.json'),'root register')
    if args.split=='held':
        need(read(HERE/'train_gate.json')['decision']=='PASS','unopened held stage')
    requested=[(r,st) for r in roots['roots'] if r['split']==args.split
               for st in (('primary','repeat') if r.get('repeat') and r['split']=='train' else ('primary',))]
    scope={(r['id'],st) for r,st in requested}
    pf=RESP/(args.split+'_labels.json')
    labels={}
    if pf.is_file():
        doc=read(pf)
        need(doc['roots_sha256']==sha(HERE/'roots.json') and doc['frozen_sha256']==sha(RESP/'frozen.json')
             and doc['protocol_sha256']==sha(HERE/'protocol.json'),'aggregate label identities')
        labels={(x['id'],x['stream']):x for x in doc['pairs']};need(set(labels)==scope,'label denominator differs')
    elif not args.id:
        raise ValueError(f'{args.split} signed aggregate not yet available; --id/--stream accepts a completed pair')
    if args.id:
        need((args.id,args.stream) in scope,'unregistered map/stream')
        requested=[(r,st) for r,st in requested if (r['id'],st)==(args.id,args.stream)]
    summaries=[]
    for r,st in requested:
        need(any(x['raw_sha256']==r['raw_sha256'] and x['raw']==r['raw'] and x['map']==r['map'] and x['split']==args.split and
                 x['seed']==r['seed'] and x['seat']==r['seat'] for x in prot['eligible']),'outside eligible frozen sources')
        case=RESP/'cases'/st/r['id'];c=read(case/'card.json')
        need(all(c.get(k)==v for k,v in r.items()) and c['stream']==st,'case/header mismatch')
        out=OUT/args.split/st/(r['id']+'.json.gz')
        identities={arm:sha(case/(arm+'.json.gz')) for arm in ARMS}
        if out.is_file():
            entry=read(out)
            if (entry.get('extractor_sha256')!=sha(Path(__file__)) or entry.get('arm_sha256')!=identities
                    or entry.get('card_sha256')!=sha(case/'card.json')):
                entry=None
        else:entry=None
        if entry is None:
            entry=audit_pair(c);entry['extractor_sha256']=sha(Path(__file__));write(out,entry)
        if labels:
            l=labels[(r['id'],st)];need(entry['official_margin_it20_minus_it50']==l['delta'] and all(
                identities[k]==l[k]['sha256'] for k in ARMS),'signed aggregate disagrees')
        summaries.append({'id':r['id'],'stream':st,'regime':r['regime'],'map':r['map'],'seat':r['seat'],'t':r['t'],
                          'delta':entry['official_margin_it20_minus_it50'],'evidence':str(out.relative_to(HERE)),
                          'sha256':sha(out),'first_differing_order_t':{who:x['action_t'] if x else None for who,x in entry['first_command_difference'].items()}})
    if not args.id:
        groups=defaultdict(dict)
        for a in summaries:groups[a['id']][a['stream']]=a['delta']
        reps=[{'id':i,'primary':v['primary'],'repeat':v['repeat'],'strict_direction_reversal':v['primary']*v['repeat']<0,
               'either_is_tie':v['primary']==0 or v['repeat']==0} for i,v in groups.items() if 'repeat' in v]
        write(OUT/(args.split+'.json'),{'signed_aggregate_sha256':sha(pf),'pairs':summaries,'independent_stream_comparisons':reps})
    print(json.dumps(summaries,ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
