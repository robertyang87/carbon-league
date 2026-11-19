"""Normal entry held-out choices, with factual high-income and failed outcomes retained separately."""
import collections,math,sys
from pathlib import Path
import numpy as np
from collect import HERE,CFG,ACTOR,PARENT,ECO,FULL,arena,policy,read,sha,save,instrument,D,TO,DIRECTIONS
from analyze import FastCarbon,close

def logp(z,mask,j):
    if not mask[j]:return None
    a=z[mask].astype(np.float64);m=a.max()
    return float(z[j]-m-np.log(np.exp(a-m).sum()))
def main():
    fit=read(HERE/'fit.json');panel=read(HERE/'panel.json');labels=read(HERE/'rows.json.gz')
    root_to_rows=collections.defaultdict(list)
    for i,r in enumerate(labels):
        if r['split']!='train':root_to_rows[r['raw'],r['seat'],r['t']].append((i,r))
    files={'parent':PARENT,'bc':HERE/'bc.py','candidate':HERE/'candidate.py'}
    models={key:instrument(path) for key,path in files.items()}
    for key in ('bc','candidate'):
        if sha(files[key])!=fit['outputs'][key]['pack_sha256']:raise ValueError('model artifact')
    outcomes=[];normal=collections.Counter()
    for c in panel['cards']:
        split=panel['split'][c['terrain']]
        if split=='train':continue
        raw=read(arena.ROOT/c['path']);cfg=raw['game_configuration']
        for s in (0,1):
            sim=FastCarbon(cfg).reset(seed=c['seed']);used=set(t for f,seat,t in root_to_rows if f==c['sha256'] and seat==s)
            allcommands=collections.defaultdict(dict)
            for t,joint in enumerate(raw['actions'][:max(used,default=0)+1]):
                obs=sim.obs(s)
                if t==0 or t in used:
                    info={}
                    for name,(mod,captured,choices) in models.items():
                        captured.clear();choices.clear();decoded=mod.agent(obs,cfg)
                        audit=mod._FEAT;before=choices[False];after=choices[True]
                        # Reconstruct the exact selected route from frozen safety conditions using the entry's public encoded features.
                        # The real physical orders bind the emitted labels; when both branches match, production selects the residual route.
                        full=audit.labels(decoded)
                        if np.array_equal(after,before):branch=True
                        elif np.array_equal(full,after):branch=True
                        elif np.array_equal(full,before):branch=False
                        else:raise ValueError('unsupported deployed command')
                        info[name]=(decoded,full,captured[branch],audit.own_uid,branch)
                    if t in used:
                        normal[split+'/joint_roots']+=1
                        for name in ('bc','candidate'):
                            normal[split+'/'+name+'_joint_changes']+=info[name][0]!=info['parent'][0]
                        normal[split+'/candidate_vs_bc_joint_changes']+=info['candidate'][0]!=info['bc'][0]
                        for i,r in root_to_rows[c['sha256'],s,t]:
                            detail={};slot=r['slot'];y=r['target']
                            for name,(moves,lab,values,units,branch) in info.items():
                                if units[slot]!=r['uid']:raise ValueError("UID alignment")
                                hidden,z,mask=values[slot];lp=logp(z,mask,y)
                                cmd=moves.get(r['uid']) or 'STAY'
                                detail[name]=dict(issued=cmd,matches_factual=cmd==r['actual_cmd'],factual_logp=lp,legal=int(mask.sum()),guard_emitted=int(lab[slot]),
                                        selected_return_branch=branch,prefix_changed_vs_parent=not np.array_equal(lab[:slot],info['parent'][1][:slot]))
                            if r['parent']!=int(info['parent'][1][slot]):raise ValueError('parent normal entry changed')
                            outcomes.append(dict(index=i,**{k:r[k] for k in ['raw','seed','terrain','seat','t','uid','slot','task','group','split','is_master','action_class','reward','positive','failed','actual_cmd']},
                                                 observed=detail))
                else:
                    for mod,_,_ in models.values():mod._FEAT.observe(obs)
                sim.step(*joint)
        if len(outcomes)%200<70:print('held',c['seed'],len(outcomes),flush=True)
    if len(outcomes)!=sum(len(v) for v in root_to_rows.values()):raise ValueError('missing held roots')
    save('normal_entry_actions.json.gz',outcomes);scores={}
    for split in ('val','test'):
        scores[split]={}
        for task in ('bank','tree'):
            base=[v for v in outcomes if v['split']==split and v['task']==task];stats={}
            for tag,b in [('all',base),('successful',[v for v in base if v['positive']]),('uncredited_or_no_payback',[v for v in base if v['failed']])]:
                within=[v for v in b if all(v['observed'][k]['factual_logp'] is not None for k in files)]
                stats[tag]=dict(roots=len(b),common_support=len(within))
                for name in files:
                    hit=sum(v['observed'][name]['matches_factual'] for v in b)
                    vals=[v['observed'][name]['factual_logp'] for v in within]
                    stats[tag][name]=dict(demo_hits=hit,mean_factual_logp=float(np.mean(vals)) if vals else None,
                        changed_from_parent=sum(v['observed'][name]['issued']!=v['observed']['parent']['issued'] for v in b))
            # Matched success-minus-failure conditional change in log probability relative to the control/parent.
            # A bin must include both outcomes on held maps and at least two different actually executed action classes.
            for comparator in ('parent','bc'):
                groups=collections.defaultdict(list)
                for v in base:
                    o=v['observed']
                    if (v['positive'] or v['failed']) and o['candidate']['factual_logp'] is not None and o[comparator]['factual_logp'] is not None:
                        groups[(v['terrain'],v['is_master'],tuple(v['group'][:2]))].append(v)
                bymap=collections.defaultdict(list);n=0;nr=0
                for k,vals in groups.items():
                    if len({v['action_class'] for v in vals})<2:continue
                    delta=lambda v:v['observed']['candidate']['factual_logp']-v['observed'][comparator]['factual_logp']
                    yes=[delta(v) for v in vals if v['positive']];no=[delta(v) for v in vals if v['failed']]
                    if yes and no:bymap[k[0]].append(float(np.mean(yes)-np.mean(no)));n+=1;nr+=len(vals)
                diffs=[float(np.mean(v)) for v in bymap.values()]
                stats['matched_candidate_vs_'+comparator]=dict(positivity_matched_bins=n,covered_roots=nr,covered_maps=len(diffs),
                       map_mean=float(np.mean(diffs)) if diffs else None,positive_maps=sum(v>0 for v in diffs),per_map=diffs)
            scores[split][task]=stats
    save('heldout.json',dict(fit=sha(HERE/'fit.json'),trace=sha(HERE/'normal_entry_actions.json.gz'),full_joint_counts=normal,results=scores))
if __name__=='__main__':main()
