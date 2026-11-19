"""Full native Sapling package calls on previously disclosed observed prefixes; no examiner called."""
from collections import Counter
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shared import ACTOR,BASE,HERE,PROTO,ROOT,TRAIN,module,read,require,save,sha
sys.path.insert(0,str(ROOT/'code'))
from fastsim import FastCarbon

TIMES={19,39,59,79,99,119,139,159,179,199,219,239,259,279,297}


def instrument(pack):
    core=pack._return_decode;choose=pack._mixed_actions;arm_state={}
    def record(x,aux,enc,apply):
        out=core(x,aux,enc,apply)
        arm_state[bool(apply)]=out.copy()
        return out
    def second_pass(x,aux):
        arm_state.clear();final=choose(x,aux);base=arm_state[False];alternate=arm_state[True]
        protected=(np.arange(len(base))>=10)|(x['tok_kind'][:len(base)]==1)
        if np.any(base[protected]!=alternate[protected]):reason='protected_role'
        elif np.any(base!=alternate) and pack._return_conflicts(x,alternate)-pack._return_conflicts(x,base):reason='new_joint_conflict'
        else:reason='use_residual'
        require(np.array_equal(final,alternate if reason=='use_residual' else base),'outer joint fallback')
        prefix=sys.modules['cnet2_feat'].Prefix(x,aux);roles=[]
        for k in np.flatnonzero(x['tok_mask'][:len(final)]):
            a=int(final[k]);_,_,legal=prefix.feats(k);require(legal[a],'illegal package prefix');prefix.advance(k,a)
            roles.append((int(k),'base' if k>=10 else ('planter' if x['tok_kind'][k]==1 else 'collector'),a))
        pack.audit=dict(decision=reason,proposal_differs=bool(np.any(base!=alternate)),final_slots=roles)
        return final
    pack._return_decode=record;pack._mixed_actions=second_pass


def run():
    plan=read(PROTO);old=ROOT/plan['arm_parent']['path'];new=TRAIN/'serving/candidate.py'
    receipt=read(TRAIN/'serving/packages.json')
    require(sha(old)==plan['arm_parent']['sha256']==receipt['base_pack_sha256'] and sha(new)==receipt['packages']['candidate']['deployed_sha256'],'pack identity')
    rawreg=read(ACTOR/'opponent_cargo_calibration_dev/sources.json')['native']
    games=[r for r in rawreg if r['seed'] in (621477120,621477121)]
    require(len(games)==4 and len({(r['seed'],r['seat']) for r in games})==4,'archived probe sources')
    fit=read(TRAIN/'panel.json')['games'];tr=[g for g in fit if g['split']=='train']
    require(not ({g['terrain_sha256'] for g in tr}&{r['terrain'] for r in games}) and not ({g['seed'] for g in tr}&{r['seed'] for r in games}),'fit overlap')
    diagnostics=[];folds=Counter();warm=0
    for g in games:
        p=ROOT/g['source'];require(sha(p)==g['sha256'],'raw tape hash')
        raw=read(p);sim=FastCarbon(raw['game_configuration']).reset(seed=g['seed']);s=g['seat']
        arms=[module(path,'entry_blended_'+a) for a,path in enumerate_names((old,new))]
        for arm in arms:instrument(arm)
        for t,joint in enumerate(raw['actions']):
            ob=sim.obs(s)
            for arm in arms:
                action=arm.agent(ob,sim.config)
                arm.audit['commands']=action
            warm+=2
            if t in TIMES:
                records=[m.audit for m in arms];changed=records[0]['commands']!=records[1]['commands']
                for a,r in zip(('parent','candidate'),records):folds[a+'/'+r['decision']]+=1
                diagnostics.append({'seed':g['seed'],'seat':s,'t':t,'root_source_sha256':g['sha256'],'changed':changed,'parent':records[0],'candidate':records[1]})
            sim.step(*joint)
        require(len(raw['actions'])==299,'early extinction in fixed warm fixture')
    require(len(diagnostics)==4*len(TIMES),'incomplete frozen panel')
    changed=sum(r['changed'] for r in diagnostics)
    out={'protocol_sha256':sha(PROTO),'parent_pack_sha256':sha(old),'candidate_pack_sha256':sha(new),'warm_calls':warm,'checked_roots':len(diagnostics),
         'changed_roots':changed,'passed':changed>=1,'second_decode_reasons':dict(folds),'rows':diagnostics}
    save(HERE/'validation/entry.json.gz',out);print(changed,len(diagnostics))


def enumerate_names(paths):
    return zip(('parent','candidate'),paths)


if __name__=='__main__':run()
