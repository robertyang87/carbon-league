"""List newly completed score cards in the pinned Git interval and obtain identity without training."""
from collections import Counter
import gzip,hashlib,json,re,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
TOP=HERE.parent/'all_top5'
sys.path.insert(0,str(TOP))
import fetch
from recover_public import extract,validate
ANCHOR='5484590b3fae60b3c906c6a0be9f7840c241388a'
LAST='99919cbf5889caa98303091e146aa210fa035d18'
BATCHES=['','appendix_499886f1','appendix_4bfdabf5','appendix_555c0c56','appendix_bf049043','appendix_bd3c0021','appendix_5484590b']
def git(*args):return fetch.git(*args)
def parse(x):return json.loads(gzip.decompress(x) if x.startswith(b'\x1f\x8b') else x)
def data():
    docs=[parse(git('show',LAST+':'+str((TOP/b/'manifest.json').relative_to(fetch.ROOT)))) for b in BATCHES]
    known={g['id']:g for d in docs for g in d['games']}
    if len(known)!=1378:raise ValueError('wrong frozen scope')
    seen_seed={g['seed'] for g in known.values()}
    seen_terrain={}
    for b in BATCHES:
        content=gzip.decompress(git('show',LAST+':'+str((TOP/b/'validation_maps.jsonl.gz').relative_to(fetch.ROOT))))
        seen_terrain.update({r['initial_terrain_u8_sha256']:r['id'] for r in map(json.loads,content.splitlines())})
    fetch.REV=LAST
    tree=fetch.tree();changes=git('diff','--name-status','--no-renames',ANCHOR,LAST).decode().splitlines()
    changed=[l.split('\t')[1] for l in changes if l.split('\t')[0]!='D' and fetch.sourcefile(l.split('\t')[1])]
    refs={};seen_changed=Counter();unchanged_in_old=Counter();states=Counter();first={}
    for p in changed:
        rows={}
        for card in fetch.walk_matches(parse(git('cat-file','blob',tree[p]))):
            if not ({card['name_a'],card['name_b']}&set(fetch.NAMES)):continue
            if card.get('state','done')!='done':states[card.get('state')]+=1;continue
            key=fetch.key_of(card);seen_changed[p]+=1
            if key in rows and fetch.identity(card)!=fetch.identity(rows[key]):raise ValueError('conflict in source')
            rows[key]=card
        for k,card in rows.items():
            if k in known:
                if fetch.identity(card)!=fetch.identity(known[k]['card']):raise ValueError('archived matchup restated')
                unchanged_in_old[p]+=1;continue
            if k in first and fetch.identity(card)!=fetch.identity(first[k]):raise ValueError('conflicting cards')
            first[k]=card;refs.setdefault(k,[]).append(p)
    out=[]
    pack_members={g['id'] for d in docs for g in d['games']}
    for key,card in sorted(first.items()):
        names=[card['name_'+s] for s in card['order']];score=[card['scores'][s] for s in card['order']]
        direct=[p for p in tree if p.endswith('/'+key+'.json.gz')]
        arm=[i for i in range(2) if names[i] in fetch.NAMES];url=fetch.SERVER+'/replay/'+card['id']+'/'+str(card['game'])
        g={'id':key,'source_index_paths':refs[key],'duplicate_score_mentions':len(refs[key])-1,'score_by_physical_seat':score,'players_by_physical_seat':names,
           'top5_appearances':[{'seat':i,'id':fetch.NAMES[names[i]],'name':names[i],'opponent':names[1-i],'wdl':('W' if score[i]>score[1-i] else 'L' if score[i]<score[1-i] else 'D')} for i in arm],
           'prior_1378_repeated':False,'present_in_frozen_git_verified_archive':key in pack_members,'frozen_git_raw_replay_paths':direct,'public_replay_url':url}
        cache=fetch.CACHE/'replays'/(key+'.json.gz')
        if direct:raw=git('show',LAST+':'+direct[0]);src='git_raw'
        elif cache.exists():raw=cache.read_bytes();src='verified_local_public_cache'
        else:
            retrieval=fetch.Retrieval({key:[]},0)
            try:raw=retrieval.get(url);src='official_public_get'
            except Exception as exc:
                g.update(replay_identity='not_available',error=repr(exc),seed=None,initial_terrain_u8_sha256=None);out.append(g);continue
        env,_,_=extract(raw);validated=validate(env,card)
        terrain=fetch.sha(bytes(env['steps'][0][0]['observation']['full_carbon']));seed=validated['random_seed']
        g.update(replay_identity=src,seed=seed,initial_terrain_u8_sha256=terrain,sha256_canonical_replay=fetch.sha(fetch.canonical(env)),
                 official_statuses=validated['statuses'],transitions=validated['last_step'],seed_seen_in_old=seed in seen_seed,
                 terrain_seen_in_old=seen_terrain.get(terrain),raw_fetched_body_sha256=fetch.sha(raw))
        out.append(g)
    seeds=[r['seed'] for r in out if r['seed'] is not None];terr=[r['initial_terrain_u8_sha256'] for r in out if r['seed'] is not None]
    if len(set(seeds))!=len(seeds) or len(set(terr))!=len(terr):raise ValueError('duplicate incoming map')
    doc={'state':'intake_only_not_training_not_play_analysis','anchor':ANCHOR,'revision':LAST,'previous_frozen_unique':1378,
         'new_unique':len(out),'total_score_cards':1378+len(out),'duplicate_mentions_within_new':sum(z['duplicate_score_mentions'] for z in out),
         'source_method':'Exhaustive registered score-index predicate on git diff. Any new card must be referenced in a changed index, or it already existed among all 1,378 cards confirmed by the prior complete scan. Checks standalone replay paths and all seven documented Git replay archive catalogs; live identity lookup does not deposit the raw tape in the repository.',
         'raw_replay_scope':'Git standalone files and seven verified archive catalogs through the pinned revision. Intake reserves new seeds and exact initial terrains separately from the frozen 302-game actor dataset.',
         'changed_sources':{p:{'git_blob':tree[p],'done_top5_mentions':seen_changed[p],'prior_unique_references':unchanged_in_old[p]} for p in changed},
         'new_by_registered_name':dict(Counter(v['name'] for r in out for v in r['top5_appearances'])),'ignored_nonterminal_score_references':dict(states),'games':out}
    (HERE/'next_ladder_intake.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n');print(len(out),len(seeds),doc['new_by_registered_name'])
if __name__=='__main__':data()
