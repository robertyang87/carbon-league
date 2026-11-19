"""Separate additions to score indexes, cross-index mirrors, previously sealed cards, and actual raw replay availability."""
from collections import Counter,defaultdict
import gzip,json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;TOP=HERE.parent
sys.path.insert(0,str(TOP))
import fetch
PREVIOUS='0137949d48ff0b954835abbba9a68afa058a281e'
REV='41e4177eaaa0ca5c808e28a5a4290b5a88355205'
BATCHES=('.', 'appendix_499886f1','appendix_4bfdabf5','appendix_555c0c56','appendix_bf049043','appendix_bd3c0021','appendix_5484590b','appendix_99919cbf','appendix_0137949d')

def decode(raw):return json.loads(gzip.decompress(raw) if raw[:2]==b'\x1f\x8b' else raw)

def scan():
    known={};batches={}
    for name in BATCHES:
        manifest=json.loads((TOP/name/'manifest.json').read_bytes());batches[name]=len(manifest['games'])
        for g in manifest['games']:
            if g['id'] in known:raise ValueError('overlapping frozen manuscripts')
            known[g['id']]=(g,name)
    if len(known)!=1386:raise ValueError('wrong baseline count')
    changed=fetch.git('diff','--name-status','--no-renames',PREVIOUS,REV).decode().splitlines();refs=defaultdict(list);cards={};counts={}
    for change in changed:
        status,f=change.split('\t')
        if not fetch.sourcefile(f):continue
        stores=[]
        for missing,rev in [(status=='A',PREVIOUS),(status=='D',REV)]:
            records={} if missing else {fetch.key_of(r):r for r in fetch.walk_matches(decode(fetch.git('show',rev+':'+f))) if r.get('state','done')=='done' and {r['name_a'],r['name_b']}&set(fetch.NAMES)}; stores.append(records)
        before,after=stores
        for k in before.keys()&after.keys():
            if fetch.identity(before[k])!=fetch.identity(after[k]):raise ValueError('changed record '+k)
        added=sorted(after.keys()-before.keys());removed=sorted(before.keys()-after.keys())
        counts[f]={'top5_reference_before':len(before),'top5_reference_after':len(after),'added':len(added),'removed':removed}
        for k in added:
            if k in cards and fetch.identity(after[k])!=fetch.identity(cards[k]):raise ValueError('conflicting references')
            refs[k].append(f);cards[k]=after[k]
    horizon=HERE/'manifest.json';new_published={} if not horizon.exists() else {g['id']:g for g in json.loads(horizon.read_bytes())['games']}
    rows=[];members=set();fetch.REV=REV;tree=fetch.tree()
    for k,card in sorted(cards.items()):
        present=k in known;names=[card['name_'+s] for s in card['order']];scores=[card['scores'][s] for s in card['order']];old,old_batch=known.get(k,({},None));payload=old if present else new_published.get(k,{})
        if present and fetch.identity(card)!=fetch.identity(old['card']):raise ValueError('index contradicts previous card')
        verified=payload.get('frame_verified') is True
        seed=payload.get('seed');raw_path=payload.get('archive_location');raw_sha=payload.get('sha256');surface=sorted(f for f in tree if f.endswith('/'+k+'.json.gz'))
        if verified and not present:members.add(k)
        rows.append(dict(id=k,players_physical=names,score_physical=scores,top5_cohort='mutual' if all(n in fetch.NAMES for n in names) else 'external',added_index_references=refs[k],already_in_1386=present,duplicate_new_index_references=max(0,len(refs[k])-1),previous_batch=old_batch,new_archive_member=k in members,seed=seed,raw_archive=raw_path,raw_sha256=raw_sha,raw_verified=verified,git_source_raw_paths_at_score_revision=surface,receipt_state='archived_before_cut' if present else 'archived_in_new_batch' if verified else 'score_card_pending_replay',source_retrieval=payload.get('source')))
    if new_published and set(new_published)!={k for k in cards if k not in known}:raise ValueError('independent score diff differs from the complete index census')
    result={'previous_revision':PREVIOUS,'revision':REV,'previous_batches':batches,'prior_verified_games':len(known),'added_score_mentions':sum(len(v) for v in refs.values()),'distinct_added_references':len(cards),'new_unique_games':sum(not x['already_in_1386'] for x in rows),'previous_archived_referenced_again':sum(x['already_in_1386'] for x in rows),'within_delta_mirror_extras':sum(x['duplicate_new_index_references'] for x in rows),'newly_archived_raw':len(members),'unique_by_frozen_indexes':len(known)+sum(not x['already_in_1386'] for x in rows),'changed_sources':counts,'games':rows}
    dst=TOP/'intake_41e4177e.json';fetch.atomic(dst,(json.dumps(result,ensure_ascii=False,indent=1)+'\n').encode());return result
if __name__=='__main__':
    doc=scan();print({k:v for k,v in doc.items() if isinstance(v,int)},flush=True)
