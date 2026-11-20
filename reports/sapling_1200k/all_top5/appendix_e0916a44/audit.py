"""Compare repository-wide scorecard unions and locally stored raw replay identities at two fixed commits."""
from collections import Counter
import gzip
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
TOP=HERE.parent
sys.path.insert(0,str(TOP))
import fetch
from continue_appendix import previous_manifests

OLD='ea2249324c75a820c983b56a2557278ac703364d'
NEW='e0916a448e36b37e87537bebd5b67ca026a548f0'


def tree(revision):
    return {path:head.split()[2] for line in fetch.git('ls-tree','-r',revision).decode().splitlines()
            for head,path in [line.split('\t',1)]}


def census(paths,cache):
    union={}
    sources={}
    refs={}
    all_sources={path:blob for path,blob in paths.items() if fetch.sourcefile(path)}
    with subprocess.Popen(['git','-C',str(fetch.ROOT),'cat-file','--batch'],stdin=subprocess.PIPE,stdout=subprocess.PIPE) as git:
        for path,blob in sorted(all_sources.items()):
            if blob not in cache:
                git.stdin.write((blob+'\n').encode())
                git.stdin.flush()
                header=git.stdout.readline().split()
                assert header[0].decode()==blob and header[1]==b'blob'
                raw=git.stdout.read(int(header[2]))
                assert git.stdout.read(1)==b'\n'
                records={}
                for row in fetch.walk_matches(json.loads(gzip.decompress(raw) if raw.startswith(b'\x1f\x8b') else raw)):
                    if row.get('state','done')!='done':
                        continue
                    gid=fetch.key_of(row)
                    order=row['order']
                    assert sorted(order)==['a','b']
                    card=tuple((row['name_'+seat],row['scores'].get(seat)) for seat in order)
                    assert records.setdefault(gid,card)==card
                cache[blob]=records
            records=cache[blob]
            if records:
                sources[path]=blob
                refs[path]=set(records)
                for gid,card in records.items():
                    assert union.setdefault(gid,card)==card
        git.stdin.close()
    return union,sources,refs


def main():
    trees={r:tree(r) for r in (OLD,NEW)}
    parsed={}
    old,old_sources,old_refs=census(trees[OLD],parsed)
    new,new_sources,new_refs=census(trees[NEW],parsed)
    assert all(gid in new and new[gid]==card for gid,card in old.items())
    def tier(card):
        count=sum(name in fetch.NAMES for name,_ in card)
        return ('external' if count==1 else 'mutual') if count else 'other'
    def counted(card):return all(isinstance(score,(int,float)) for _,score in card)
    published={}
    faults={}
    manifests=previous_manifests(TOP/'appendix_ea224932')
    for path in manifests:
        for g in fetch.load(path)['games']:
            assert g['id'] not in published
            published[g['id']]=list(zip(g['seat_names'],g['score_by_seat']))
        fault=path.parent/'unscored/manifest.json'
        if fault.exists():
            for g in fetch.load(fault)['games']:
                faults[g['id']]=str(fault.relative_to(TOP))
    prior={k for k,c in old.items() if tier(c)!='other' and counted(c)}
    now={k for k,c in new.items() if tier(c)!='other' and counted(c)}
    partial={k for k,c in new.items() if tier(c)!='other' and not counted(c)}
    assert prior==now==set(published) and all(list(old[k])==published[k] for k in published)
    assert partial==set(faults)=={k for k,c in old.items() if tier(c)!='other' and not counted(c)}
    changed={}
    added={}
    for path in sorted(set(old_sources)|set(new_sources)):
        if old_sources.get(path)==new_sources.get(path):
            continue
        a,b=old_refs.get(path,set()),new_refs.get(path,set())
        assert not (a-b)
        changed[path]=dict(before=len(a),after=len(b),additions=sorted(b-a),old_blob=old_sources.get(path),new_blob=new_sources.get(path))
        for gid in sorted(b-a):
            added.setdefault(gid,dict(seats=new[gid],group=tier(new[gid]),previously_indexed=gid in old,paths=[]))['paths'].append(path)
    fresh=set(new)-set(old)
    assert fresh<=set(added) and not fresh & now
    corpus={}
    for path,blob in sorted(trees[NEW].items()):
        if trees[OLD].get(path)==blob or not path.startswith('data/') or '_corpus/' not in path:
            continue
        match=fetch.ID.fullmatch(Path(path).name)
        if not match:
            continue
        gid=match[1]
        env=json.loads(gzip.decompress(fetch.git('cat-file','blob',blob)))
        card=new[gid]
        row=dict(id=gid.rsplit('-',1)[0],game=gid.rsplit('-',1)[1],order=['a','b'],name_a=card[0][0],
                 name_b=card[1][0],scores=dict(a=card[0][1],b=card[1][1]))
        proof=fetch.validate(env,row)
        corpus[path]=dict(gid=gid,group=tier(card),previously_indexed=gid in old,blob=blob,proof=proof)
    result=dict(previous=OLD,revision=NEW,ranking=fetch.RANKS,previous_archive_files=len(manifests),
                previous_distinct_indexed_games=len(old),current_distinct_indexed_games=len(new),
                previous_score_index_files=len(old_sources),current_score_index_files=len(new_sources),
                changed_indexes=changed,added_reference_ids=added,added_references=sum(len(x['paths']) for x in added.values()),
                distinct_added_reference_ids=len(added),first_indexed_game_ids=sorted(fresh),
                first_indexed_games=len(fresh),old_games_mirrored=len(set(added)-fresh),
                new_scored_top_five_ids=sorted(now-prior),active_top_five_faults=faults,
                prior_scored_top_five_ids_sha256=fetch.sha('\n'.join(sorted(prior)).encode()),
                updated_scored_top_five_ids_sha256=fetch.sha('\n'.join(sorted(now)).encode()),
                scored_by_group=dict(sorted(Counter(tier(new[k]) for k in now).items())),total_scored=len(now),fault_count=len(faults),
                changed_replay_files=corpus,qualifying_games_requiring_new_tactical_replay=0,new_downloads=0)
    fetch.dump(HERE/'receipt.json',result)


if __name__=='__main__':
    main()
