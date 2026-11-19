"""Cross-check each new fixture against every tracked family score index, including non-Top-Five publishers."""
import json
import re
import subprocess
from latest5_inventory_github import BASE,ROOT,REPORT,dump,identity,key_of,references,walk_matches


def indexes(ref,names):
    tree=subprocess.check_output(["git","-C",str(ROOT),"ls-tree","-r","--name-only",ref,"data"],text=True)
    paths=[p for p in tree.splitlines() if re.fullmatch(r"data/[^/]+_matches\.json",p)]
    with subprocess.Popen(["git","-C",str(ROOT),"cat-file","--batch"],stdin=subprocess.PIPE,stdout=subprocess.PIPE) as git:
        grouped={}
        for path in paths:
            git.stdin.write((ref+":"+path+"\n").encode());git.stdin.flush()
            size=int(git.stdout.readline().split()[2]); data=git.stdout.read(size);git.stdout.read(1)
            for row in walk_matches(json.loads(data)):
                if row.get("state")!="done" or not {row["name_a"],row["name_b"]}&names:continue
                k=key_of(row)
                if k in grouped and grouped[k]["seats_and_scores"]!=identity(row):raise ValueError("conflicting card "+k)
                grouped.setdefault(k,dict(seats_and_scores=identity(row),sources=[]))["sources"].append(path)
        git.stdin.close()
    return grouped,paths


def main():
    from argparse import ArgumentParser
    p=ArgumentParser(description=__doc__);p.add_argument("--as-of",default="6d24fa0f");a=p.parse_args()
    names,*_=references()
    base,old_paths=indexes(BASE,names);head,paths=indexes(a.as_of,names)
    data=json.loads((REPORT/"latest5_inventory.json").read_bytes())
    targets={g["id"] for g in data["games"]}
    mutual=lambda d:sorted(k for k,g in d.items() if len({n for n,s in g["seats_and_scores"]})==2 and {n for n,s in g["seats_and_scores"]}<=names)
    summary=dict(as_of=subprocess.check_output(["git","-C",str(ROOT),"rev-parse",a.as_of],text=True).strip(),
                 base=BASE,as_of_num_indexes=len(paths),base_num_indexes=len(old_paths),baseline_matches=len(base),
                 as_of_matches=len(head),as_of_mutual=len(mutual(head)),base_mutual=len(mutual(base)),
                 verified_present_in_github_before=sorted(targets&set(base)),verified_present_in_github_now=sorted(targets&set(head)),
                 unique_new_all_indexes=sorted(set(head)-set(base)),
                 per_verified_replay_github_index_sources={k:head.get(k,{}).get("sources",[]) for k in sorted(targets)})
    dump(REPORT/"latest5_inventory_repo.json",summary)
    if len({g["id"] for g in data["games"]})!=len(data["games"]):
        raise ValueError("duplicate increment")
    for g in data["games"]:
        g["github_any_index"]=g["id"] in head
        g["github_index_paths"]=summary["per_verified_replay_github_index_sources"][g["id"]]
    data["all_github_families_as_of"]=summary["as_of"]
    data["all_github_families_index_count"]=len(paths)
    data["all_families_present_in_github"]=len(targets&set(head))
    data["absent_from_all_github_families"]=len(targets-set(head))
    dump(REPORT/"latest5_inventory.json",data)
    print({k:v for k,v in summary.items() if not isinstance(v,(list,dict))},len(targets&set(head)))


if __name__=="__main__":
    main()
