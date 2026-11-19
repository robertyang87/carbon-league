"""Build the small reversible decision-address table directly from the archived match manifest."""
import argparse
import csv
import gzip
import io
import json
from pathlib import Path
import tarfile

if __package__:
    from .context import HERE, ROOT, digest, git_blob
else:
    from context import HERE, ROOT, digest, git_blob


def catalog(manifest, root=ROOT):
    raw=Path(manifest).read_bytes()
    doc=json.loads(raw)
    rows=[];seen=set();archives={};start=0
    for card in sorted(doc["games"],key=lambda r:r["id"]):
        gid=card["id"];loc=card["archive_location"]
        if gid in seen or not card.get("frame_verified") or not card["sha256"]:
            raise ValueError(f"duplicate or unverified replay {gid}")
        seen.add(gid)
        source=loc.get("archive") or loc["repo_path"]
        if Path(source).is_absolute() or ".." in Path(source).parts:
            raise ValueError(f"invalid repository path {source}")
        member=loc.get("member","");offset="";length=""
        if member:
            if source not in archives:
                path=root/source
                tmp=None if path.exists() else io.BytesIO(git_blob(root,"HEAD:"+source))
                with tarfile.open(path if tmp is None else None, fileobj=tmp, mode="r:*") as ar:
                    archives[source]={m.name:(m.offset_data,m.size) for m in ar if m.isfile()}
            data_offset,length=archives[source][member]
            if source.endswith(".tar"): offset=data_offset
        n=int(card["transitions"])
        if n<=0:raise ValueError(f"missing decision frames {gid}")
        rows.append(dict(id=gid,ordinal_begin=start,ordinal_end_exclusive=start+2*n,transitions=n,
                         source=source,member=member,git_blob=loc.get("git_blob",""),tar_byte_offset=offset,
                         packed_bytes=length,replay_sha256=card["sha256"]))
        start+=2*n
    return rows


def write_index(manifest, output):
    rows=catalog(manifest)
    f=io.StringIO(newline="")
    w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter="\t",lineterminator="\n")
    w.writeheader();w.writerows(rows)
    output.write_bytes(gzip.compress(f.getvalue().encode(),mtime=0))
    print(len(rows),rows[-1]["ordinal_end_exclusive"],digest(output.read_bytes()))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,default=HERE/"manifest.json")
    p.add_argument("--output",type=Path,default=HERE/"context_index.tsv.gz")
    args=p.parse_args();write_index(args.manifest,args.output)
