"""Freeze historically exposed and mutual Master6 losses as a separate post-training diagnostic."""
import json,gzip
from pathlib import Path
from data_build import prep,sha,HERE

def main():
    src=HERE/'data_split.json';rows=json.loads(src.read_bytes())['games']
    manifests={}
    eligible=[];rejected=[];out=Path('/workspace/sapling-master6-actor-exposed-loss');out.mkdir(exist_ok=True)
    for r in rows:
        if r['split']!='excluded' or r['wdl']!='L':continue
        if r['batch'] not in manifests:
            cards=json.loads((HERE.parent/'all_top5'/r['batch']/'manifest.json').read_bytes())['games']
            manifests[r['batch']]={g['id']:g for g in cards}
        g=manifests[r['batch']][r['id']]
        if not g['clean_terminal'] or g['terminal']!=['DONE','DONE']:
            rejected.append({'id':r['id'],'reason':'source terminal or execution status'});continue
        eligible.append({**r,'original_split':'excluded','split':'diagnostic_seen_loss'})
    dest=HERE/'data_loss_split.json'
    doc={'schema':'master6-postfit-seen-loss/1','original_split_sha256':sha(src.read_bytes()),'games':eligible,'nonclean':rejected,
         'scope':'Historically exposed and/or mutual losses only, outcomes purpose-selected. Open only after the new candidate was predetermined. Never used to select or retrain this candidate.'}
    b=(json.dumps(doc,ensure_ascii=False,indent=1)+'\n').encode()
    if dest.exists() and dest.read_bytes()!=b:raise ValueError('diagnostic selection differs')
    dest.write_bytes(b)
    infos=[prep(g,out) for g in eligible]
    full={'data_split_sha256':sha(b),'shard_directory':str(out),'games':infos,'rows':sum(g['rows'] for g in infos)}
    dest=HERE/'data_loss_receipts.json.gz';tmp=dest.with_suffix('.partial');tmp.write_bytes(gzip.compress(json.dumps(full,separators=(',',':')).encode(),mtime=0));tmp.replace(dest)
    print(len(eligible),len(rejected),full['rows'])
if __name__=='__main__':main()
