from pathlib import Path
import gzip,json,hashlib,zipfile
HERE=Path(__file__).resolve().parent
FAMILY=HERE.parents[1]
def fp(raw):return hashlib.sha256(raw).hexdigest()
def main():
    source=HERE.parent/'warmed.json.gz';source_data=json.loads(gzip.decompress(source.read_bytes()))
    register=json.loads((HERE.parent/'protocol.json').read_bytes());out=[]
    for row in source_data['rows']:
        if not row['changed']:continue
        seed,s,t=row['seed'],row['seat'],row['t'];n=row['natural_abba']
        assert all(i['action']==row['reference' if i['mode']=='P' else 'corrected']['action'] for i in n)
        assert all(i['menu']==n[0]['menu'] and i['clocks']['K']==n[0]['clocks']['K'] and i['clocks']['H']==n[0]['clocks']['H'] for i in n)
        key=next(a for a in register['archives'] if (lambda g:(g['seed'],g['candidate_seat']))(json.loads(zipfile.ZipFile(FAMILY/'cold_start/development/hosted'/f'{a}.zip').read('cold_repair/registration.json')))==(seed,s))
        archive=FAMILY/'cold_start/development/hosted'/f'{key}.zip'
        tr=zipfile.ZipFile(archive);reg=json.loads(tr.read('cold_repair/registration.json'))
        ticket=dict(seed=seed,seat=s,step=t,archive=key,archive_sha256=fp(archive.read_bytes()),terrain=reg['physical_map_sha256'],
                    orders={k:row[v]['action'] for k,v in [('stock','reference'),('home','corrected')]},
                    streams=[dict(kind=k,prng=[(seed+340207+i*1000003+physical*65537+t*73)%2**32 for physical in (0,1)],
                                  order=(['stock','home'] if (i+len(out))%2==0 else ['home','stock']))
                             for i,k in enumerate(['archived-root-response','fresh-1','fresh-2'])],empirical_costs=row['costs'])
        out.append(ticket)
    assert len(out)==3 and len({t['terrain'] for t in out})==2
    job=dict(warmed_source=fp(source.read_bytes()),candidate=register['candidate'],parent=register['parent'],
             original='65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5',
             target='All 3 previously changed roots, two exposed terrain clusters. A and B are the warmed frozen full teams. Intervene once, then common parent 7cf5 responds naturally versus stock original at platform model sampling temp and game native clock. Per root, report observed archived opponent root command separately plus two independent natural original root replies. All three streams initialize registered seat RNGs. Common branch-pair RNG, official and policy states; warm opponent cost using prior real official search. Stream with archive-root uses historically issued enemy root move but fresh independent suffix entropy; archived private entropy is absent. No early stopping; no using results to move roots.',tickets=out)
    raw=(json.dumps(job,sort_keys=True,indent=2)+'\n').encode();p=HERE/'protocol.json';assert not p.exists() or p.read_bytes()==raw;p.write_bytes(raw);print(fp(raw))
if __name__=='__main__':main()
