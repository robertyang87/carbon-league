"""Independent readback of pack identity, terminal payoffs, search clocks, pairings and release."""
from collections import Counter,defaultdict
import gzip,hashlib,json
from pathlib import Path
from register_fit import HERE,ROOT,sha
def read(p):return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())
def main():
    p=read(HERE/'protocol.json');sel=read(HERE/'selection.json');b=read(HERE/'binding.json');rows=read(HERE/'scores.json')
    assert sha(ROOT/sel['selected_pack_file'])==sel['selected_pack_sha256'] and sha(ROOT/sel['selected_weights_file'])==sel['selected_weights_sha256']
    want={(k,s,side,cpu) for e in p['waves'] for k,s,side,cpu in e}
    assert len(want)==len(rows)==8 and want=={(r['sign'],r['seed'],r['seat'],r['cpu']) for r in rows}
    checks=[]
    for r in rows:
        q=HERE/'games'/r['driver_label'];stem=f"{r['seed']}_{r['seat']}"
        raw=q/f'raw_{stem}.json.gz';receipt=q/f'receipt_{stem}.json';wall=q/f'wall_{stem}.json.gz'
        assert sha(raw)==r['raw_sha256'] and sha(wall)==r['wall_sha256'] and sha(receipt)==r['receipt_sha256']
        doc,off,cl=read(raw),read(receipt)['official'],read(wall);calls=cl['calls']
        assert off['statuses']==['DONE','DONE'] and off['transcript_sha256']==sha(raw) and not doc['errors'] and not doc['audit_messages']
        assert doc['configuration']['_TEMP']==1 and doc['configuration']['_SEED'] is None and doc['configuration']['_S_ABORT']==.58
        assert doc['pack_sha256']==b['signs'][r['sign']]['pack']==sha(HERE/(r['sign']+'.py'))
        seat=r['seat'];v=off['official_reward_by_seat'];assert v[seat]-v[1-seat]==r['margin']
        assert len(calls)==len(doc['actions'])==doc['steps']==r['actual_opponent_calls'] and cl['identity']['cpu']==r['cpu']
        checks.append(dict(seed=r['seed'],seat=seat,sign=r['sign'],conf_budget=.58,temperature=1,rng_seed=None,cpu=r['cpu'],steps=len(calls),
                           official_margin=r['margin'],natural_search_depth8=sum(c.get('max_depth')==8 for c in calls),
                           original_public_wall_seconds=sum(c['public_wall_seconds'] for c in calls)))
    out=dict(selection_sha256=sha(HERE/'selection.json'),actual_training_games=len(rows),distinct_maps=len(set(r['seed'] for r in rows)),original_clock_rechecked=checks,chosen_pack=sel['selected_pack_sha256'],rules={'source_examiner':p['examiner']['sha256'],'unmodified_temperature':1,'unmodified_budget':.58,'new_eval_maps':0})
    (HERE/'verification.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
