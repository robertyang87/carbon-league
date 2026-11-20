"""Apply the already frozen model to all eight archived response-check tickets, with no new games."""
from pathlib import Path
import gzip,json,zipfile
from fit import HERE,DER,sha,read,save
from model import rr_scores


def main():
    p=DER/'endgame_cash_projection/response_check';w=read(HERE/'weights.json');assert sha((HERE/'weights.json').read_bytes())=='7a1270e3905e396d24acf79bb6b33f1308628e7892b809f6013e6b37c55fdecf'
    out=[]
    for i in range(8):
        raw=read(p/'roots'/f'{i}.json');card=raw['ticket'];s=card['seat'];t=card['step'];book=raw['source'];arm=card['arm'];arc=DER/'ladder/cargo_defense/development/hosted'/(str(book['artifact_id'])+'.zip');assert sha(arc.read_bytes())==book['archive_sha256']
        with zipfile.ZipFile(arc) as z:
            frames=z.read(f'{arm}/observations.jsonl.gz');assert sha(frames)==book['captured_observations_sha256'];row=json.loads(gzip.decompress(frames).splitlines()[t]);obs=row['observations'][s];assert obs['step']==t and obs['player']==s
        rec=dict(seed=card['seed'],step=t,seat=s,history_arm=arm,source=raw['status'],qualifies_for_main_gate=False)
        if raw['status']!='scored_pair': rec['reason']='Historical pair wrapper selected the same first action. No two distinct real responses available.';out.append(rec);continue
        parent=raw['matched']['parent'];head=raw['matched']['head'];menu=parent['menu'];a=menu.index(parent['order']);b=menu.index(head['order']);assert a!=b and a in head['pair'] and b in head['pair']
        d={int(k):v for k,v in parent['dep'].items()};v={int(k):v for k,v in parent['native_mean'].items()};scores=rr_scores(obs['players'],s,t,[menu[a],menu[b]],w);delta=float(scores[0]-scores[1]);original=v[a]-v[b];adjusted=original+w['pair_scale']*delta
        results={};
        for name in ('parent','head'):
            e=raw['branch']['arms'][name];backup=p/'branches'/arm/f"{card['seed']}_{s}_{t}_{name}.json.gz";assert sha(backup.read_bytes())==e['evidence_sha256'];br=read(backup);flow=[sum(q['flow'][side].get('own_delivery',0) for q in br['rows'][:12]) for side in (s,1-s)];results[name]=dict(margin=e['margin'],official_scores=e['scores'],early_own_enemy_receipts=flow)
        rec.update(depths=[d[a],d[b]],native_delta=original,estimated_receipt_delta=delta,adjusted_delta=adjusted,response=results,selects_other_if_eligible=bool(d[a]==d[b]==8 and adjusted<0),delta_official_margin_parent_minus_alternative=results['parent']['margin']-results['head']['margin']);out.append(rec)
    save('outside_pairs.json',dict(frozen_weights='7a1270e3905e396d24acf79bb6b33f1308628e7892b809f6013e6b37c55fdecf',scope='After the primary held result, all 8 archived response check tickets elsewhere, no refit, no new probe',ticket=out))

if __name__=='__main__':main()
