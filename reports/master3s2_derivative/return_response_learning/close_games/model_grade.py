"""Grade the precommitted predictions only when all four natural-response map pairs exist."""
import hashlib
import json
from pathlib import Path
from statistics import mean

HERE=Path(__file__).resolve().parent
BLIND='8d90d61ac02281b59a67818116f840e10ba94d9245835b0183a71d5efb438489'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def sign(x):return 1 if x>1e-9 else -1 if x<-1e-9 else 0

def main():
    bp=HERE/'model_blind.json';assert sha(bp)==BLIND;pred=json.loads(bp.read_bytes());files=[HERE/'cases'/f'{i}.json' for i,_ in enumerate(pred['rows'])]
    if not all(p.exists() for p in files):raise SystemExit('Waiting for the entire precommitted panel')
    rows=[]
    for p,m in zip(files,pred['rows'],strict=True):
        c=json.loads(p.read_bytes());t=m['ticket'];assert c['forecast_sha256']==BLIND and c['menu_sha256']==pred['menus_sha256'] and c['protocol']==pred['protocol_sha256'] and (c['seed'],c['seat'],c['step'])==(t['seed'],t['seat'],t['step']);seat=c['seat'];d=c['delta_A_minus_B'];assert set(c['arms'])=={'A','B'}
        if c.get('status') in ('missing','failed'):rows.append(dict(seed=t['seed'],status=c['status']));continue
        for k,v in c['arms'].items():
            credited=sum(min(mass,v['first_credited_payments_by_seat'][seat].get(uid,0)) for uid,mass in v['root_carriers_by_seat'][seat].items());assert abs(credited-v['actual_initial_same_carrier_cash_by_seat'][seat])<1e-7
        true=d['same_carrier_own'];cash=m['money_A_minus_B'];term=d['terminal_margin'];native=m['native_A_minus_B'];adj=m['native_plus_correction_A_minus_B'];tgt=d['terminal_native'];new=m['selected_if_scored'];parent=m['baseline_native_pick'];own=[c['arms'][a]['actual_initial_same_carrier_cash_by_seat'][seat] for a in ('A','B')]
        rows.append(dict(seed=c['seed'],seat=seat,step=c['step'],case_sha256=sha(p),status='complete',coverage=m['coverage'],predicted_bank=m['predicted_money'],realized_initial_bank=own,expected_initial_gap=cash,actual_initial_gap=true,gap_error=cash-true,
                gap_sign=sign(cash),actual_gap_sign=sign(true),own_total_deposit12_gap=d['own_team_bank12'],net_total_deposit12_gap=d['net_team_bank12'],terminal_diff=term,terminal_native_target=tgt,native_gap=native,adjusted_gap=adj,
                paid_A=c['arms']['A']['margin'],paid_B=c['arms']['B']['margin'],native_winner=parent,selected=new,terminal_choice_correct=int(sign(adj)==sign(term)) if sign(term) else None,raw_margin_gain_vs_native=c['arms'][new]['margin']-c['arms'][parent]['margin'],observed_win_delta_vs_native=c['arms'][new]['win_indicator']-c['arms'][parent]['win_indicator']))
    valid=[r for r in rows if r['status']=='complete'];ne=[r for r in valid if r['gap_sign'] and r['actual_gap_sign']];nt=[r for r in valid if sign(r['terminal_diff'])];summary=dict(blind_sha256=BLIND,maps=len(rows),complete=len(valid),receipt_gap_strict_correct=sum(r['gap_sign']==r['actual_gap_sign'] for r in ne),receipt_gap_strict_tested=len(ne),actual_receipt_ties=sum(not r['actual_gap_sign'] for r in valid),predicted_receipt_ties=sum(not r['gap_sign'] for r in valid),
        mean_same_carrier_gap_absolute_error=mean(abs(r['gap_error']) for r in valid),baseline_zero_gap_mae=mean(abs(r['actual_initial_gap']) for r in valid),
        terminal_correct=sum(r['terminal_choice_correct'] for r in nt),terminal_non_ties=len(nt),changed_issued_count=sum(r['selected']!=r['native_winner'] for r in valid),mean_terminal_gain_vs_native=mean(r['raw_margin_gain_vs_native'] for r in valid),observed_wins_gain_vs_native=sum(r['observed_win_delta_vs_native'] for r in valid),original_value_gap_mse=mean((r['native_gap']-r['terminal_native_target'])**2 for r in valid),adjusted_value_gap_mse=mean((r['adjusted_gap']-r['terminal_native_target'])**2 for r in valid),rows=rows)
    (HERE/'model_grade.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');print({k:v for k,v in summary.items() if k!='rows'},flush=True)

if __name__=='__main__':main()
