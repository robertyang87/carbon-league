import os
from pathlib import Path
import hashlib,json,math,subprocess
repo=Path(os.environ['CARBON_REPO']).resolve()
out=Path(os.environ['CARBON_REVIEW_OUT']).resolve(); out.mkdir(parents=True,exist_ok=True)
review_sha=os.environ.get('CARBON_REVIEW_REF','5463f504048bcca28868ebce0a063ac8d6c51b8e')
sg=lambda x:1/(1+math.exp(-x))
def u(m):return .5*(1 if m>0 else .5 if m==0 else 0)+.5*sg(m/1000)
def stats(xs):return {'expected_margin':sum(p*m for p,m in xs),'p_win':sum(p for p,m in xs if m>0),'expected_platform_U':sum(p*u(m) for p,m in xs)}
def huber(res,delta):return .5*res*res if abs(res)<=delta else delta*(abs(res)-.5*delta)
# A 90/10 two-point distribution has derivative .9*x-.1*delta=0 in the relevant region.
p0,p1=0.9,0.1
delta=1000.;robust=p1*delta/p0;mean=p1*10000
score=lambda x:sum(p*huber(x-y,delta) for p,y in [(p0,0),(p1,10000)])
ce=lambda ps:-sum(.5*math.log(x) for x in ps)
a=[.45,.45];b=[.005,.005]
regular=.25*200+.75*20
late=.25*400+.75*20
sims_all=270*regular+29*late
sims_from200=70*regular+29*late
result={
 'baseline_sha':review_sha,
 'evidence_scope':'Numerical counterexamples and static contracts; no training, production bug, natural-game incidence or performance claim.',
 'files':{str(p):hashlib.sha256(subprocess.check_output(['git','-C',str(repo),'show',review_sha+':'+p])).hexdigest() for p in ['reports/katago_architecture_design.md','reports/live/plan_to_first.md','reports/REVIEW_FIXES_20261113.md']},
 'D02_full_joint_CE':{'old_conditioned_ce_a':ce([x/sum(a) for x in a]),'old_conditioned_ce_b':ce([x/sum(b) for x in b]),'new_full_joint_ce_a':ce(a),'new_full_joint_ce_b':ce(b),'new_formula_distinguishes_bad_mass':ce(b)>ce(a)},
 'D03_Huber_not_mean':{'distribution':[{'probability':p0,'margin':0},{'probability':p1,'margin':10000}],'delta':delta,'true_mean':mean,'huber_optimum':robust,'huber_risk_at_optimum':score(robust),'huber_risk_at_true_mean':score(mean),'MAE_at_optimum':p0*abs(robust)+p1*abs(10000-robust),'MAE_at_true_mean':p0*abs(mean)+p1*abs(10000-mean),'safe_action_margin':500,'huber_argmax':'safe','mean_argmax':'lottery'},
 'D03_mean_sign_not_winrate':stats([(.9,-100),(.1,10000)]),
 'N01_platform_objective_reversal':{'safe_action':stats([(1.,100)]),'risky_action':stats([(.9,-100),(.1,10000)]),'margin_argmax':'risky_action','platform_U_argmax':'safe_action'},
 'D01_strategy_fusion_test':{'hidden_states':['L','R'],'probabilities':[.5,.5],'followup_observation_same_in_both_states':True,'payoffs':{'left':[10,0],'right':[0,10],'safe':[6,6]},'full_state_separate_max_expected_value':10,'information_set_followup_optimal_value':5,'root_safe_value':6,'correct_root_action':'safe','particle_separate_action_root_action':'defer_and_select_with_hidden_state','status':'Design ambiguity test, not demonstrated production-code leakage'},
 'D06_arithmetic':{'ordinary_expected_sims':regular,'late_expected_sims':late,'document_90_10_approx':.9*regular+.1*late,'299_decisions_exact_mean':sims_all/299,'from_step200_suffix_cost_fraction_same_search_budget':sims_from200/sims_all,'assumptions':'decisions t=0..298; t>=270 gets late budget; one searched player; no setup, reuse or batching cost assumed'},
 'D07_same_summary_different_utility':{'policy_A':stats([(1,1000)]),'policy_B':stats([(.5,6000),(.5,-4000)]),'same_expected_final_margin':True,'other_features_assumption':'same tree counts / first-tree timings and same mean cash before terminal; illustrates aggregated diagnostic summaries do not define strength'},
 'N04_operational_vs_significance':{'mean_gain':90,'SE_gain':1,'gain_div_SE':90,'operational_min_gain':100,'passes_operational_gate':False,'conclusion':'Failure of an operational >=100 gate is not absence of a statistically detectable effect.'},
 'deployment_int8_raw_bytes':{'12M_parameters':12_000_000,'14M_parameters':14_000_000,'platform_file_limit':10_000_000,'status':'Raw byte accounting only; actual compressed serialized artifact must be measured.'}
}
assert result['D02_full_joint_CE']['new_formula_distinguishes_bad_mass']
assert score(robust)<score(mean)
assert result['N01_platform_objective_reversal']['risky_action']['expected_margin']>result['N01_platform_objective_reversal']['safe_action']['expected_margin']
assert result['N01_platform_objective_reversal']['risky_action']['expected_platform_U']<result['N01_platform_objective_reversal']['safe_action']['expected_platform_U']
(out/'evidence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
