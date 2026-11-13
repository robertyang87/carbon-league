#!/usr/bin/env python3
"""Portable read-only review probes. No model load, training, or network access.
Usage: python check_method.py --repo /path/to/carbon-league --out /tmp/new-evidence
Requires NumPy only for calling the repository's extracted reward function.
"""
import argparse, ast, hashlib, json, math, subprocess
from pathlib import Path
from types import SimpleNamespace

def main():
 p=argparse.ArgumentParser(); p.add_argument('--repo',type=Path,required=True); p.add_argument('--out',type=Path,required=True);p.add_argument('--ref',default='98ffadfa19a8db8a1d91998386e7fa9dbfc61a4c');a=p.parse_args()
 def read(path):return subprocess.check_output(['git','-C',str(a.repo),'show',f'{a.ref}:{path}'],text=True)
 files=['reports/katago_architecture_design.md','reports/live/plan_to_first.md','reports/CARBONFORMER_KATAGO_DESIGN_20261113.md','code/rl2_ppo_train.py','code/cnet2_build.py','code/cnet2_model.py','code/cnet2_train.py']
 sources={f:read(f) for f in files};arch=sources[files[0]];plan=sources[files[1]]
 sig=lambda x:1/(1+math.exp(-x))
 def utility(m):return .5*(1 if m>0 else .5 if m==0 else 0)+.5*sig(m/1000)
 # Proper mean improves MSE and calibration while MAE worsens, including legal normal-game U labels.
 dist=[(.9,.1),(.1,.9)];target_mean=sum(p*y for p,y in dist)
 mae_rows=[{'prediction':v,'MAE':sum(p*abs(v-y) for p,y in dist),'MSE':sum(p*(v-y)**2 for p,y in dist),'mean_bias':v-target_mean} for v in [.1,.14,.18]]
 assert all(mae_rows[i+1]['MSE']<mae_rows[i]['MSE'] and mae_rows[i+1]['MAE']>mae_rows[i]['MAE'] for i in range(2))
 # An exact root-Q maximizer can improve but the alpha=.5 soft target can be worse.
 priors=[.99]+[.0099/14]*14+[.0001];qs=[.8]+[.7]*14+[.81]
 ww=[p**.5*math.exp((q-max(qs))/.05) for p,q in zip(priors,qs)];qq=[w/sum(ww) for w in ww]
 ep=sum(p*q for p,q in zip(priors,qs));eq=sum(p*q for p,q in zip(qq,qs))
 assert max(qs)>qs[0] and eq<ep
 # Execute only the exact repository helper via AST, not its module/main/import side effects.
 import numpy as np
 tree=ast.parse(sources['code/rl2_ppo_train.py']);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='game_rewards')
 ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'game_rewards_from_fixed_ref','exec'),ns)
 defaults=SimpleNamespace(reward='terminal',c_dep=0.,c_lost=0.,shape_iters=200,rscale=1/5000,win_bonus=.1)
 def reward_for_margin(m):
  z=np.zeros(299);dm=z.copy();dm[-1]=m
  g={'dmargin':dm,'dworth':dm,'dep':z,'lost':z,'margin':m}
  return float(ns['game_rewards'](g,defaults,0).sum())
 risky=[(.8,-100),(.2,10000)]
 reward_safe=reward_for_margin(100);reward_risky=sum(p*reward_for_margin(m) for p,m in risky)
 u_safe=utility(100);u_risky=sum(p*utility(m) for p,m in risky)
 assert reward_risky>reward_safe and u_risky<u_safe
 z=np.zeros(299);dw=z.copy();dw[0]=1000;dw[-1]=-1000
 g={'dmargin':z,'dworth':dw,'dep':z,'lost':z,'margin':0};defaults.reward='worth'
 r=ns['game_rewards'](g,defaults,0);discounted=float(np.dot(r,.995**np.arange(len(r))))
 assert abs(float(r.sum()))<1e-6 and discounted>0
 checks={
  'M01_mean_loss': '平方损失（MSE，未裁标签）' in arch,
  'M01_wdl_not_mean_sign':'分差条件均值头的符号**不得**覆盖概率结论' in arch,
  'M02_utility_main':'主价值改由 E[U] 主头' in arch,
  'M03_m6_dev':'**开发验证对手（明确名单：M6 r1）**' in arch,
  'M03_threshold': 'ΔU 点估计 ≥ +0.02 且配对 95% CI 下界 > 0' in arch,
  'D01_info_set_stats':'信息历史相同的节点必须共享决策统计' in arch,
  'D02_full_joint':'**不在 A 内再归一化**' in arch,
  'D04_truth_mask':'truncated ≠ terminal' in arch and 'terminal 标签占比 ≥ 70%' in arch,
  'D06_cost_corrected':'37.75%' in arch,
  'D07_profile_diagnostic':'不作 Stage 1 验收条件、不单独判过/判否、不触发 kill' in arch,
  'remaining_MAE_kill_plan':'价值 MAE 两检查点不降 → kill' in plan,
  'remaining_MAE_kill_arch':'价值留出 MAE 连续两检查点不降' in arch,
 }
 result={'review_ref':a.ref,'scope':'Read-only static checks and mathematical/reward-helper probes; no training or game-strength measurement.',
 'source_sha256':{f:hashlib.sha256(s.encode()).hexdigest() for f,s in sources.items()},'document_contract_checks':checks,
 'mean_vs_MAE_stop_rule':{'normal_margin_for_U_point1':1000*math.log(.2/.8),'normal_margin_for_U_point9':1000*math.log(.8/.2),'distribution':[{'p':p,'U':y} for p,y in dist],'true_conditional_mean':target_mean,'checkpoints':mae_rows,'original_kill_triggered':True},
 'teacher_soft_target_validation':{'prior':priors,'exact_Q':qs,'q':qq,'bare_greedy_Q':qs[0],'teacher_best_Q':max(qs),'prior_expected_U':ep,'q_expected_U':eq,'selected_action_gain':max(qs)-qs[0],'q_minus_prior_expected_U':eq-ep,'interpretation':'A gate validating only argmax-Q action does not validate q. Not a proved defect if actual gate samples the full q/behavior distribution.'},
 'current_reward_actual_helper':{'parameters':'reward=terminal (supported controlled mode); rscale=1/5000 and win_bonus=.1 from defaults; no aux shaping; game_rewards aggregates dmargin and assigns reward to final step','safe_margin':100,'risky_distribution':[{'p':p,'margin':m}for p,m in risky],'safe_reward':reward_safe,'risky_expected_reward':reward_risky,'safe_platform_U':u_safe,'risky_platform_U':u_risky,'objective_rank_reversal':True,'not_claiming_current_remote_run_uses_these_defaults':True},
 'current_worth_discount_actual_helper':{'gamma':.995,'steps':299,'rscale':1/5000,'transient_worth':1000,'final_margin':0,'undiscounted_reward_sum':float(r.sum()),'discounted_reward_sum':discounted,'interpretation':'A transient potential produces positive discounted reward despite unchanged terminal cash.'},
 'operational_threshold_not_significance':{'mean_gain':90,'SE':1,'effect_threshold':100,'z_like_ratio':90,'fails_effect_threshold':True,'still_has_high_precision_effect':True},
 'heldout_identity_lifetime':{'first_confirmation':'opponent H unseen by training/development; independent confirm valid','after_result_is_used_for_next_generation':'same H is no longer an unseen-opponent test, even with a new map seed block','required_manifest_fields':['candidate_sha','opponent_ids_and_hashes','seed_blocks','claim_type','consumed_by_confirmation','selection_inputs','CI_cluster_unit']}}
 a.out.mkdir(parents=True,exist_ok=True);(a.out/'evidence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'output':str(a.out/'evidence.json'),'checks':checks,'mae':mae_rows,'q_expected_U':eq,'prior_expected_U':ep,'reward_reversal':[reward_safe,reward_risky,u_safe,u_risky],'worth_discount':discounted},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
