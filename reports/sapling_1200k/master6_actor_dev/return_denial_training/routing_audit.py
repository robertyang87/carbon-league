"""After results are frozen, distinguish choosing a wrong exit from no target taking an interceptable shortest exit."""
from collections import Counter, deque
from forecast import HERE, MOVES, bind, extract, public_turn, restore, select
from policy import _commitments, _options


def audit():
    model=extract.load(HERE/"weights.json"); rows=[]; groups={}
    for g in bind("validation")["tapes"]:
        raw,cfg,sim=restore(g);s=g["seat"];h=deque(maxlen=10); counts=Counter()
        for t,joint in enumerate(raw["actions"]):
            obs=sim.obs(s)
            carriers=public_turn(obs,list(h),joint[s],cfg)
            choice=select(carriers,model,cfg)["forecast"]
            if choice:
                selected,o,gain=choice
                goal=selected["target"]
                own,enemy=obs["players"][s],obs["players"][1-s]
                movers,commit=_commitments(own,enemy,joint[s],cfg)
                pre={}
                for c in carriers:
                    actual_options={opt["cell"] for opt in c["options"]}
                    loads=[c["cargo"]*(1-cfg["moveCost"]) if i else c["cargo"] for i in range(5)]
                    broad=_options(c["action_cells"],loads,[True]*5,movers,commit,cfg["moveCost"])
                    all_cells={opt["cell"] for opt in broad}
                    pre[c["target"]]=(set(actual_options),all_cells)
                # All feasible endpoints and the learned decision have now been fixed independently of enemy action.
                real={c["target"]:c["action_cells"][MOVES.index(joint[1-s].get(c["target"]) or "NONE")] for c in carriers}
                chosen_actual=real[goal]
                target_any=chosen_actual in pre[goal][0]
                some=any(real[uid] in locs[0] for uid,locs in pre.items())
                broad_any=any(real[uid] in locs[1] for uid,locs in pre.items())
                eligible_root_broader=chosen_actual in pre[goal][1]
                target_move=(joint[1-s].get(goal) or "NONE")
                physical=MOVES.index(target_move)
                isshort=selected["shortest"][physical]
                counts["roots"]+=1
                counts["observed_selected_exit"]+=chosen_actual==o["cell"]
                counts["selected_target_any_reachable_shortest"]+=target_any
                counts["any_eligible_target_any_reachable_shortest"]+=some
                counts["target_walked_any_shortest"]+=isshort
                counts["selected_target_any_nonshortest_reachable"]+=eligible_root_broader and not target_any
                counts["any_eligible_target_any_reachable_including_off_shortest"]+=broad_any
                idx=selected["action_cells"].index(o["cell"]); counts["predicted_target_landings_at_selected_exit"]+=selected["p"][idx]
                counts["original_target_motion_"+target_move]+=1
                rows.append(dict(seed=g["seed"],seat=s,t=t,target=goal,chosen_exit=o["cell"],observed_target=chosen_actual,
                                 available_shortest=sorted(pre[goal][0]),available_any=sorted(pre[goal][1]),
                                 target_walked_shortest=isshort,any_target_observed_on_reachable_shortest=some,
                                 any_target_observed_on_any_reachable=broad_any,model_p=selected["p"][idx]))
            sim.step(*joint);h.append(obs)
        groups[f"{g['seed']}/{s}"]=dict(counts)
    allc=Counter()
    for z in groups.values():allc.update(z)
    doc=dict(weights_sha256=extract.sha(HERE/"weights.json"),results_sha256=extract.sha(HERE/"results.json"),
             scope="Post hoc availability diagnostics only. Actual opposing commands never change the frozen choices. Physical co-location is not proof of net transfer; neither counterfactual profits nor opponent adaptation have been simulated for these hindsight alternatives.",
             total=dict(allc),per_map_seat=groups,rows=rows)
    extract.write(HERE/"routing_audit.json",doc)
    lines = ["# 当拍实际经过的路线能否被轻工占住", "",
        f"在冻结策略实际改令的 {allc['roots']} 个验证回合里，目标进入所选出口 {allc['observed_selected_exit']} 次。事后遍历同拍全部合格返航重货、所有本方采集工和所有满足注册条件的改令，可由其中任何一个新动作占到敌方真实最短落点的回合仍只有 {allc['any_eligible_target_any_reachable_shortest']} 个。", "",
        f"候选选中的重货有 {allc['target_walked_any_shortest']} 次走了某个最短出口。将可尝试格放宽至其五种出手的所有落点，这 {allc['roots']} 拍中仍有 {allc['any_eligible_target_any_reachable_including_off_shortest']} 拍同时满足：我方较轻、能以一个不同于 v1 的合法动作抵达、不和队友冲突，且敌人确实进入该格。其他返航但没有最短拦截机会的回合不在此项事后枚举内。", "",
        f"学习头给所选格的命中概率求和为 {allc['predicted_target_landings_at_selected_exit']:.2f}，实际命中是 {allc['observed_selected_exit']}。这揭示了一个具体偏差：即使五种落点整体的概率误差低，在专挑我方轻工已能改令威胁的位置时，预测仍高估了敌人会走进去的频率。对手是否有意预判我方反应，单靠已执行的命令不能判断。", "",
        "这是冻结结果后的可达性诊断。没有用对手同时动作重新挑正式候选，也没有把走近或封路误记为打劫得手。要检验提前数拍布局能否迫使绕行、延迟回款或制造后续可抢窗口，须另行登记更早的位置选择和双方真实续弈。[逐工时点与机会集](routing_audit.json) · [冻结一步检验](README.md)"]
    (HERE/"ROUTING_AUDIT.md").write_text("\n".join(lines)+"\n")
    print(dict(allc),flush=True)


if __name__=="__main__": audit()
