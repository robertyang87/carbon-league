"""Summarize the predeclared intervention and numerical cash differences with physical seat labels."""
from run import HERE, R


def report():
    result=R.load(HERE/"results.json")
    output=["# 只学种植工移动方向的学生，对原版考官的连续 16 步复核","",
        f"冻结学生 `{result['binding']['student'][:12]}` 在两张此前未进入该网络训练、校准、留出评估或教师窗口的旧地图，从相同的第 128 步局面接管到第 143 步。随后统一交还首胜包 `8da4b465`，考官从第 128 步起每一拍都以完整原版重新应对。","",
        "| 图、我方席 | 父代终分（我／敌） | 学生窗口终分（我／敌） | 官方分差改善 | 同局面实际改令步数／种植工方向条数 |",
        "|---|---:|---:|---:|---:|"]
    details=[]
    imbalances=[]
    for row in result["rows"]:
        s=row["seat"]
        p,q=[row["arms"][a]["score"] for a in ("parent","student")]
        changes=row["arms"]["student"]["student_parent_disagreements_on_this_trajectory"]
        turns=sum(len([d for d in z["directions"] if d["role"]=="PLANTER" and d["student"]!="NONE" and d["parent"]!="NONE"]) for z in changes)
        output.append(f"| {row['seed']}，席 {s} | {p[s]:.2f}／{p[1-s]:.2f} | {q[s]:.2f}／{q[1-s]:.2f} | {row['official_delta']:+.2f} | {len(changes)}／{turns} |")
        v=row["net_component_deltas"]
        first=row["first_executed_intervention"]
        rival=row["first_between_arm_opponent_difference"]
        marker=("考官两臂的敌令差异早于本次学生实改，其后现金差包含墙钟所致轨迹差异。" if rival is not None and (first is None or rival<first)
             else "原版在同一拍已产生不同敌令；该拍为同时行动，不能归因为对方看见了学生新令。" if first==rival and first is not None
             else "双方顺序见逐拍记录。")
        zeros={k:row["arms"][k]["clock"]["return_types"].get("depth_zero_return_base",0) for k in ("parent","student")}
        if zeros["student"]>0 or zeros["parent"]>0:
            imbalances.append(f"图 {row['seed']} 原版考官在父代臂零深回退 {zeros['parent']} 次，学生臂零深回退 {zeros['student']} 次。该场正式计分有效。两条轨迹上考官实际搜索强度悬殊，单凭终差不能把收益归给学生。")
        count=lambda k:row["arms"][k]["original_calls"]
        w=lambda k:row["arms"][k]["original_wall"]
        details.append(f"图 {row['seed']} 的学生窗口首次实改在 {first}，双臂的本方首次异令在 {row['first_between_arm_own_difference']}，考官首次异令在 {rival}。{marker} 到第 199 步的地面树租净差变化 {v['through_199'].get('ground_rent',0):+.2f}，至终局地租 {v['whole'].get('ground_rent',0):+.2f}、常规回款 {v['whole'].get('own_delivery',0):+.2f}，均按本方减对手，再取学生减父代。两臂考官公开入口 {count('parent')}／{count('student')} 次，累计墙钟 {w('parent'):.2f}／{w('student'):.2f} 秒。")
    output.extend(["",*imbalances,"",details[0],"",details[1],"","每图各有一条父代和一条学生后缀，原版自行决定温度 1 的采样与 0.58 秒搜索深度。所有结果包括未改令和失败的局面；这一试验没有新增从开局打到终局的胜场。两席逐手日志保留 CPU、实时时钟、候选数、深度和回退等数据。","",
        "[预登记](protocol.json) · [部署动作独立复核](public_changes.json) · [原版逐席计分与现金分解](results.json) · [执行器](run.py)"])
    (HERE/"README.md").write_text("\n".join(output)+"\n")


if __name__=="__main__":
    report()
