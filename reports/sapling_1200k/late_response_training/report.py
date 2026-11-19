"""Export six map-clustered training examples and describe actual, individually verified cash transfers."""
from collections import Counter
import gzip
import json
from pathlib import Path
from statistics import mean

HERE=Path(__file__).resolve().parent
FNAMES={"own_return":"载货返库","deny_enemy_return":"堵敌返库"}
def read(p):return json.loads(p.read_bytes())


def examples(summary):
    data=[]
    for v in summary:
        r=v["root_features"];parts=[]
        for ref in v["evidence"]:
            trace=json.loads(gzip.decompress((HERE/ref).read_bytes()))
            a,b=(trace["arms"][u] for u in ("parent","alternative"))
            target_move=None
            if r.get("enemy_worker"):
                dr,dc={"UP":(-1,0),"DOWN":(1,0),"LEFT":(0,-1),"RIGHT":(0,1),"NONE":(0,0)}[a["path"][0]["enemy_target_order"] or "NONE"]
                pos=r["enemy_origin"];size=15
                target_move=((pos//size+dr)%size)*size+(pos%size+dc)%size
            parts.append(dict(stream=trace["stream"],actual_opponent_root_destination=target_move,
                              actual_opponent_used_blocked_square=(target_move==r["target_cell"] if target_move is not None else None),
                              gross_bank_base_alt=[x["gross"][r["seat"]]["bank"] for x in (a,b)],
                              enemy_gross_bank_base_alt=[x["gross"][1-r["seat"]]["bank"] for x in (a,b)],
                              focal_own_cash_base_alt=[x["focal_own_deposits"] for x in (a,b)],
                              focal_enemy_cash_base_alt=[x["focal_enemy_deposits"] for x in (a,b)],
                              score_margin_base_alt=[x["margin"] for x in (a,b)],wdl_base_alt=[x["wdl"] for x in (a,b)],
                              delta_u=trace["delta_u"],delta_margin=trace["delta_margin"],cash_margin_by_category=trace["delta_by_source"],
                              verifying_evidence=ref))
        is_deny=v["family"]=="deny_enemy_return"
        data.append(dict(identity={"map":v["seed"],"seat":v["seat"],"turn":v["t"],"unit":v["worker"],"opponent_unit":r.get("enemy_worker")},
                         split=v["split"],family=v["family"],from_action=r["old"],to_action=r["alternative"],
                         public_features=dict(remaining=r["remaining"],own_cargo=r["cargo"],own_cash=r["own_cash"],opponent_cash=r["enemy_cash"],
                                              own_home_distance=r["own_home_distance"],original_home_distance=r["original_home_distance"],
                                              alternative_home_distance=r["alternative_home_distance"],
                                              enemy_target_present=is_deny,enemy_target_cargo=r.get("enemy_cargo"),
                                              enemy_target_home_distance=r.get("enemy_home_distance"),
                                              enemy_target_shortest_choices=len(r["enemy_shortest_cells"]) if is_deny else None),
                         targets=dict(mean_delta_u=v["mean_delta_u"],mean_margin=v["mean_margin"]),paired_observations=parts))
    return data


def main():
    progress=read(HERE/"progress.json");registry=read(HERE/"roots.json")
    if not progress["complete"]:raise ValueError("unverified pairs remain")
    rows=[]
    for split in ("train","holdout"):
        vals=examples(read(HERE/f"{split}_labels.json"))
        payload="".join(json.dumps(x,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n" for x in vals).encode()
        (HERE/f"{split}_examples.jsonl.gz").write_bytes(gzip.compress(payload,mtime=0));rows+=vals
    a={r["identity"]["map"] for r in rows if r["split"]=="train"}
    b={r["identity"]["map"] for r in rows if r["split"]=="holdout"}
    if a & b or len(a)!=2 or len(b)!=1:raise ValueError("split identity")
    total=Counter();lines=[];results=[];tagged=[]
    for r in rows:
        who=r["identity"];parts=r["paired_observations"];seed=who["map"];f=FNAMES[r["family"]];key=r["family"];direction=[]
        for v in parts:
            total["search_arms"]+=2
            for w in v["wdl_base_alt"]:total[w]+=1
            d=v["delta_margin"];total["pos" if d>0 else "neg" if d<0 else "equal"]+=1
            direction.append(f"{d:+.2f}");results.append({**who,"family":key,"split":r["split"],**v})
        own=[x["focal_own_cash_base_alt"] for x in parts];foe=[x["focal_enemy_cash_base_alt"] for x in parts]
        effect=lambda pair:sum(x[1]-x[0] for x in pair)/len(pair)
        tagged.append((f,seed,effect(own),effect(foe) if foe[0][0] is not None else None))
        qty=own if key=="own_return" else foe
        use="；".join(f"{x[0]:.1f}→{x[1]:.1f}" for x in qty)
        label="训练" if r["split"]=="train" else "留出"
        lines.append(f"| {seed}／{who['seat']}（{label}） | {f} t{who['turn']}，{r['from_action']}→{r['to_action']} | {'／'.join(direction)} | {r['targets']['mean_delta_u']:+.5f} | {use} |")
    out=["# 终局前 5–12 步，返库与堵截的真实响应标签","",
         f"三张已曝光开发图各取两个不同的合法动作根，每根给原版 Master3s2 两次独立采样。共 {len(rows)} 根、{len(results)} 对、{total['search_arms']} 臂。与亲本原动作相比，{total['pos']} 对终局分差提高、{total['neg']} 对降低、{total['equal']} 对持平。这些分支的 Sapling 终局计 {total['W']} 胜、{total['D']} 平、{total['L']} 负；它们共享历史前缀，不能当作从开局抽取的独立正式考试。每种替代动作只在它自己的真实地图、席位和同一对随机流里与亲本比较。分母按地图或根报告时，两次采样必须合并。[选图与选根协议](protocol.json) · [所有事前合格机会的分母与选择](roots.json)","",
         "| 地图／Sapling 席 | 动作 | 两次配对 Δ 分差 | 平均 ΔU | 目标工实际存入：原→改，两次 |","| --- | --- | ---: | ---: | --- |",*lines,"",
         "返库行最后一列追踪本方所改工人，堵截行追踪入选的敌方工人；这直接区分所改动作有没有影响目标现金。整体终局分差还包括两边其他工人的交付、树收入、碰撞、招工和种树成本。独立账本逐拍对齐原版解释器，并检查各项双方现金差变化之和等于正式分差变化。[逐对分项、终局和动作路径](arms/)",""]
    # Describe the strongest directional contradiction without discarding any rows.
    return_failed=[r for r in results if r['family']=='own_return' and r['focal_own_cash_base_alt'][1]>r['focal_own_cash_base_alt'][0] and r['delta_margin']<0]
    denial=[r for r in results if r['family']=='deny_enemy_return']
    real=sum(r["focal_enemy_cash_base_alt"][1]<r["focal_enemy_cash_base_alt"][0] for r in denial)
    contacted=sum(r["actual_opponent_used_blocked_square"] for r in denial)
    out.append(f"6 个堵截配对里，同时下令的敌工实际走向被堵方格的有 {contacted} 个，目标敌工在替代臂实际少存入的有 {real} 个。每根的两个配对共用同一历史同时根指令。对手另一个等短方向、对手本来决定采集或改道、我方离开原任务，都会影响结果；原始档给出敌方是否按几何预期落格、此后每一拍位置和实际交款。")
    if return_failed:
        ex=sorted(return_failed,key=lambda r:r["delta_margin"])[0];p,q=ex["focal_own_cash_base_alt"]
        out.append(f"一个本方多交付、整局却更差的例子是地图 {ex['map']} t{ex['turn']} 的采样 {ex['stream']+1}：该工从 {p:.2f} 增加到 {q:.2f}，终局分差反而改变 {ex['delta_margin']:+.2f}。")
    out+=["","## 两条返航路径为何不同","",
          "训练图 620224310，席 0，t289：工人 47 载 115.29、距库两格。第一条随机流中，原地多采后到 t299 才交 155.11；改为向右后 t291 先交 115.29，又在 t295 交 105.03。这次实际多一次交付。[逐笔现金与位置](arms/620224310/own_return/0/verified.json.gz)","",
          "留出图 620224311，席 1，t290：工人 34 载 227.95、距库三格。第一条随机流里，向上走近一步的路线首拍从 227.95 掉到 205.15，经过格 68，紧邻当时格 54 的敌树；原令向下到 98 没有这笔损失。两臂最终均在 t296 交付，分别交了 152.11／175.83。走近一格没有节省最终送达时间，还增加了途中被抽取的货。这个看完留出后的解释可以在新地图检验，不能再把该地图当作新路线特征的盲测。[逐笔现金与位置](arms/620224311/own_return/0/verified.json.gz)","",
          "## 训练与留出","",
          "620224310 与 621454920 是本次动作标签的训练侧，共两张图、四个不同根，620224311 单列留出，两根。三张均为历史开发图，留出仅对本次标签的拟合与调参隔离。数据将 11 项当时可见特征与终局目标分开保存；地图、时间、工人身份只作归档和分组。每个例子的两个收益观测共享同一地图和根。训练侧地图和动作根不足以验证权重更新，本批未拟合网络、未更换候选、未开启正式考试。[训练示例](train_examples.jsonl.gz) · [留出示例](holdout_examples.jsonl.gz)","",
          "原版考官使用未改搜索与温度 1。根上保留历史实际同时动作，从下一步起双方都读取本臂的新观测。母局没有保存原始采样器与墙钟估计；每一对重新抽一份随机流并共用暖启成本估计。各配对动作带都经官方引擎从开局逐步重放至终局。重启下列入口会核验已完成成果，并接续尚缺的臂。","",
          "```sh","export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CARBON_DEBUG=1","d=reports/sapling_1200k/late_response_training",'python "$d/study.py" --cpus 1 2','python "$d/report.py"',"```","",
          "[逐对核分与收支复算代码](study.py) · [最终进度](progress.json)"]
    (HERE/"README.md").write_text("\n".join(out)+"\n")


if __name__=="__main__":main()
