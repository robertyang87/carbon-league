"""Publish compact action-pair targets and summarize the full-economy evidence."""
import gzip
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
def read(path):
    data=path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix==".gz" else data)

def main():
    rows=[];text=["## 两根续弈到终局的结果","",
        "| 训练图、动作 | 原令分差 | 改令分差 | 改令带来的分差变化 | 原令 U／改令 U |",
        "|---|---:|---:|---:|---:|"]
    for root in read(HERE/"roots.json")["selected"]:
        name=root["family"];wd=HERE/name;verified=read(wd/"verified.json.gz");s=root["seat"];o=1-s
        a,b=[verified["arms"][k] for k in ("parent","alternative")]
        dm=verified["delta_margin"];du=verified["delta_u"]
        text.append(f"| {root['seed']} 席 {s}，t{root['t']}，{root['old']} → {root['alternative']} | {a['margin']:+,.2f} | {b['margin']:+,.2f} | {dm:+,.2f} | {a['u']:.6f}／{b['u']:.6f} |")
        reps={arm:read(wd/f"live_{arm}.json") for arm in ("parent","alternative")}
        assert all(v["terminal_t"]==root["t"]+len(part["path"]) for v,part in zip(reps.values(),(a,b)))
        assert reps["parent"]["root_opponent_observation_sha256"]==reps["alternative"]["root_opponent_observation_sha256"]
        targets={arm:{"action":root["old" if arm=="parent" else "alternative"],
                      "terminal_scores_by_seat":v["score_by_physical_seat"],"terminal_margin":v["margin"],"utility":v["u"],
                      "focal_tree_income_by_seat":v["focal_tree_income"],"cumulative_cash_sources_by_seat":v["gross"],
                      "full_opponent_calls_after_root":reps[arm]["search_calls"],"completed_searches":reps[arm]["search_completed"]}
                 for arm,v in (("parent",a),("alternative",b))}
        context=wd/"root_input.json.gz";assert hashlib.sha256(context.read_bytes()).hexdigest()==verified["root_input_sha256"]
        rows.append({"split":"previously exposed development data, training only","map_seed":root["seed"],"perspective":s,"decision_t":root["t"],
                     "input_file":str(context.relative_to(HERE)),"input_sha256":verified["root_input_sha256"],
                     "target_tree":root["tree"],"worker":root["worker"],"one_stream_per_root":True,
                     "shared_start_sha256":verified["shared_init_sha256"],"outcomes":targets,
                     "alternative_minus_parent":{"terminal_margin":dm,"utility":du,"own_minus_enemy_cash_sources":verified["cash_delta_by_channel"]},
                     "verification_file":str((wd/"verified.json.gz").relative_to(HERE))})
    text+=["","现金归因是两个分支间“己方变化减去敌方变化”，以下拆分加总回终局分差变化。目标树的实际入账按其当步所有者追踪至消失。"]
    for r in rows:
        s=r["perspective"];o=1-s;a,b=r["outcomes"].values();v=r["alternative_minus_parent"]["own_minus_enemy_cash_sources"]
        channels=(("bank","正常交付"),("donation","对手交入"),("tree","树息"),("collision_cash","碰撞直入现金"),("recruit","招工支出"),("plant","种树支出"),("seize","夺树支出"),("round","取整"))
        sums='，'.join(f"{label} {v[k]:+,.2f}" for k,label in channels)
        at=a["focal_tree_income_by_seat"];bt=b["focal_tree_income_by_seat"]
        text+=["",f"- 图 {r['map_seed']}。{sums}。原令下目标树给己／敌收入 {at[s]:.2f}／{at[o]:.2f}，改令后 {bt[s]:.2f}／{bt[o]:.2f}。"]
    text+=["","每根只有一对终局，没有并行随机流估计，后续所有单位的决策随分支改变。训练可以保留这两条带符号的响应样本与目标树局部实际收入；仅凭它们不能确定平均动作价值。两根是旧双网亲本局面，新训练的 Master6 示范 actor 未在本试验中出场。","",
           "[可消费的两条动作对目标](action_pair_labels.json) · [夺树逐步账](seize_enemy_tree/verified.json.gz) · [守树逐步账](hold_own_tree/verified.json.gz)",""]
    (HERE/"action_pair_labels.json").write_text(json.dumps(rows,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    (HERE/"results.md").write_text("\n".join(text))

if __name__=="__main__":main()
