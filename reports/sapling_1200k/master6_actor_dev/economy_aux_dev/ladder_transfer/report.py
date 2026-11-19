"""Check the future targets against the separately replayed treasury records and report game-level results."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
TOP=HERE.parents[2]/"all_top5"


def read(path):
    raw=path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix==".gz" else raw)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_labels(cards,r):
    records={};paths={}
    for batch in sorted({g["batch"] for g in cards}):
        p=TOP/batch/"turn_cash.json.gz"; paths[batch]=digest(p); body=read(p)
        if body["components"][0]!="bank" or body["components"][2]!="tree":
            raise ValueError("unknown cash extraction schema")
        records.update({g["id"]:g for g in body["games"]})
    by_target={n:[] for n in r["income_targets"]}
    z=HERE/"predictions.npz"
    if r["predictions_sha256"]!=digest(z):
        raise ValueError("changed predictions")
    with np.load(z) as data:
        for g in cards:
            record=records[g["id"]]
            if len(record["turns"])!=g["turns"]:
                raise ValueError("incomplete ledger")
            flow=np.asarray([t[2] for t in record["turns"]],dtype=np.float64)
            flow=flow[:,[g["seat"],1-g["seat"]]][:,:,[0,2]].reshape(-1,4)
            cum=np.vstack((np.zeros((1,4)),flow.cumsum(axis=0)))
            ix=np.flatnonzero(data["gid"]==g["gid"]); t=data["step"][ix].astype(int)
            for h,offset in ((10,0),(30,4)):
                use=t+h<=g["turns"]
                for col in range(4):
                    if not np.array_equal(data["valid_horizon_targets"][ix,offset+col],use):
                        raise ValueError("masked future horizon differs")
                    expected=cum[t[use]+h,col]-cum[t[use],col]
                    observed=np.expm1(data["target_log1p"][ix[use],offset+col].astype(np.float64))
                    by_target[r["income_targets"][offset+col]].extend(np.abs(observed-expected).tolist())
    summary={"published_ledger_sha256":paths,"per_cash_target":{"%s"%k:{"roots":len(v),"max_abs_cash":max(v),
              "mean_abs_cash":float(np.mean(v)),"above_one_cent":sum(e>.01 for e in v)} for k,v in by_target.items()}}
    (HERE/"income_labels_audit.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def main():
    r=read(HERE/"result.json");source=read(HERE/"selection.json");cards=source["cards"]
    verified=verify_labels(cards,r)
    bad=sum(v["above_one_cent"] for v in verified["per_cash_target"].values())
    lines=["# Sapling 在两批新天梯图上的公开动作与收入预测","",
           f"累计 {len(cards)} 张已完成且有双方数值比分的 Master6 地图，Master6 取得 {r['all_counts'].get('W',0)} 胜、{r['all_counts'].get('D',0)} 平、{r['all_counts'].get('L',0)} 负。"
           "按比赛 ID、种子和实际初盘核对早期 1,475 局、原 Sapling 分组及已登记训练和开发场。另有一场对手执行出错、比分缺失，原始记录单列。"
           "两批回放已用于通用天梯复盘；本次拿已冻结的模型检验迁移，全程不更新参数。"
           "[赛前选取口径](protocol.json) · [全体清单、报名名及隔离证据](selection.json)",
           "",
           "每席从四个步段各抽最多 8 根，在原网络里逐工依次提供 Master6 已提交的更早槽位动作。未知碳与赛后比分只作独立的监督字段，已验证移除后动作与收入编码不变；首批根还逐槽验证当前及后续标签改变不影响当前输出。"
           "策略用量化后的三张网，两个训练中保留的收入头沿用原 FP32 编码。父代没有这八个预测输出。"
           "[冻结权重与逐图分数](result.json) · [逐根概率、收入和有效掩码](predictions.npz)","",
           "| 分组 | 图 / 根 / 有选择的动作 | 动作 NLL，父 / 对照 / 候选 | 动作命中率，父 / 对照 / 候选 | 收入误差，对照 / 候选 | 候选收入误差较小地图 |",
           "|---|---:|---:|---:|---:|---:|"]
    specs=(("all","全部"),("seat_0","Master6 席 0"),("seat_1","Master6 席 1"),("seen_opponent","训练见过同名对手"),
           ("unseen_opponent","训练未见该报名名"),("W","Master6 胜局"),("D","Master6 平局"),("L","Master6 败局"))
    for key,label in specs:
        if key not in r["panels"]:
            continue
        a=r["panels"][key];e=a["equal_map"]
        fmt=lambda d:" / ".join(f"{d[arm]:.4f}" for arm in ("parent","control","candidate"))
        hit=" / ".join(f"{100*e['accuracy'][arm]:.2f}%" for arm in ("parent","control","candidate"))
        incomes=" / ".join(f"{e['income'][arm]:.4f}" for arm in ("control","candidate"))
        lines.append(f"| {label} | {a['maps']} / {a['roots']} / {a['actor']['candidate']['actions']} | {fmt(e['nll'])} | {hit} | {incomes} | "
                     f"{e['counts_lower_than_control']['income']}/{a['maps']} |")
    lines.extend(["","表内每图等权。动作 NLL 和收入 Smooth L1 越低越好；后者先对实际现金做 log1p，并使用训练集原有的中心和尺度。"
                  "下面各通道等权看每个有效根，把预测还原成现金后取绝对误差。任一根未来剩余步数不足该时限即屏蔽。按原报名名判断是否见过，只说明字符串是否在 216 张微调训练图出现。","",
                  "| 从发令前向后 | 主体与收入 | 有效根 | 真实平均到账 | 预测平均，对照 / 候选 | 现金 MAE，对照 / 候选 |",
                  "|---|---|---:|---:|---:|---:|"])
    total=r["panels"]["all"]["income"]
    for name in r["income_targets"]:
        a=total["control"]["by_channel"][name];b=total["candidate"]["by_channel"][name]
        var,art,window=name.split("_");who="我方" if var=="own" else "对手";mode="回库" if art=="delivery" else "树息"
        lines.append(f"| {window} 步 | {who}{mode} | {b['valid_roots']} | {b['actual_mean_cash']:.2f} | "
                     f"{a['predicted_mean_cash']:.2f} / {b['predicted_mean_cash']:.2f} | "
                     f"{a['mae_cash']:.2f} / {b['mae_cash']:.2f} |")
    lines+=["",f"标准化误差可提升而原单位绝对误差未必同步。独立用两批原版资金流水全量重建目标，高于 0.01 现金的标签对差异 {bad} 个。"
            "[独立目标复核](income_labels_audit.json)",""]
    for g in cards:
        if g["wdl"]=="L":
            q=r["per_game"][g["id"]];inc=q["income"]
            lines.append(f"唯一败局是第 {g['seat']} 席对 {g['opponent']}，Master6 官方分差 {g['margin']:+,.2f}。在 {q['roots']} 个根，"
                         f"候选的前缀动作命中率 {q['actor']['candidate']['agreement']:.2%}，对照为 {q['actor']['control']['agreement']:.2%}；"
                         f"标准化收入误差从 {inc['control']['normalized_huber']:.4f} 变为 {inc['candidate']['normalized_huber']:.4f}。"
                         "一局无法代表落后时的总体表现。")
    lines+=["","这里的策略动作来自专家走出的局面，真实回款也来自专家继续下完的路线。可据此检验观测与表征的迁移；要计 Sapling 对原版 Master3s2 的胜率与终局分差，仍需让冻结策略自己出令并承担对手的响应。"]
    diag=read(HERE/"rival_freight.json")
    g=diag["groups"]["three_or_more"]; w=diag["largest_absolute_error"]
    lines.extend(["","## 对手的运货能力何时被高估","",
                  f"按发令前可见的敌采集工数量作事后拆分。对手至少有三名采集工的 {g['roots']} 个根，随后 30 步平均入库 {g['realized']:.1f}，"
                  f"对照预报 {g['control']['mean']:.1f}，候选预报 {g['candidate']['mean']:.1f}。"
                  f"最大的单根偏差出现在对 {w['opponent']} 的第 {w['step']} 步。对手当时有 {len(w['rival_collectors_before'])} 名采集工，"
                  f"随身共 {w['rival_cargo_before']:.2f}，随后 30 步实收 {w['actual_30_steps']:.2f}，候选预报 {w['candidate']:.2f}。"
                  "如果用这个收入读数决定是否动用人手拦截，高产能对手会被误判为正在兑现巨款。"
                  "[冻结预测的按局面拆分](rival_freight.json)"])
    (HERE/"README.md").write_text("\n".join(lines)+"\n")


if __name__=="__main__":
    main()
