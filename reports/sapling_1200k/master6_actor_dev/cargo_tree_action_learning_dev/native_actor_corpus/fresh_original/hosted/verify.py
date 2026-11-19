"""Verify all four original hosted artifacts against their registered matrix and GitHub digest list."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from tempfile import TemporaryDirectory
import zipfile

sys.dont_write_bytecode = True
FULL = "6c54899562382552b95bc1299d83119088ef1d1206b93135e33769815ed69c9f"
PARENT = "8da4b465f7fa6e601e51bd7c2e81f25825cb59b9b5ab285f7f56d0f6bbfe1cac"
ORIGINAL = "65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"
ARMS = ("parent", "student")
ROLES = {"parent":"父代", "student":"全骨干"}
FIELD = {"own_delivery":"回款","ground_rent":"地租","cargo_tax":"货税","plant":"栽树","seize":"夺树",
         "hire_planter":"招种植工","hire_collector":"招采集工","tree_collision_cash":"树格碰撞",
         "base_collision_cash":"基地碰撞","opponent_worker_delivery_to_this_base":"敌方存入","rounding":"舍入"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_zip(path, seed, seat, matrix, run, S, temp):
    directory = Path(temp) / f"{seed}-{seat}"
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        require(len(members)==len(set(members)) and archive.testzip() is None, "damaged or repeated ZIP entries")
        github_pair = json.loads(archive.read("verification.json"))
        require((github_pair["seed"], github_pair["seat"]) == (seed,seat), "hosted pair identity")
        require(not github_pair["verification_errors"], "hosted replayer failed")
        games = []
        for arm in ARMS:
            target = directory/arm
            target.mkdir(parents=True)
            for member in ("receipt.json","registration.json","attempt.json","turns.jsonl"):
                name=f"{arm}/{member}"
                if name in members:
                    (target/member).write_bytes(archive.read(name))
            verified = S.verify_game(target,seed,seat,arm,matrix)
            require(json.loads(json.dumps(verified)) == github_pair["games"][arm], "independent official replay differs from hosted replay")
            require(str(verified["workflow_run"]) == str(run), "different run identity")
            registration = S.read(target/"registration.json")
            receipt = S.read(target/"receipt.json")
            if (target/"attempt.json").is_file():
                attempt = S.read(target/"attempt.json")
                require(attempt == {"seed":seed,"runtime_image":receipt["image_id"],"protocol_sha256":digest(target/"registration.json")}, "attempt ticket mismatch")
            if receipt.get("probes"):
                require(len({c["id"] for c in receipt["containers"]}) == 2, "both seats shared a container")
            if receipt["status"] in ("complete","forfeit","bilateral_failure"):
                for c in receipt["containers"]:
                    require(c["network"]=="none" and c["read_only_root"] and c["memory"]==c["memory_swap"]==536870912 and c["nano_cpus"]==1500000000,"resource qualification")
            game = S.settle_game(verified,target)
            game.update(probes=receipt.get("probes"), container_settings=receipt.get("containers"),
                        package_sha256=registration["files"]["candidate"]["sha256"],
                        examiner_sha256=registration["files"]["opponent"]["sha256"])
            games.append(game)
        require(games[0]["game"]["image_id"] == games[1]["game"]["image_id"], "pair images differ")
        return games


def number(v, digits=2):
    return "—" if v is None else f"{v:,.{digits}f}"


def outcome(game):
    if not game["scored"]:
        return "未计"
    if game["winner"] is None:
        return "平" if game["status"]=="complete" else "双方故障"
    return "胜" if game["winner"]==game["game"]["seat"] else "负"


def tally(games):
    c=Counter(map(outcome,games))
    return "/".join(str(c[k]) for k in ("胜","平","负")) + ("" if sum(c[k] for k in ("胜","平","负"))==len(games) else " 另有未定")


def mean(rows):
    return statistics.mean(rows) if rows and None not in rows else None


def money(row,s):
    led=row[s]
    return "/".join(number(v,0) for v in (led.get("own_delivery",0),led.get("ground_rent",0),led.get("cargo_tax",0),-led.get("plant",0),-led.get("seize",0),-led.get("hire_collector",0)-led.get("hire_planter",0)))


def pivot_actors(z):
    p=z["largest_observed_cash_loss"]
    if not p:return None
    tape=p["from_t_minus_10"];last=tape[-1];own=z["game"]["seat"]
    candidates=[e for e in last["events"] if len(e)>=5 and "worker" in str(e[3])]
    if not candidates:return None
    signed=lambda e: e[2] if e[1]==own else -e[2]
    event=min(candidates,key=signed)
    if signed(event)>=0:return None
    uid=event[3]
    side=next((s for s in (0,1) if uid in last["players_before"][s][2]),None)
    if side is None:return None
    def dist(a,b):
        dx=abs(a%15-b%15);dy=abs(a//15-b//15)
        return min(dx,15-dx)+min(dy,15-dy)
    other=last["players_before"][1-side][2]
    near=min(other,key=lambda key:(dist(other[key][0],event[4]),key), default=None)
    def trace(who,s):
        rows=[]
        for back in (10,5,3,0):
            t=last["t"]-back
            for item in tape:
                if item["t"]!=t: continue
                obj=item["players_before"][s][2].get(who)
                rows.append(dict(t=t,uid=who,side=s,position=obj[0] if obj else None,cargo=obj[1] if obj else None,
                    command=item["joint_orders"][s].get(who,"NONE") if obj else None))
        return rows
    return dict(event=event,candidate_signed_change=signed(event),actor=trace(uid,side),
                nearest_opposite_worker_before_settlement=trace(near,1-side) if near else [],
                basis="largest individually identified worker event against candidate at largest cash-loss transition; closest opposing worker to its settlement cell before that turn, for descriptive context only")


def clustered(games):
    seeds=sorted({g["game"]["seed"] for g in games})
    require(len(seeds)==2,"this reduction is for two paired physical maps")
    by_arm={arm:[[z for z in games if z["game"]["seed"]==seed and z["game"]["arm"]==arm] for seed in seeds] for arm in ARMS}
    points={}
    for arm in ARMS:
        points[arm+"_win_rate"]=[sum(outcome(z)=="胜" for z in pair)/2 if len(pair)==2 and all(z["scored"] for z in pair) else None for pair in by_arm[arm]]
        points[arm+"_official_margin"]=[mean([z["official_margin"] for z in pair]) for pair in by_arm[arm]]
    for metric in ("win_rate","official_margin"):
        points["student_minus_parent_"+metric]=[a-b if None not in (a,b) else None for a,b in zip(points["student_"+metric],points["parent_"+metric])]
    measures={}
    for k,v in points.items():
        if None in v:
            measures[k]={"map_points":v,"mean":None,"empirical_cluster_bootstrap_percentile_95":None,"normal_map_effect_paired_t_95_df1":None}
            continue
        x=mean(v)
        # All two-map cluster resamples with replacement: (a,a), (a,b), (b,a), (b,b).
        draws=sorted([v[0],x,x,v[1]])
        half=12.7062047364*abs(v[0]-v[1])/2
        measures[k]=dict(map_points=v,mean=x,exact_empirical_bootstrap_draws=draws,
            empirical_cluster_bootstrap_percentile_95=[draws[0],draws[-1]],normal_map_effect_paired_t_95_df1=[x-half,x+half])
    return dict(seeds=seeds,num_independent_maps=2,paired_unit="the same physical map, both seats and arms kept together",
        method="Exact enumeration of the 4 possible map-cluster resamples. Model-dependent t interval for the mean has one degree of freedom. A degenerate bootstrap or t interval with two equal map outcomes does not establish population certainty.",measures=measures)


def report(games, out, run):
    lines=["# Sapling 全骨干与首胜父代对原版 Master3s2 的开发场","",
           "每个候选在两张预登记地图各打双席。考官逐局自然采样，逐拍回放公开请求与返回动作，并按官方判灭绝和终局计分。","",
           "|版本|席0 胜/平/负|席1 胜/平/负|全批 胜/平/负|官方平均分差|","|---|---|---|---|---:|"]
    for arm in ARMS:
        rows=[z for z in games if z["game"]["arm"]==arm]
        lines.append("|"+"|".join([ROLES[arm],*(tally([z for z in rows if z["game"]["seat"]==s]) for s in (0,1)),tally(rows),number(mean([z["official_margin"] for z in rows]))])+"|")
    lines+=["","|地图|席|版本|官方比分（本方/对手）|结果|结算步|考官正深/多候选零深/单候选|墙钟秒|","|---|---:|---|---|---|---:|---|---:|"]
    for z in games:
        v=z["game"];s=v["seat"];scores=v["official_scores_by_seat"];kind=z["original_search"]["branches"]
        lines.append(f"|{v['seed']}|{s}|{ROLES[v['arm']]}|{number(scores[s]) if scores else '—'} / {number(scores[1-s]) if scores else '—'}|{outcome(z)}（{z['status']}）|{v['official_transitions']}|{kind.get('searched',0)}/{kind.get('zero',0)}/{kind.get('single',0)}|{number(z['whole_wall_seconds'])}|")
    lines+=["","## 运行条件","","每格依次为冷启秒、外部请求 P95 / P99 / 最大毫秒、慢步 / 超时 / 错误、累计超额秒、容器峰值 MiB。百分位使用最近秩。","","|地图|席|版本|本方|考官|","|---|---:|---|---|---|"]
    for z in games:
        v=z["game"];s=v["seat"]
        def clock(side):
            t=z["ladder_clock_by_seat"][side]
            if not t["requests"]:return "未发步请求"
            return f"{t['init']:.3f} · " + "/".join(number(1e3*t[k],1) for k in ("p95","p99","max")) + f" · {t['slow']}/{t['timeouts']}/{t['errors']} · {t['excess']:.3f} · {t['cgroup_peak_bytes']/2**20:.1f}"
        lines.append(f"|{v['seed']}|{s}|{ROLES[v['arm']]}|{clock(s)}|{clock(1-s)}|")
    lines+=["","## 经济账与转折","","各方依次为回款/纯地租/货税/栽树开支/夺树开支/招工开支。逐拍现金另核对基地或树格碰撞、敌方误入存入和舍入。末段取实际最后 99 个已推进回合。","","|地图|席|版本|本方|考官|末段回款净差|最大损失步及该步差|回溯3/5/10步含该步的现金净变化|","|---|---:|---|---|---|---:|---|---|"]
    for z in games:
        v=z['game'];s=v['seat'];p=z['largest_observed_cash_loss'];l=z['money_by_period_by_seat']['whole']
        look=" / ".join(number(p["lookbacks"][str(h)]["cumulative_margin_change"]) for h in (3,5,10)) if p else "—"
        lines.append(f"|{v['seed']}|{s}|{ROLES[v['arm']]}|{money(l,s)}|{money(l,1-s)}|{number(z['margin_components']['last_99_observed']['own_delivery'])}|{str(p['t'])+' / '+number(p['change']) if p else '—'}|{look}|")
    lines+=["","最大损失拍中，以下列出使本方少收或多付最多的单笔可归身工人事件，以及该拍距结算格最近的对侧工人。指令按 t−10 / −5 / −3 / 当拍排列，附出令前位置与货量。没有可归身工人事件的场次保留空值。","",
            "|地图|席|版本|入账方与事件；本方净影响|当事工人及动作|对侧最近工人及动作|","|---|---:|---|---|---|---|"]
    def desc(rows):
        if not rows:return "—"
        def entry(y):
            return f"t{y['t']} 未出场" if y["position"] is None else f"t{y['t']} {y['command']}@{y['position']}(货{number(y['cargo'],0)})"
        return f"席{rows[0]['side']} {rows[0]['uid']} " + " / ".join(entry(y) for y in rows)
    for z in games:
        v=z["game"];ev=z.get("pivot_actors") or pivot_actors(z)
        if not ev:lines.append(f"|{v['seed']}|{v['seat']}|{ROLES[v['arm']]}|—|—|—|");continue
        e=ev["event"];ls=FIELD.get(e[0],e[0])
        lines.append(f"|{v['seed']}|{v['seat']}|{ROLES[v['arm']]}|席{e[1]} {ls} {number(ev['candidate_signed_change'])}|{desc(ev['actor'])}|{desc(ev['nearest_opposite_worker_before_settlement'])}|")
    lines+=["","## 同图换席比较","","|地图|全骨干双席胜数|父代双席胜数|全骨干双席平均分差|父代双席平均分差|配对差|","|---|---:|---:|---:|---:|---:|"]
    for seed in sorted({g["game"]["seed"] for g in games}):
        arms=[[z for z in games if z["game"]["seed"]==seed and z["game"]["arm"]==arm] for arm in ("student","parent")]
        ms=[mean([z["official_margin"] for z in rows]) for rows in arms];diff=ms[0]-ms[1] if None not in ms else None
        lines.append(f"|{seed}|{sum(outcome(z)=='胜' for z in arms[0])}|{sum(outcome(z)=='胜' for z in arms[1])}|{number(ms[0])}|{number(ms[1])}|{number(diff)}|")
    metrics=clustered(games)["measures"]
    lines+=["","按图聚类重采样保留同图两席及两版本。两图有四个等可能重采样，95% 百分位端点等于两张图的结果。下表也列出假定地图效应独立且近似正态时的 t 区间（自由度 1）。两图相同时区间可能退化。","",
            "|全骨干减父代|均值|按图重采样 95%|按图配对 t 区间 95%|","|---|---:|---|---|"]
    for field,label in (("official_margin","官方分差"),("win_rate","胜率差（百分点）")):
        q=metrics["student_minus_parent_"+field]
        scale=100 if field=="win_rate" else 1
        interval=lambda a: "—" if a is None else " 至 ".join(number(v*scale) for v in a)
        lines.append(f"|{label}|{number(None if q['mean'] is None else q['mean']*scale)}|{interval(q['empirical_cluster_bootstrap_percentile_95'])}|{interval(q['normal_map_effect_paired_t_95_df1'])}|")
    lines+=["","两张独立地图不足以估出可靠的总体胜率或分差区间。现金转折记录了当时的实际行动、开支和结算，单局相关性不能证明替代动作的收益。","",
            f"[原始运行](https://github.com/robertyang87/carbon-league/actions/runs/{run}) · [八局机器证据](settled_games.json.gz) · [GitHub 工件签名](archives.json)"]
    (out/"README.md").write_text("\n".join(lines)+"\n")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repo",required=True,type=Path)
    p.add_argument("--matrix",required=True,type=Path)
    p.add_argument("--archives",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    if not __debug__:raise RuntimeError("verification requires assertions")
    shared=args.repo/"reports/sapling_1200k/master6_actor_dev/cargo_tree_action_learning_dev/territory_sequence/fresh_original/economy"
    sys.path.insert(0,str(shared))
    import settle as S
    manifest=S.read(args.archives); matrix=S.read(args.matrix)
    registered_blob=subprocess.check_output(["git","-C",str(args.repo),"show",manifest["head_sha"]+":"+str(args.matrix.resolve().relative_to(args.repo.resolve()))])
    require(registered_blob==args.matrix.read_bytes(),"local matrix differs from frozen hosted revision")
    for role,sign in (("student",FULL),("parent",PARENT),("opponent",ORIGINAL)):
        require(matrix["files"][role]["sha256"]==sign,role)
    pairs={(int(seed),seat) for seed in matrix["seeds"] for seat in (0,1)}
    require(len(pairs)==4 and len(manifest["archives"])==4,"need the registered four pairs")
    games=[]
    with TemporaryDirectory() as tmp:
        for item in manifest["archives"]:
            path=args.archives.parent/item["file"];seed,seat=item["seed"],item["seat"]
            require((seed,seat) in pairs,"unexpected/duplicate map/seat")
            pairs.remove((seed,seat))
            require("sha256:"+digest(path)==item["digest"] and path.stat().st_size==item["size_in_bytes"],"GitHub ZIP binding")
            games.extend(verify_zip(path,seed,seat,args.matrix,manifest["run_id"],S,tmp))
    require(not pairs and len(games)==8,"matrix incomplete")
    games.sort(key=lambda z:(z["game"]["seed"],z["game"]["seat"],z["game"]["arm"]))
    for g in games:g["pivot_actors"]=pivot_actors(g)
    args.out.mkdir(exist_ok=True,parents=True)
    S.store(args.out/"settled_games.json.gz",dict(matrix_sha256=digest(args.matrix),run_id=manifest["run_id"],
        git_commit=manifest["head_sha"],program_sha256=digest(Path(__file__)),cash_replayer_sha256=digest(shared/"settle.py"),clustered=clustered(games),games=games))
    report(games,args.out,manifest["run_id"])
    print([(g['game']['seed'],g['game']['seat'],g['game']['arm'],outcome(g),g['official_margin']) for g in games])


if __name__=="__main__":
    main()
