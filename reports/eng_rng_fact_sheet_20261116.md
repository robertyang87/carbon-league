# RNG／调用序事实单（提速实现轨用，DGX 线直读钉版源码，2026-11-16 21:42 CST）

对象：stage0_teacher.evaluate_root（c6ade5ce）＋ cnet2_np.Policy.act（d8c15ad4 系）＋ k0_annotate 调用点（7f567a2e）。生产形态：belief="sample"（v3）、K=16、M=8、H=2、teacher_style=2、guard on。

## 种子

- 每根 seed_base ＝ **seed_teacher ＋ step**（k0_annotate 调用点写死此式；seed_teacher 生产值 7000＝钉版默认，config 哈希 9a2b2440 覆盖）。同一根的全部随机性只由 (step, m) 决定，与候选臂、进程、机器无关。

## 随机抽取点：全函数仅两处 default_rng

1. **信念世界采样**（每场景 m＝0..7，在任何 act 之前）：rng ＝ default_rng([seed_base＋m, 0xC0])。belief_carbon 内对每个未知镜像轨道（按轨道代表 id 升序）**恒耗两抽**：rng.random()（非零判定）→ rng.uniform(1, 2·cmean)（初值）；与各轨道结果无关、流位置恒定。已知格零抽。mode="mean" 时整段不抽（生产为 sample）。
2. **对手根动作**（每场景 m，候选生成之前）：rng_m ＝ default_rng(seed_base＋m)——与信念流是**两条独立流**。以 sample_rng 传入 pol.act(temp=1.0)：未掩码单位逐个（单位序 s＝0..OWN−1）各耗一抽 sample_rng.choice(5, p=softmax(lg/temp))；掩码单位不抽。

除此之外 evaluate_root 内**再无随机源**（全文件 default_rng 仅上述两行）。

## pol.act 是否确定性

- **sample_rng=None 时全确定**：合法掩码后逐单位 argmax、自回归前缀推进；同 (x, aux, 权重, teacher) 必同输出。
- sample_rng 非空时仅上列第 2 处的单位被采样（sample_kinds=None ⇒ 全部未掩码单位）；其余调用（候选生成、续弈、守卫重解码）一律不传 sample_rng ⇒ 确定性。

## 调用序（每根）

1. 场景世界构建：m 升序 belief_carbon → roots[m]（**全臂共享，CRN**）。
2. 对手根动作：m 升序各算一次（**全臂共享**，候选不可影响对手根动作）。
3. 候选生成 gen_candidates：pol.act 贪心一臂（守卫 apply）＋按 (prio, gap) 排序的前 ≤6 个单位做 forced 前缀偏离（每次一回 pol.act＋守卫），去重至 K=16 止。无随机。
4. 评估：臂 ci 升序 × 场景 m 升序——克隆 roots[m]、根步落子、守卫自根快照恢复、H=2 续弈步（每步座 0 先 act、座 1 后 act，均无 sample_rng）、终局检查在每步后、叶值确定性网络读数 → Q[ci, m]。
5. q 由 Q／priors 按钉版构造（确定性）。

## 对批量化实现的约束（事实推论）

- 可批的维度：根 × 场景 × 臂——三者的种子与世界互不耦合；批内重排不改变任何一根的输入流，**前提是两条 rng 流的构造与消耗序逐根原样保留**（尤其信念流「每轨道恒两抽」与对手流「逐单位一抽」）。
- 不可批改的：对手根动作必须在候选生成前算毕且全臂共享；场景世界先于一切 act。
