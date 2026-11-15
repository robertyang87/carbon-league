# 实体机中转设计（仓库总线 + 每机轮询代理）— 2026-11-13 08:55 草案

背景：GitHub Actions 托管 runner 通道自 2026-11-13 04:13 起因账号 Actions 预算 $0「Usage blocked」全断（08:51 Billing 页直读坐实），全线调度冻结约 4.5 小时。所有者提议：用一台实体机克隆代码做中转，摆脱对 Actions 的单点依赖。

## 架构：仓库当总线，不设单点中转机
- 每台机器在本地 clone（carbon-league）里跑一个小守护脚本（poller）：每 30–60 秒 `git pull` → 查 `relay/<机器名>/inbox/` 是否有新任务文件 → 本地执行 → 结果写 `relay/<机器名>/outbox/` 并 `git push`。
- 作战室侧只用 GitHub API（与 Actions 额度无关、现已验证可用）下发任务文件、读回结果文件。
- Studio 兼任文件接力角色（既有已验证链）：局域网连 DGX（10.188.0.23，可 pull 可执行）、公网 SSH 连 L20；跨机大文件继续走 DGX→Studio→L20。
- 往返延迟约 1 分钟（轮询间隔量级）；结果以文件回传，无 Actions 的 ~15KB stdout 上限，大输出分片成文件即可。

## 各机 bootstrap 现状
- DGX：现成 — 已有真 clone（carbon-repo）＋可写 deploy key（id 169691313，2026-11-13 翻为 RW 并读回验证）。今天就能装 poller。
- Studio：工作目录是 rsync 副本、非 clone（2026-11-12 实测教训）。需新建真 clone 并确认推送凭据；没有就加一把 deploy key（所有者点一下，同 DGX 先例）。
- spot-29 / spot-30：用阿里云控制台网页云助手（现可用、不依赖 Actions）各装一次 poller，装完自驱；被回收则新机重装一次。
- gpu-l20-6：同 spot，可经控制台云助手装；或由 Studio 公网 SSH 装。

## 安全与纪律
- 任务只认 `relay/<机器名>/inbox/` 内带唯一标记的任务文件（沿用「认领只认标记」纪律）；执行日志与自证块随结果回传。
- 仓库为私有，写入方只有我方与 Codex；poller 只执行本机 inbox，避免跨机误认领。
- 不可逆动作（提交/删除/购买）仍走所有者闸，不因通道更换放宽。

## 与 Actions 的关系
- 中转搭好后 Actions 降级为备用通道；预算是否提高由所有者单独决定（应急仍以改预算最快：spot-29 回填死线 2026-11-13 13:08，中转 bootstrap 赶不上这条死线）。
- 待所有者点头后开工：① DGX 先装并跑通一趟端到端（下发→执行→回传）；② Studio 建 clone＋装；③ 两台 spot＋L20 装；④ 全线切换、Actions 冻结令解除改走中转。
