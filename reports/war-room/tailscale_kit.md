# Tailscale 接入作战包（Carbon 中转链）

## 现状与总图

- 作战室这台 VM 内置 Tailscale 客户端（`/opt/hatch/bin/tailscale`，非上游 CLI），**已在所有者的 tailnet 内**：设备名 `muse`，IPv4 `100.92.70.87`，控制面 `controlplane.tailscale.com`。
- 本 VM 是**纯客户端**：只出站 TCP，经运行时代理改端口 **3130** 走 CONNECT 到 tailnet 地址；无 ping/UDP/ICMP，不接受入站、不做子网路由、不支持 exit node。测连通只能用真实 TCP 请求（ssh/curl），ping 不通什么都不证明。
- 目标拓扑：DGX、Studio、家里板子、gpu-l20-6 各自登进同一个 tailnet → 作战室逐台直连 SSH 派活/读回，替代已死的 GitHub Actions 通道；大文件仍走局域网直连/既有公网路径，tailnet 只作控制通道。
- 地址纪律：对某个 tailnet 地址的**第一次隧道请求需所有者批准一次**，之后同一地址的其他 TCP 端口在有效期内通用；短名/MagicDNS 名/IPv4/IPv6 算**不同地址**——全程固定用 **IPv4（100.x）**，不要混用。

## 所有者动作（每台一次，约 1 分钟）

- **DGX / 板子 / 其他 Linux**：`curl -fsSL https://tailscale.com/install.sh | sh && sudo tailscale up` → 终端弹出 `login.tailscale.com` 链接 → 所有者本人用账号登录批准。
- **Mac Studio**：装官方 Tailscale App，打开后登录同一账号。
- **gpu-l20-6**：无需所有者上机——安装与登录链接由作战室控制台任务在办，链接回传后所有者点开批准即可。
- 账号登录/批准动作只由所有者本人做；本文件不写任何密码/密钥值。

## 作战室验收清单（逐台勾选，通了才算接通）

目标设备：DGX spark-29db（LAN 10.188.0.23，用户 arm）；Mac Studio（LAN 10.188.0.21，用户 arm，macOS）；家里板子（10.188.5.23，用户 radxa）；gpu-l20-6（公网 139.196.109.84，root，注册名 `gpu-l20-6`）。

- [ ] **每台通用**：① `tailscale status` 确认该设备在列，记下其 100.x 地址（记到本节对应行）；② 经代理 ssh 跑一条只读命令（`hostname`、`uptime`），有回显才算接通。
  - DGX 地址：______（100.x 待记）
  - Studio 地址：______
  - 板子地址：______
  - gpu-l20-6 地址：______
- [ ] **DGX**：ssh 通后，从 DGX 侧在局域网内验板子 `10.188.5.23`（跨网段可达性，ping/ssh 均可——这是 LAN 内的测法，与 tailnet 无关）与 Studio `10.188.0.21`。
- [ ] **DGX 补钥**：把 gpu-l20-6 新生成的 `dgx_access` 公钥补进 DGX 的 `~/.ssh/authorized_keys`。公钥全文（2026-11-13 09:24 在 gpu-l20-6 生成，结构已验：base64 68 字符、解码 51 B）：`ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIG1jxVwnlc7IsmtGlVLtO2Wp0KdM9K1z9CgJqRKvZYFe l20-6-dgx-access-20261113`（指纹 SHA256:Fdt2wPTMWdP7LHi+S6ptm97b9zDcBsuc3jw8QkPC1yo）。补完从 L20-6 侧 ssh DGX 实测一次。
- [ ] **板子投用**：建 `~/junyang`，clone `robertyang87/carbon-league`（deploy key 先只读，写权限另议）；跑一趟端到端中转验证：作战室经 GitHub API 下发任务文件 → 板上执行 → 结果回传作战室读到，三步全过才算中转成立。
- [ ] **gpu-l20-6**：授权链接批准后，`status` 确认在网，代理 ssh 只读命令实测。

## 常用命令片段

```bash
# 本机状态与自身地址
/opt/hatch/bin/tailscale status
/opt/hatch/bin/tailscale ip

# 代理变量（每次开工先设；端口固定 3130）
# 注意：本环境 HTTPS_PROXY 含 userinfo（密码带冒号），${HTTPS_PROXY%:*} 会切错——直接写死主机名：
tunnel_proxy="hatch-egress-proxy:3130"
# gpu-l20-6 = 100.79.251.87（2026-11-13 09:52 入网，所有者本人点 Connect）；本 VM muse = 100.92.70.87。
# 全舰地址（2026-11-13 10:25 C1 自 tailscale status 读回）：dgx-spark 100.84.148.73、mac-studio 100.78.149.20、radxa-orion-o6 100.104.105.100、gpu-l20-6 100.79.251.87。按名（MagicDNS 短名）经隧道 SSH 已实测握手通（10:26），三台均差本 VM 公钥追加（已请所有者在设备前各跑一条命令）。
# 实测 09:57–09:59：经 3130 隧道连 100.79.251.87:22，TCP＋SSH 握手成功、host key 已存；当时机上 authorized_keys 还没有本 VM 钥匙（Permission denied publickey），已派控制台追加（…WAjlW hatch），追加后复测登录。每台设备第一个隧道请求按精确地址要所有者批准一次。

# HTTP 类只读探测（通了才算数）
curl --proxy "$tunnel_proxy" --fail-with-body "http://<100.x地址>:<端口>/<路径>"

# ssh 经代理（ProxyCommand 形式，二选一，以实测为准）
ssh -o ProxyCommand="nc -X connect -x ${HTTPS_PROXY%:*}:3130 %h %p" <用户>@<100.x地址>
# 等价写法：部分环境的 nc 用 -X connect 指定 HTTP CONNECT；若该写法不通，
# 改用支持 CONNECT 的代理客户端（如 corkscrew/ncat --proxy-type http）替换 ProxyCommand，其余参数不变。
```

## 纪律

- **传输**：大文件不走 tailnet 慢慢挪——局域网内机器互传走局域网直连，云机大文件走既有公网路径；tailnet 定位是控制通道＋小文件＋状态读取。
- **安全**：ssh 一律钥匙登录，不用密码；所有者的账号登录/批准只由所有者本人做；本文件与任何回报都不写密码/私钥值。
- **判定**：接通 = 真实 TCP 请求（ssh/curl）成功；status 在列、ping 不通/通，都不单独作数。
