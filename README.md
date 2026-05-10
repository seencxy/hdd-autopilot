# hdd-autopilot

面向号多多 AI 公益站的多账号自动化 CLI。项目主体是 Rust，包含自动挖矿、批量账号玩法、游戏求解器、任务日志、跨平台打包和 GitHub Release 发布流程。

## 核心能力

- 自动挖矿：支持邀请码和余额兑换码两类奖励，可选择优先级或只挖其中一种。
- CPU + GPU 挖矿：CPU 始终可用；GPU 按平台自动尝试 CUDA、OpenCL、Metal，并根据设备参数和 HTTP challenge 参数自动调优。
- 多账号批量运行：账号登录态保存在本地缓存，玩法按账号并发执行。
- 有次数限制的白嫖玩法：签到、扫雷、羊了个羊、谜题2048、推箱子、点灯、迷宫、数织、连线、记忆翻牌、华容道、数独。
- 无次数限制的白嫖玩法：箭头逃离，支持单独运行和全自动持续运行。
- 赌狗玩法：自动随机刮刮乐。
- 稳定性处理：统一 HTTP 错误文案、token 过期重登、临时错误重试、服务端状态回查和残局恢复。
- 发布工具：本地脚本和 GitHub Actions 都按 target triple 输出规范包名。

## 快速开始

从源码运行：

```bash
cargo run --release --bin hdd-autopilot
```

打包后运行 `dist/` 下的对应平台文件：

| 平台 | 文件 |
| --- | --- |
| Windows x86_64 | `hdd-autopilot-x86_64-pc-windows-msvc.exe` |
| macOS Intel | `hdd-autopilot-x86_64-apple-darwin` |
| macOS Apple Silicon | `hdd-autopilot-aarch64-apple-darwin` |
| Linux x86_64 | `hdd-autopilot-x86_64-unknown-linux-gnu` |
| Linux aarch64 | `hdd-autopilot-aarch64-unknown-linux-gnu` |

macOS / Linux 包是自解压 shell wrapper，可以直接用 `sh` 启动：

```bash
sh dist/hdd-autopilot-x86_64-unknown-linux-gnu
```

如果要直接执行文件：

```bash
chmod +x dist/hdd-autopilot-x86_64-unknown-linux-gnu
./dist/hdd-autopilot-x86_64-unknown-linux-gnu
```

## 菜单结构

主菜单：

```text
1. 挖矿
2. 需要登录的多账号批量操作功能
3. 退出脚本
```

挖矿菜单：

```text
1. 先挖邀请码再挖余额码
2. 先挖余额码再挖邀请码
3. 只挖邀请码
4. 只挖余额码
5. 返回上一级菜单
6. 退出脚本
```

批量功能入口：

```text
1. 添加账号
2. 账号添加完成，选择脚本功能
3. 返回上一级菜单
4. 退出脚本
```

脚本功能分组：

```text
1. 白嫖玩法
   1. 有次数限制的白嫖玩法
      1. 全自动运行所有有次数限制的白嫖玩法
      2. 自动扫雷
      3. 自动羊了个羊
      4. 自动谜题2048
      5. 自动推箱子
      6. 自动点灯
      7. 自动迷宫
      8. 自动数织
      9. 自动连线
      10. 自动记忆翻牌
      11. 自动华容道
      12. 自动数独
      13. 自动签到
   2. 无次数限制的白嫖玩法
      1. 全自动运行所有无次数限制的白嫖玩法
      2. 自动箭头逃离
2. 赌狗玩法
   1. 自动随机刮刮乐
```

## 挖矿说明

挖矿不需要手动选择 CPU、GPU 或后端。每轮会从 HTTP 接口获取当前 challenge 参数，包括 `seed`、`round_id`、`visitor_id`、`challenge_id`、`session_salt`、`time_cost`、`memory_cost_mb`、`parallelism` 和 `difficulty_bits`，benchmark 和最终提交都基于当前真实参数。

后端选择规则：

- CPU 后端始终参与，作为稳定兜底。
- CUDA、OpenCL、Metal 会在当前平台和构建环境允许时自动探测。
- 每张 GPU 会读取显存、最大单次分配、计算单元、线程组限制、本地/共享内存、subgroup/warp 大小、统一内存、低功耗和外接设备等参数。
- 同一张设备如果被多个后端暴露，会按实测速度去重，只保留最快路径。
- 实际挖矿会同时使用最快 CPU 和去重后的可用 GPU；某个后端运行失败后会被临时黑名单并重新选择。

奖励输出：

```text
var/data/mining/invite-codes.txt
var/data/mining/balance-codes.txt
```

GPU 后端支持情况：

| 后端 | 平台 | 构建要求 |
| --- | --- | --- |
| CPU | 全平台 | 无额外要求 |
| CUDA | Windows x86_64、Linux x86_64；Linux aarch64 和 macOS x86_64 为可选支持目标 | CUDA Toolkit / `nvcc` |
| OpenCL | Windows、macOS、Linux | OpenCL headers、ICD loader 和厂商运行时 |
| Metal | macOS | macOS 原生构建环境 |

### RPOW2 挖矿

RPOW2（`https://rpow2.com/#/mine`）和号多多挖矿不是同一套协议。网页端会从 `https://api.rpow2.com/challenge` 取 `challenge_id`、`nonce_prefix` 和 `difficulty_bits`，然后计算：

```text
sha256(nonce_prefix_bytes || little_endian_u64(solution_nonce))
```

命中条件是 SHA-256 digest 的尾部零 bit 数大于等于 `difficulty_bits`。找到后提交：

```text
POST https://api.rpow2.com/mint
{ "challenge_id": ..., "solution_nonce": "..." }
```

本仓库提供独立入口：

```bash
RPOW2_COOKIE='从已登录浏览器请求里复制的 Cookie 请求头' cargo run --release --bin rpow2mine
```

同一入口也支持 RPOW3/RPOW4。RPOW4 的接口默认使用 `https://rpow4.com/challenge` 和 `https://rpow4.com/mint`；它的 `/mint` 需要浏览器会话 Cookie 加上匹配账号的钱包签名密钥：

```bash
export RPOW4_COOKIE='从已登录浏览器请求里复制的 Cookie 请求头'
export RPOW4_MNEMONIC='钱包助记词'
# 或者 export RPOW4_PRIVATE_KEY='钱包私钥/seed，base58 或 hex'
cargo run --release --bin rpow2mine -- --network rpow4
```

RPOW4 代理池默认读取 `rpow4-proxies.txt`，也可用 `RPOW4_PROXY_FILE` 或 `--proxy-file` 指定。开启代理并发时，challenge 会轮询代理池，mint 会固定回到获取该 challenge 的同一个代理：

```bash
export RPOW4_PROXY_FILE=rpow4-proxies.txt
export RPOW4_PROXY_CONCURRENCY=8
cargo run --release --bin rpow2mine -- --network rpow4 --loop
```

Windows 构建启用 CUDA 后会优先用 NVIDIA GPU；macOS 构建启用 Metal 后会优先用 Metal GPU；失败或不可用时回退 CPU。RPOW2 的 GPU 路径在 GPU 命中后会用 CPU 重新验算 digest 再提交。Windows CUDA 默认直接使用 `2147483648` hashes 的大 batch，避免启动后先跑小 batch；CUDA 路径会复用持久计算会话，避免每个 batch 重新分配显存。也可以离线验证 solver：

```bash
cargo run --release --bin rpow2mine -- --prefix <nonce_prefix_hex> --difficulty <bits> --cpu-only
```

固定 GPU batch size 时传 `--gpu-batch-size <hashes>`；Windows 默认 `2147483648`，传 `--gpu-batch-size 0` 表示自动调优。多块 GPU 时可用 `--gpu-device <index>` 指定设备。

Windows 上要真正看到 `backend=CUDA`，需要在那台机器安装 CUDA Toolkit、CMake 和可用的 NVIDIA 驱动，然后用 release 构建运行：

```powershell
$env:RPOW2_COOKIE='从已登录浏览器请求里复制的 Cookie 请求头'
cargo run --release --bin rpow2mine -- --loop --gpu-device 0
```

## 自动玩法策略

所有需要登录的功能共用同一套账号和异常处理：

- 添加账号时登录并保存账号缓存。
- 后续接口优先复用缓存 token；token 失效或缺失时自动重登并回写缓存。
- 所有需要登录的业务接口都会携带 `Authorization` token。
- 账号任务并发执行，单个账号或单个玩法失败不会阻塞其它账号。
- 401、登录态失效会触发一次登录恢复；408、425、429、5xx 和网络错误按统一重试策略处理。
- 点击、移动、填数、翻牌、结算等会改变服务端状态的接口，如果遇到超时、5xx、409 或响应丢失，会先回查 `/me` 和 `/history`，确认服务端是否已经推进，避免重复提交同一步或重复结算。
- 如果 `/me` 返回未完成残局，会先续残局，再按玩法策略开新局。
- 有剩余次数字段的玩法优先使用接口返回的剩余次数；没有可靠剩余字段的玩法会运行到接口明确拒绝继续开局。
- 无次数限制玩法持续运行，按 `ESC` 停止，终端固定显示所有账号实时总收益。

有次数限制玩法：

| 玩法 | 处理方式 |
| --- | --- |
| 自动签到 | 检查账号状态，未签到时领取奖励，失败后记录原因并继续后续玩法 |
| 自动扫雷 | 确定性规则、约束枚举、全局雷数概率和残局恢复 |
| 自动羊了个羊 | 按接口剩余次数和 active session 续局 |
| 自动谜题2048 | 支持 3x3 / 4x4 / 5x5，使用合法移动枚举和 expectimax |
| 自动推箱子 | 搜索箱子和玩家状态，带死角剪枝 |
| 自动点灯 | GF(2) 线性方程求解 |
| 自动迷宫 | 按开放边 BFS 最短路径 |
| 自动数织 | 行列候选生成和约束传播，只提交填充格子 |
| 自动连线 | 为每种颜色搜索端点路径，快速搜索失败后进入更完整搜索 |
| 自动记忆翻牌 | 记录已知卡牌，优先消除已知对子 |
| 自动华容道 | 校验可解性并按搜索路径提交移动 |
| 自动数独 | 求解完整棋盘，提交空格或可编辑错误格 |

无次数限制玩法：

| 玩法 | 处理方式 |
| --- | --- |
| 自动箭头逃离 | 读取当前局或新开局，按箭头方向、身体占用、障碍物和出口方向计算清除顺序 |

赌狗玩法：

| 玩法 | 处理方式 |
| --- | --- |
| 自动随机刮刮乐 | 补结算历史未开奖轮次，再随机选择玩法直到接口拒绝继续 |

## 运行数据和日志

运行时文件统一放在 `var/`，该目录已被 `.gitignore` 排除。

```text
var/
  data/
    auth.json
    mining/
      invite-codes.txt
      balance-codes.txt
  log/
    YYYYMMDD/
      checkin/checkin.log
      scratch/<sanitized-email>.log
      sheepmatch/<sanitized-email>.log
      puzzle_2048/<sanitized-email>.log
      minesweeper/<sanitized-email>.log
      sokoban/<sanitized-email>.log
      lightsout/<sanitized-email>.log
      maze/<sanitized-email>.log
      nonogram/<sanitized-email>.log
      flowfree/<sanitized-email>.log
      memory/<sanitized-email>.log
      puzzle_15/<sanitized-email>.log
      sudoku/<sanitized-email>.log
      arrow_out/<sanitized-email>.log
```

说明：

- `var/data/auth.json` 包含邮箱、密码和 access token，是敏感文件，不要提交。
- 启动时会尝试把旧版根目录数据迁移到 `var/data/`。
- 日志按系统当前时区分 `YYYYMMDD` 日期目录，再按项目和账号拆分。
- 签到日志按项目共用一个 `checkin.log`；其它玩法按账号拆分日志。
- 登录状态校验会传入系统当前 IANA 时区；无法识别时回退到 `Etc/UTC`。
- 旧缓存里的业务 Cookie 字段会被忽略，后续保存为 token-only 格式；挖矿内部 HTTP 会话不写入 `auth.json`。

## 构建和发布

单平台构建：

```bat
scripts\build-x86_64-pc-windows-msvc.bat
```

```bash
bash scripts/build-x86_64-apple-darwin.sh
bash scripts/build-aarch64-apple-darwin.sh
bash scripts/build-x86_64-unknown-linux-gnu.sh
bash scripts/build-aarch64-unknown-linux-gnu.sh
```

本地全平台汇总：

```bat
scripts\release.bat
```

```bash
bash scripts/release.sh
```

包状态文件：

| 状态 | 含义 |
| --- | --- |
| `built` | 完整包构建成功 |
| `built_degraded` | 基础包构建成功，但可选 GPU 原生后端缺少环境或构建失败 |
| `failed` | 基础构建失败 |

本地打包允许在缺少可选 GPU 构建环境时生成 `built_degraded` 包，方便当前机器可用；GitHub Actions 发布包要求状态必须是 `built`，不会自动发布降级包。

GitHub Actions：

- workflow：`.github/workflows/release.yml`
- 手动触发：构建五个平台并上传 Actions artifacts。
- 推送 `v*` tag：构建五个平台并发布 GitHub Release。
- Windows x86_64 和 Linux x86_64 job 会安装 CUDA Toolkit。
- Linux job 会安装 OpenCL build dependencies。

上传本地包到 GitHub Release 需要 GitHub CLI：

```bash
gh auth login
```

Windows PowerShell：

```powershell
powershell -NoLogo -ExecutionPolicy Bypass -File scripts/upload-release.ps1 v0.1.0
```

macOS / Linux / Git Bash：

```bash
bash scripts/upload-release.sh v0.1.0
```

上传脚本默认只上传状态为 `built` 的包；需要上传降级包时显式传 `--allow-degraded` 或 PowerShell 的 `-AllowDegraded`。

## 开发验证

常用检查：

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo check --examples
```

benchmark 辅助工具：

```bash
cargo run --release --bin benchcmp
cargo run --release --bin benchcuda
```

Windows 批量菜单 smoke：

```powershell
powershell -NoLogo -ExecutionPolicy Bypass -File scripts/smoke-batch-menu.ps1
```

额外覆盖更多独立入口：

```powershell
powershell -NoLogo -ExecutionPolicy Bypass -File scripts/smoke-batch-menu.ps1 -SkipScratch:$false -SkipMemory:$false -SkipPuzzle15:$false -SkipSudoku:$false -SkipArrowOut:$false
```

## 项目结构

```text
hdd-autopilot/
  src/
    api/          HTTP client、接口路径和错误文案
    cli/          CLI 菜单、账号添加和功能入口
    model/        API DTO、难度和玩法常量
    runtime/      var/dist/artifacts 路径解析和旧数据迁移
    solver/       游戏求解器
    storage/      账号缓存读写、归一化和旧格式兼容
    ui/           终端日志视图、滚动和 ESC 中断
    workflows/    单玩法流程和全自动调度
    bin/          benchcmp、benchcuda 辅助工具
  crates/
    mining/              挖矿控制面、CPU/GPU 选择和调优
    mining-cuda-sys/     CUDA FFI 封装
    mining-opencl-sys/   OpenCL FFI 封装
    mining-metal-sys/    Metal FFI 封装
  native/
    mining-cuda/         CUDA 原生计算核心
    mining-opencl/       OpenCL 原生计算核心
    mining-metal/        Metal 原生计算核心
  scripts/               构建、release、上传和 smoke 脚本
  .github/workflows/     GitHub Actions
```

## 隐私和安全

提交或推送前重点确认：

- 不提交 `var/data/auth.json`、`.env`、私钥、证书、数据库和本地日志。
- 不提交 `dist/`、`target/`、`artifacts/`、native build 目录。
- 示例账号只使用 `example.com` / `example.org` 这类占位数据。
- 不在 README、测试 fixture、脚本参数里写真实邮箱、密码、token、Cookie 或本地用户目录。

当前 `.gitignore` 已覆盖运行数据、构建产物、密钥文件、编辑器目录和本地 smoke 产物。
