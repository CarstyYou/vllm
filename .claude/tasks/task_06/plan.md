# task_06 — nsys e2e breakdown（单卡，cc=1/128 × 4 backend）

对应母 plan P5 可选佐证项 + exp_05 低并发劣势归因。分支 `cute_sm120_precision_internal`。
状态：**plan 待 xiy lock**（exp_05 bench 完成后启动）。

## 目标

回答 exp_05 观察到的**低并发（cc=1）cute 明显慢（-35%）而高并发（cc=128）仅 -3~5%** 的成因：
同一 case profile 全部 backend，看 e2e time breakdown（MoE GEMM 段 vs attention/GDN/其他），
定位 cute kernel 在小 batch 的劣势来源（kernel launch / tile 效率 / 占用率）。

## 范围（xiy 2026-07-15 lock）

| 维度 | 值 |
|---|---|
| 规模 | **单卡 35B**（1×6K Pro） |
| Case | **cc=1**（低并发劣势点）+ **cc=128**（高并发主场景），各 1 个 profile |
| Backend | **cute_sm120_mxfp8_32 vs deep_gemm_mxfp8_32**（xiy 2026-07-15：1×128×128 粗粒度精度不够，
           聚焦细粒度 OCP K=32 一对，同 recipe 只差 kernel 实现 → 直接归因 cute kernel 的低并发劣势）；
           triton 作 throughput/latency 参照锚（轻量，可选保留） |
| 固定 | ISL/OSL 8192/1024、`--ignore-eos`、prefix caching 关（同 exp_05 serve 配置）；
        同 case 两 backend apple-to-apple |

= 2 case × 2 backend（+triton 参照）= **4-6 个 .nsys-rep**。

## 方法

- nsys profile serve 进程（`nsys profile -o <tag> --trace cuda,nvtx --capture-range ...`），
  用 delay/duration 只 capture 稳态窗口（避开 warmup/首 batch JIT）；bench 打固定 num_prompts 负载
- 分析走 veloq **nsys-profile-analysis** skill：GPU idle gap / launch cause / NVTX / kernel 占比
- breakdown 表：每 backend × case 的 e2e 时间分解（MoE GEMM % / attention % / GDN % / 其他 / GPU idle）
- 对比锚点：cute_fp8 cc=1 vs triton cc=1 的 MoE GEMM 段绝对时间差 → 定位 -35% 来自哪段

## Sub-tasks

| # | 内容 | Gate |
|---|---|---|
| 0 | nsys capture 脚本 + 单点冒烟（triton cc=128，验证 .nsys-rep 可开 + NVTX 有 MoE 段） | rep 非空 + veloq 能解析 |
| 1 | profile run（2 case × cute_mxfp8_32/dg_mxfp8_32 [+triton]，单卡串行） | 4-6 .nsys-rep |
| 2 | veloq 分析 + breakdown 对比表 → result.md + 收口 | xiy 确认 → commit |

## Risks

| 风险 | 预案 |
|---|---|
| nsys overhead 扭曲 timing | 只看相对占比 breakdown，不用 nsys 下的绝对 tok/s |
| cc=128 profile 文件过大（8k×512 prompt） | capture window 收窄到几个 decode step；只需稳态代表段 |
| MoE kernel 在 NVTX 无明确 range | 用 kernel name 匹配（cute/DG/triton MoE GEMM kernel 名）替代 NVTX |

## Plan Review

（exp_05 收口后、启动前补 experiment-plan-review）
