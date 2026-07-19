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

- nsys profile serve 进程（`nsys profile --trace cuda,nvtx`），单 request（cc=1）= 1 次 8192
  prefill + 1024 decode step；负载用 bench 打固定 1 prompt（cc=1）/ 512 prompt（cc=128）
- **prefill / decode 相位必须拆分**（review gap A/C）：cc=1 的 -35% = prefill（M 大，cute 不吃亏）
  + decode（M=1，cute 小 batch 惩罚在此）的混合。用 NVTX/iteration step marker（vLLM 每
  forward step 有 marker）把 MoE GEMM kernel 按相位归属，**产出每 decode step 的 MoE kernel
  绝对时间 cute vs dg**，不只一个混合 "MoE GEMM %" 桶
- **capture window 锚到干净 decode step**（review gap B）：用 NVTX/step marker 锚定若干稳态
  decode step（避开 warmup/JIT/prefill），不用脆弱的 time-based delay/duration
- 分析走 veloq **nsys-profile-analysis** skill：GPU idle gap / launch cause / NVTX / kernel 占比
- breakdown 表：每 backend × case ×（prefill 段 / decode 段）的时间分解（MoE GEMM 绝对 μs +
  attention / GDN / 其他 / GPU idle 占比）
- **闭环校验**（review gap B）：(cute−dg 每 decode step MoE Δ)×1024 + prefill Δ ≈ 126 vs 185
  tok/s 隐含的 e2e 单 request 时间差 → breakdown 数字必须能对上 exp_05 的宏观 tok/s，否则重查
- **GDN backend 对齐**（review gap D）：全部列（含 triton 参照）用 **FI GDN**（同 exp_05 sm120 列），
  不用 triton-GDN 变体，避免 GDN 路径差污染 MoE 归因

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

**Date**: 2026-07-16
**Reviewer**: subagent

**Verdict**: ⚠️ Gaps（4 条，已全部一次性修入方法节，不 re-review）

**Gaps + suggested fix**:
- A（重大）prefill MoE vs decode MoE 未拆分 → cc=1 的 -35% 混了 large-M prefill（cute 不吃亏）
  和 tiny-M decode（惩罚所在）：按相位拆两段，产出每 decode step MoE kernel 绝对时间 ✓ 已修
- B（重大）capture window 未锚 decode + 缺闭环校验 → 用 NVTX/step marker 锚干净 decode step，
  加 reconciliation（breakdown Δ × steps ≈ tok/s 隐含时间差）✓ 已修
- C（中）kernel-name 匹配拆不开相位 → 需 per-step marker 归相位 ✓ 已修（并入 A）
- D（次）triton 参照 GDN backend 未指定 → 全列用 FI GDN（同 exp_05 sm120 列）✓ 已修
