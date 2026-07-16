# task_05 — e2e serving 性能对比（cute vs baselines，xiy 2026-07-15 lock 负载矩阵）

对应母 plan 5.x（P5）。分支 `cute_sm120_precision_internal`。
状态：**plan 待 experiment-plan-review + xiy lock**。

## 目标

同一 serve 配置下，5 个 backend 列的 serving 吞吐/延迟对比，单卡 + 多卡；
与精度矩阵（task_04）配对成 manager 报告的第二张表。

## 负载矩阵（xiy 2026-07-15 lock）

| 维度 | 值 |
|---|---|
| ISL/OSL | **8192 / 1024**（单场景） |
| 并发 | **1 / 2 / 4 / 8 / 16 / 32 / 64 / 128**（`--max-concurrency` 模式，8 点） |
| Backend 列 | H20 fp8（= H20 硬件上 deep_gemm float，`E8M0=0`）/ triton / **deep_gemm**（sm120 UE8M0-128，xiy 2026-07-15 加）/ cute_sm120_fp8 / cute_sm120_mxfp8_32 / deep_gemm_mxfp8_32（后 5 列在 sm120）——共 6 列 |
| 规模 | 单卡 35B；多卡 397B TP8 + DSv4-Flash TP4（若 xiy 裁剪多卡模型另 lock） |

## 测试纪律（沿 P5 草案）

- 工具 `vllm bench serve`（openai backend 打独立 serve 实例），random dataset 固定 seed
- 每 cell：warmup 1 轮不计 + 正式 3 轮取 median；同一 backend-model 的 8 个并发点在同一 serve 实例上连跑
- 同一模型的全部 backend 列在同型号节点上跑完（sm120 列同一台 6K Pro；H20 列同一台 H20/H20-3e）
- 指标：output tok/s（主）、TTFT p50/p99、TPOT p50/p99；ratio 报百分比
- serve 配置 = `--max-model-len 10240`（8k+1k+余量）+ **`--no-enable-prefix-caching`**
  （review gap 1：固定 seed 多轮下 prefix cache 会让 8k prefill 全命中、prefill 从测量消失
  ——perf 与精度 v2 的这一差异 explicit 记录）；cudagraph/compile 默认开；
  H20 列加 `--gdn-prefill-backend triton` + `VLLM_USE_DEEP_GEMM_E8M0=0`；
  DSv4 加 `--kv-cache-dtype fp8_ds_mla`
- bench 命令 **lock `--ignore-eos`**（review gap 2：防各列 early-EOS 生成长度不等），
  收数时 verify 实际 output len = 1024
- **GDN 控制点**（review gap 3）：sm120 列用 FI GDN、H20 列用 triton GDN——补 1 个
  sm120 triton-GDN 控制 run（35B triton 列 cc=32 单点）bound GDN 路径对跨硬件对比的污染；
  DSv4 无 GDN 不受影响
- **H20 SKU 分标**（review gap 4）：单卡 = 96G H20、多卡 = H20-3e（HBM3e）——perf 上不等价，
  报告列名带 SKU，不合并
- **KV 容量预算**（review gap 5）：35B 单卡 @10240 ctx ≈ 2.1M token KV ≥ cc128×9216=1.18M ✓
  无 preemption；DSv4 TP4 fp8_ds_mla（MLA 压缩 KV ~0.6KB/token）≈ 0.7G/GPU ✓ 充裕；
  397B TP8 sub-task 3 起跑前用 serve log 的 KV 容量行核对，不足则记录 preemption 拐点语义
- num-prompts per 点 = max(4×cc, 16)（低并发点保证样本量、高并发点控制时长）
- 每 run 产 json + evidence 行；结果只进白名单（bench json/csv + evidence）

## Sub-tasks

| # | 内容 | Gate |
|---|---|---|
| 0 | bench 脚本 + 单点冒烟（35B triton cc=8 一点，验证 metrics/json/output-len） | 冒烟 json 齐全 + output len=1024 |
| 1 | 单卡 35B × sm120 5 列 × 8 并发点（1×6K Pro）+ GDN 控制点 1 run | 41 cells 落表 |
| 2 | 单卡 35B × H20 列（1×96G H20） | 8 cells |
| 3 | 多卡 397B TP8 × 6 列（8×6K Pro + 8×H20-3e）；起跑前 KV 容量核对 | 48 cells |
| 4 | 多卡 DSv4 TP4 × 6 列（整机独占，不 lane 并行） | 48 cells |
| 5 | 汇总 result.md（perf 表 + TTFT/TPOT 附表）+ data-audit + 收口 | xiy 确认 → commit |

注：**nsys e2e breakdown 移到 task_06**（xiy 2026-07-15）——单卡 profile，cc=1/128 两 case × 4 backend。

## 时长估算

每 cell ≈ 1 轮 warmup + 3 轮 ×（num_prompts×1024 tok 生成）；cc=1 点最慢（串行 16 prompts×3 轮）。
估：单 backend-model 8 点 ≈ 1.5-2.5h（含 serve 起停 1 次）→ sm120 侧 ~12×2h ≈ 24h（可双节点分流）、
H20 侧 ~3×2h ≈ 6h。GPU 需求：1×6K Pro、8×6K Pro、1×H20、8×H20-3e 各一段。

## Risks

| 风险 | 预案 |
|---|---|
| ISL 8k × cc128 的 KV 压力（35B 单卡 47G KV ≈ 5k×9216 tokens；128 并发会触发 preemption） | 如实测到 preemption，记录并保留（真实 serving 行为）；TTFT p99 会体现 |
| H20 8 卡节点不稳（前科：一晚死两台） | run 粒度 = backend-model，死了单个重跑 |
| cc=1 点时长（纯串行） | num_prompts=16 下 ~25min/轮 ×3——若超预算改 cc=1 只跑 1 正式轮（记录偏差） |
| 多卡 DSv4 lane 并行与 bench 干扰 | perf run 一律整机独占（不 lane 并行，与精度不同） |

## Results

（待填；result.md 收口）

## Plan Review

**Date**: 2026-07-15
**Reviewer**: subagent

**Verdict**: ⚠️ Gaps（5 条，已全部一次性修入 plan，不 re-review）

**Gaps + suggested fix**:
- prefix caching × 固定 seed 多轮 → prefill 全命中、测量变 decode-only：perf serve 改 `--no-enable-prefix-caching`，与精度 v2 差异 explicit 记录 ✓ 已修
- 未 lock `--ignore-eos` → 列间生成长度不等不可比：bench lock + 收数 verify output len=1024 ✓ 已修
- GDN prefill backend 跨列不一致（H20=triton / sm120=FI）污染跨硬件归因：加 sm120 triton-GDN 控制点 1 run bound 效应 ✓ 已修
- H20 列混 96G H20 与 H20-3e 两个 SKU，perf 不等价：报告按 SKU 分标不合并 ✓ 已修
- DSv4/397B KV-preemption 预算缺失：补容量估算 + 397B 起跑前 serve log 核对 ✓ 已修
