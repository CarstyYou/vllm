# task_01 — 环境 + vLLM 双 baseline 对比（triton float-scale vs deepgemm UE8M0）

对应母 plan sub-task 1.1 + 1.2。分支 `cute_sm120_precision_internal`（基 `dcf4072`）。

## Scope

- In：单卡 6K Pro 环境（uv venv + vLLM 源码安装）；Qwen3.5-35B-A3B-FP8 用
  `--moe-backend=triton` 与 `--moe-backend=deep_gemm` 分别 serve + GSM8K + MMLU，数字落表
- Out：cute kernel 接入（task_02）、多卡、perf
- 资源约束：GPU 只用 **rtx-pro-6000-blackwell-server-edition**（不用 workstation edition）

## Sub-tasks

| # | 内容 | Gate |
|---|---|---|
| 0 | ssh-gw 申请单卡 6K Pro（24h）+ 确认 scratch.trt_llm_data 挂载 | 节点可用，模型路径可读 |
| 1 | uv venv (py3.12) + vLLM 安装（先试 VLLM_USE_PRECOMPILED=1；DG kernel 缺 sm120 则 fallback 源码 build / DEEPGEMM_SRC_DIR） | `import vllm` OK + `torch.cuda` 识别 sm120 + deep_gemm 可 import |
| 2 | triton baseline：`--moe-backend=triton` serve 35B + GSM8K smoke（~100 题）→ 全量 1319 题 | 精度数字落表 |
| 3 | deepgemm baseline：`--moe-backend=deep_gemm` serve + 确认 UE8M0 生效（log "DeepGEMM E8M0 enabled" + requant 路径）+ GSM8K 同规模 | 精度数字落表；若 qwen3_5 排除列表把 DG 自动禁掉 → 记录行为并用显式 backend 强制 |
| 4 | MMLU：lm-eval local-completions 对两 backend 各跑一轮 | 数字落表 |
| 5 | Results 汇总 + findings 沉淀 | plan.md ## Results 填表 |

## Risks

| 风险 | 预案 |
|---|---|
| 预编译 wheel 不含 DG sm120 kernel | pip deep-gemm / DEEPGEMM_SRC_DIR 指向本地 DeepGEMM checkout 源码 build |
| `should_auto_disable_deep_gemm` 对 qwen3_5 自动禁 DG | 显式 `--moe-backend=deep_gemm` 应绕过 auto-disable（oracle 显式优先）；行为差异记 findings |
| DG sm120 kernel 运行时问题（nv-dev 分支成熟度） | 记录错误，DG 列降级为"不可用"结论本身也是有效结果 |
| compute 节点无 HF 数据集网络访问 | GSM8K/MMLU 数据集预下载到 scratch，HF_DATASETS_OFFLINE |

## Results (2026-07-13)

### GSM8K（Qwen3.5-35B-A3B-FP8，单卡 sm120，5-shot，temp=0，seed=42）

| backend | recipe | n=100 冒烟 | n=1319 全量 | invalid |
|---|---|---|---|---|
| triton | float-scale FP8 (1x128/128x128) | 0.900 | **0.797** | 0.001 |
| deep_gemm | UE8M0 (分块同上, scale ceil-pow2 packed) | 0.810 | **0.772** | 0.000 |

### 运行遥测（非对比数据，仅留档）

eval 过程吞吐（fire-all 并发、混合 prefill/decode、无 warmup 控制，**不用于 perf 结论**）：
triton 全量 101.0s / 13.05 QPS / 1956 tok/s；deep_gemm 全量 106.4s / 12.40 QPS / 1905 tok/s。
正式 perf 对比 = P5 独立 serving bench（设计见母 plan）。

### 观察

- DG-UE8M0 全量比 triton float-scale 低 **2.5pp**（1319 题, stderr≈1.1pp）——方向与幅度和
  vLLM 对 qwen3_5 家族 auto-disable DG 的官方理由（"E8M0 causes accuracy degradation"）一致。
- 显式 `--moe-backend=deep_gemm` 可覆盖 auto-disable；DG nv-dev kernel sm120 实跑稳定（两轮无 crash）。
- 冒烟 100 题波动大（0.900/0.810），下结论只用全量数字。
- eval 吞吐：全量一轮 ~4-5 分钟（含模型加载），单卡足够快，全矩阵成本低。

### 待办

- MMLU 两 backend（sub-task 4）
- Results 数字进母 plan 4.1 矩阵（cute_fp8 列等 task_02）
