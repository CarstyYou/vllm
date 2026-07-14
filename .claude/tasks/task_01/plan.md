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

## Results

见 [result.md](result.md)（数据总结 + 结论，review 入口）。
