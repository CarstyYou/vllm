# Shared Task Findings（vLLM cute_sm120_precision）

记客观发现 / bug / 踩坑 / 性能，不记主观设计决策。风格同 FI `tasks/memory/findings.md`。

## task_01: vLLM triton vs deepgemm baseline on sm120 (2026-07-13)

### 环境踩坑

- **裸机节点无 CUDA toolkit（无 nvcc），DG JIT import 直接 assert**：`vllm.third_party.deep_gemm`
  的 `_find_cuda_home()` 找不到 CUDA_HOME → `has_deep_gemm()=False`，DG 静默从 oracle 候选消失。
  解法：借共享 toolkit `CUDA_HOME=/home/scratch.jief_sw/cuda_toolkit/cuda-13.3`（配 torch cu130），
  设完 `is_deep_gemm_supported()=True`。所有 serve/eval 脚本必须带这个 env。
- `VLLM_USE_PRECOMPILED=1` 的 wheel **确实 vendor 了 `vllm.third_party.deep_gemm`**（源码树里没有，
  安装后出现），无需单独装 deep-gemm 包。
- ssh-gw 自动推导 partition 会拿 sinfo 显示串当 partition 名（salloc 报不存在）；
  必须 `--partition` 显式传 `scontrol show node` 里的真实名。
- `gsm8k_eval.py` 参数是 `--num-shots`/`--num-questions`/`--save-results`，无 `--model`
  （从 server 自动发现）。
- **ssh-gw task 自动 cd 到提交时的本地 cwd**：若 cwd=`mega_inference/`，其下的 `vllm/`
  repo 目录会被 Python 当 namespace package 盖掉 editable 安装 →
  `ImportError: cannot import name 'SamplingParams' from 'vllm' (unknown location)`。
  解法：跑 vLLM 的脚本必须显式 `cd $VLLM_ROOT`（不能依赖提交时的 cwd）。

### 实测

- **`qwen3_5_moe_text` 排除列表实锤命中**：serve Qwen3.5-35B-A3B-FP8 时 log
  `Auto-disabled DeepGemm for model_type=qwen3_5_moe_text on Blackwell. DeepGemm E8M0 scale
  format causes accuracy degradation for this architecture. Falling back to CUTLASS.`
  → hf sub-config 的 model_type 就是 `qwen3_5_moe_text`，AUTO 模式下 DG 不会被选；
  deep_gemm 列必须显式 `--moe-backend=deep_gemm` 强制。
- 显式 `--moe-backend=triton` 生效：oracle log `Using TRITON Fp8 MoE backend`。
- DG 环境自检通过：`DeepGEMM PDL enabled` + `DeepGEMM E8M0 enabled on current platform`（sm120）。
- **显式 `--moe-backend=deep_gemm` 覆盖 auto-disable 生效**（oracle 显式优先级 > 排除列表），
  DG nv-dev kernel sm120 实跑两轮（100+1319 题）无 crash。
- **GSM8K 全量（Qwen3.5-35B, 1319 题）：triton 0.797 vs deep_gemm-UE8M0 0.772（-2.5pp）**，
  与 vLLM 官方 auto-disable 理由方向一致；数据 link: [task_01/plan.md ## Results](../task_01/plan.md)。
  冒烟 100 题波动大（0.900/0.810），不可用于结论。
