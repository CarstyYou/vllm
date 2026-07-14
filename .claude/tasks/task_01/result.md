# task_01 Result — vLLM 双 baseline（triton float-scale vs deep_gemm UE8M0）

日期：2026-07-13。环境：Qwen3.5-35B-A3B-FP8，单卡 6K Pro (sm120, server edition)，
vLLM `0.23.1rc1.dev+gdcf4072`（分支基线），temp=0 / seed=42。状态：**完成，待 xiy 收口确认**。

## 结论

1. **UE8M0 (deep_gemm) 相对 float-scale (triton) 在 GSM8K 全量上低 2.5pp**（0.772 vs 0.797，
   stderr≈1.1pp），**而 MMLU 全量持平**（0.8478 vs 0.8465，差 < 1 stderr）——退化只在长推理链
   （逐 token 误差累积）显形，单点判别任务无感；与 vLLM 对 `qwen3_5_moe_text` 家族
   auto-disable DeepGEMM 的官方理由（E8M0 accuracy degradation）方向一致，
   且说明该类退化会被 MMLU 型评测漏检。float-scale 路径的精度价值在该家族成立。
2. **DG nv-dev kernel 在 sm120 实跑稳定**：显式 `--moe-backend=deep_gemm` 覆盖 auto-disable，
   GSM8K 两轮 + MMLU serve 无 crash。
3. sm120 上 float-scale FP8 MoE 原生只有 triton 兜底（其余 backend 全被 arch/recipe gate）——
   task_02 cute FP8 接入的对齐目标与性能对手即它。

## 测试条件（完整复现所需）

| 项 | 值 |
|---|---|
| 模型 | `/home/scratch.trt_llm_data/llm-models/Qwen3.5-35B-A3B-FP8`（fp8, weight_block_size [128,128], act dynamic） |
| 硬件 | 1× RTX PRO 6000 Blackwell Server Edition（96GB, sm120），节点 smc521ge-0036 |
| 软件 | vLLM `0.23.1rc1.dev1092+gdcf4072da`（分支基线 dcf4072，未含本分支改动）；torch 2.11.0+cu130；`CUDA_HOME=cuda-13.3`（DG JIT） |
| serve | `--max-model-len 4096 --moe-backend={triton\|deep_gemm} --disable-uvicorn-access-log`，其余默认（cudagraph/compile 开） |
| GSM8K | vLLM 自带 `gsm8k_eval.py`：1319 题全量、5-shot、temp=0、seed=42、max_tokens=256 |
| MMLU | lm-eval 0.4.12 local-completions：mmlu 全集 14042、5-shot、seed=42、num_concurrent=128 |
| 脚本 | `tests/run_gsm8k.sh` / `tests/run_mmlu.sh`（本目录，含全部 env） |

## Recipe 说明（两 backend 吃同一份 checkpoint，差异在 load/runtime 的量化语义）

| backend | weight | activation | 与 checkpoint 的关系 |
|---|---|---|---|
| triton | fp8 e4m3 + **float32 scale**，128×128 block（checkpoint 原样） | 动态 1×128 分组，**float32 scale** | 零转换 |
| deep_gemm | load 期 requant：每 128×128 块 dequant → scale **ceil 到 2 的幂（UE8M0）** → fp8 重铸（mantissa 变）；scale 4 个/int32 packed | 动态 1×128 分组，scale 同样 ceil-pow2 + packed | 有损转换（`requant_weight_ue8m0_inplace`） |

即 deep_gemm 列 = **UE8M0 GranK=128** 语义（分块随 checkpoint，非 OCP MXFP8 的 1×32）；
与第二轮路径 3a 同 recipe，可直接作其原生参照。

## 精度（结论只看全量）

| backend | recipe | GSM8K 1319 | MMLU 14042 | 证据 |
|---|---|---|---|---|
| triton | float-scale FP8 (1x128/128x128) | **0.797**（invalid 0.001） | **0.8465**（±0.0029） | `results/gsm8k_triton_n1319.{log,json}`、`results/mmlu_triton/` |
| deep_gemm | UE8M0（分块同上，scale ceil-pow2 packed int32） | **0.772**（invalid 0.000） | **0.8478**（±0.0029） | `results/gsm8k_deep_gemm_n1319.{log,json}`、`results/mmlu_deep_gemm/` |

Backend 证据行（serve log）：triton = `Using TRITON Fp8 MoE backend`（`results/serve_triton.log`）；
deep_gemm = `Using DEEPGEMM Fp8 MoE backend` + `DeepGEMM E8M0 enabled`（`results/serve_deep_gemm.log`）。
