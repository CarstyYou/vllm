# Precision 汇总（e2e 精度矩阵，随任务更新）

单一数据表，跨 task 汇总；每个数字可回溯到 task_NN/results/。
serve config：v1 = 无 prefix caching（task_01/02）；v2 = `--enable-prefix-caching`（task_03 起）。
**跨列比较只在同 config 版本内成立。**

## 单卡 Qwen3.5-35B-A3B-FP8

| Backend | Recipe | Config | GSM8K 1319 | MMLU 14042 | AIME24+25 | MBPP 500 | 证据 |
|---|---|---|---|---|---|---|---|
| triton | float-scale (1×128/128×128) | v1 | 0.797 | 0.8465 ±0.0029 | — | — | task_01/results |
| deep_gemm | UE8M0 GranK=128 | v1 | 0.772 | 0.8478 ±0.0029 | — | — | task_01/results |
| cute_sm120_fp8 | float-scale（零转换） | v1 | 0.790 | 0.8480 ±0.0029 | — | — | task_02/results |
| triton | float-scale | v2 | 0.7923 | 0.8470 ±0.0029 | 24: 0.033 / 25: 0.033 | 0.678 ±0.021 | task_04/results |
| deep_gemm | UE8M0 GranK=128 | v2 | 0.7726 | 0.8478 ±0.0029 | 24: 0.000 / 25: 0.000 | 0.656 ±0.021 | task_04/results |
| cute_sm120_fp8 | float-scale | v2 | 0.7801（invalid 0.0015） | 0.8472 ±0.0029 | 24: 0.000 / 25: 0.100 | 0.676 ±0.021 | task_04/results |
| cute_sm120_mxfp8_128 (3a) | UE8M0 GranK=128 | v2 | 0.7862 | 0.8463 ±0.0029 | 24: 0.033 / 25: 0.000 | 0.662 ±0.021 | task_03+04/results |
| cute_sm120_mxfp8_32 (3b) | UE8M0 GranK=32 (OCP) | v2 | 0.7885 | 0.8472 ±0.0029 | 24: 0.033 / 25: 0.000 | 0.668 ±0.021 | task_03+04/results |
| deep_gemm_mxfp8_32 | UE8M0 GranK=32 (OCP)，DG kernel | v2 | 0.7604（invalid 0.0008） | 0.8461 ±0.0029 | 24: 0.000 / 25: 0.033 | 0.666 ±0.021 | task_04/results |

注：AIME 两年各 30 题、stderr ~3-6pp，仅与 GSM8K 联看方向（母 plan Q6）；
35B 无思维链模式下 AIME 绝对值低是预期行为，列间无可分辨差异。
| **H20 baseline**（sm90，跨硬件） | **deep_gemm 显式 + `VLLM_USE_DEEP_GEMM_E8M0=0`**（float-scale；manager 要求非 triton。AUTO 落 TRITON 弃；E8M0 无架构 gate 会静默 UE8M0 requant）；serve 另加 `--gdn-prefill-backend triton` | v2 | **0.7930**（invalid 0） | **0.8478** ±0.0029 | 24: 0.000 / 25: 0.000 | 0.678 ±0.021 | task_04/results（tag `h20_deep_gemm`） |
| H20 参考行（非 baseline）：DG-UE8M0 | 同上但 E8M0 默认开（sm90 UE8M0 requant） | v2 | 0.6626 | 0.8466 | 0.033/0.000 | 0.666 | tag `h20_dgue8m0` |
| H20 参考行（非 baseline）：triton | AUTO 落点（FLA GDN） | v2 | 0.7445 | —（run 弃） | — | — | tag `h20_native`（仅 GSM8K） |

## 多卡（task_04 sub-task 5；节点 A=smc521ge-0080 / B=smc521ge-0039）

| Model | TP | Backend | GSM8K | MMLU | AIME | MBPP | 证据 |
|---|---|---|---|---|---|---|---|
| Qwen3.5-397B-A17B-FP8 | 8 | triton | 0.8886（invalid 0.011） | 0.8958 ±0.0025 | 24: 0.033 / 25: 0.000 | 0.020* | task_04/results |
| Qwen3.5-397B-A17B-FP8 | 8 | deep_gemm | 0.9007（invalid 0.003） | 0.8957 ±0.0025 | 24: 0.067 / 25: 0.000 | 0.010* | task_04/results |
| Qwen3.5-397B-A17B-FP8 | 8 | cute_sm120_fp8 | 0.8976（invalid 0.008） | 0.8957 ±0.0025 | 24: 0.067 / 25: 0.000 | 0.016* | task_04/results |
| Qwen3.5-397B-A17B-FP8 | 8 | cute_sm120_mxfp8_128 | 0.8999（invalid 0.007） | 0.8963 ±0.0025 | 24: 0.067 / 25: 0.000 | 0.004* | task_04/results |
| Qwen3.5-397B-A17B-FP8 | 8 | cute_sm120_mxfp8_32 | 0.9045（invalid 0.006） | 0.8955 ±0.0025 | 24: 0.200 / 25: 0.033 | 0.008* | task_04/results |
| Qwen3.5-397B-A17B-FP8 | 8 | deep_gemm_mxfp8_32 | 0.8961（invalid 0.005） | 0.8956 ±0.0025 | 24: 0.100 / 25: 0.000 | 0.002* | task_04/results |
| DeepSeek-V4-Flash-Base | 4 | triton | 0.9083（invalid 0） | 0.8852 ±0.0026 | 24: 0.000 / 25: 0.000 | 0.738 ±0.020 | task_04/results |
| DeepSeek-V4-Flash-Base | 4 | deep_gemm | 0.9098（invalid 0） | 0.8866 ±0.0026 | 24: 0.033 / 25: 0.033 | 0.732 ±0.020 | task_04/results |
| DeepSeek-V4-Flash-Base | 4 | cute_sm120_fp8 | 0.9090（invalid 0） | 0.8865 ±0.0026 | 24: 0.033 / 25: 0.067 | 0.714 ±0.020 | task_04/results |
| DeepSeek-V4-Flash-Base | 4 | cute_sm120_mxfp8_128 | 0.9037（invalid 0） | 0.8859 ±0.0026 | 24: 0.000 / 25: 0.000 | 0.744 ±0.020 | task_04/results |
| DeepSeek-V4-Flash-Base | 4 | cute_sm120_mxfp8_32 | 0.9060（invalid 0.001） | 0.8864 ±0.0026 | 24: 0.000 / 25: 0.067 | 0.742 ±0.020 | task_04/results |
| DeepSeek-V4-Flash-Base | 4 | deep_gemm_mxfp8_32 | 0.9030（invalid 0） | 0.8855 ±0.0026 | 24: 0.033 / 25: 0.033 | 0.738 ±0.020 | task_04/results |

*397B MBPP 全列 0.010-0.020：backend 无关的系统性坍塌（三 backend 一致）→ 模型 × completion
prompt 格式问题，非 kernel；结论表中该行不用于列间比较，归因待查（疑 397B 对 3-shot
completion 风格输出长推理文本致 pass@1 失败）。

## 测试条件（共通）

| 项 | 值 |
|---|---|
| 硬件 | 1×RTX PRO 6000 Blackwell server edition（多卡另记） |
| vLLM | `cute_sm120_precision_internal` @ dcf4072 base + 本分支改动；FI fork `sm120_moe_gemm_fp8_internal` |
| serve | `--max-model-len 4096`（AIME run 专用 36864，Q-A lock）+ `--moe-backend=<列>`；v2 加 `--enable-prefix-caching`；H20 行另加 `--gdn-prefill-backend triton`（FI sm90 GDN 编译broken + baseline 纯净性）；DSv4 另加 `--kv-cache-dtype fp8_ds_mla` |
| GSM8K | vLLM `gsm8k_eval.py`，1319 全量，5-shot，temp=0，seed=42，max_tokens=256 |
| MMLU | lm-eval 0.4.12 local-completions，全集 14042，5-shot，seed=42 |
| AIME | lm-eval `aime24,aime25`（各 30 题，0-shot，greedy，max_gen_toks=32768） |
| MBPP | lm-eval `mbpp`（500 题，3-shot 内置，pass@1，`--confirm_run_unsafe_code`） |

## 单层 parity（kernel 正确性 gate，非 e2e）

| 列 | 对拍 | 结果 | 证据 |
|---|---|---|---|
| cute_sm120_fp8 | vs triton，24 cells | PASS（~7e-4） | task_02/results/cute_fp8_vs_triton.csv |
| cute 3a/3b | vs eager fp32 reference，20 cells | PASS（2.3e-5~7.9e-5） | task_03/results/cute_mxfp8_parity.csv |
| deep_gemm_mxfp8_32 | vs eager fp32 reference，10 cells | PASS（3.4e-6~3.9e-6） | task_04/results/dg_mxfp8_parity.csv |
