# task_03 Result — cute MXFP8 3a/3b 接入（cute_sm120_mxfp8_128 / _32）

日期 2026-07-14。分支 `cute_sm120_precision_internal`。数据审计：全部数字已回源核对
（csv/json/log/evidence 行，见各行链接）。

## 结论

1. **接入完成且正确**：FI `moe_gemm_mxfp8_nt_groupwise` 以两个 explicit-only backend
   （`--moe-backend=cute_sm120_mxfp8_128` / `_32`）注册进 vLLM；单层 parity 20/20
   PASS（calc_diff 2.3e-5 ~ 7.9e-5，gate 1e-3 余量 >10×）。
2. **e2e 精度与既有列同带**：GSM8K 0.786/0.788、MMLU 0.846/0.847，无 backend 级异常
   （invalid 均 0）；granK 32 vs 128 差 ≤0.2pp（噪声内）——**UE8M0 K 粒度本身不是精度变量**。
3. 与五/六列矩阵的横向比较（同 recipe kernel 对照、UE8M0 效应分解）在
   [task_04/result.md] 统一给出；本 task 只对 3a/3b 自身数据负责。

## 单层 parity（gate < 1e-3）

| granK | cells | calc_diff 区间 | verdict |
|---|---|---|---|
| 128 | 10（M∈{1,7,33,128,1024} × topk/E {6/32, 8/128}） | 3.1e-5 ~ 7.8e-5 | 20/20 ✓ PASS |
| 32 | 10（同 shapes） | 2.3e-5 ~ 7.9e-5 | 同上 |

Reference = eager fp32 pipeline（镜像每个量化边界：act 1×gran UE8M0、weight requant 后
dequant、GEMM1 bf16 写回）。证据：[results/cute_mxfp8_parity.csv](results/cute_mxfp8_parity.csv)。
初版 0/20 失败的 root cause（FI kernel b_scale MN-major 存储 contract，shape check 挡不住
存储序错误）与修复记 [memory/findings.md](../memory/findings.md) `## task_03`。

## e2e（全量）

| Backend | Recipe | GSM8K 1319 | MMLU 14042 | 证据 |
|---|---|---|---|---|
| cute_sm120_mxfp8_128（3a） | UE8M0 GranK=128（DG-convention） | **0.7862**（invalid 0） | **0.8463** ±0.0029 | [gsm8k json](results/gsm8k_cute_sm120_mxfp8_128_n1319.json)、[mmlu log](results/mmlu_cute_sm120_mxfp8_128.log) |
| cute_sm120_mxfp8_32（3b） | UE8M0 GranK=32（OCP） | **0.7885**（invalid 0） | **0.8472** ±0.0029 | [gsm8k json](results/gsm8k_cute_sm120_mxfp8_32_n1319.json)、[mmlu log](results/mmlu_cute_sm120_mxfp8_32.log) |

Backend 证据行：`Using CUTE_SM120_MXFP8_{128,32} Fp8 MoE backend`
（[evidence_*.txt](results/)，每 run 自动摘录）。

## 测试条件

| 项 | 值 |
|---|---|
| 模型 | Qwen3.5-35B-A3B-FP8（fp8-block checkpoint，act 1×128 / weight 128×128 float scale） |
| 硬件 | 1×RTX PRO 6000 Blackwell server edition（smc521ge-0036，allocation sc-3092583） |
| 软件 | vLLM `cute_sm120_precision_internal`（base dcf4072）+ FI fork `sm120_moe_gemm_fp8_internal`；CUDA 13.3 |
| serve（v2） | `--max-model-len 4096 --enable-prefix-caching --moe-backend=<backend>`，其余默认（cudagraph/compile 开） |
| weight requant（load 期） | `requant_weight_for_cute_mxfp8`：3a = (128,128) UE8M0 requant → N 广播 per-token → int32 pack；3b = dequant → per-row (1,32) UE8M0 → pack；存储 MN-major |
| act quant（runtime） | GEMM1 = `per_token_group_quant_fp8(use_ue8m0)` + `ep_scatter(pack_ue8m0)`；GEMM2 = fused silu+quant (colmajor, ue8m0) + eager pack |
| GSM8K | vLLM `gsm8k_eval.py`：1319 全量、5-shot、temp=0、seed=42、max_tokens=256 |
| MMLU | lm-eval 0.4.12 local-completions：全集 14042、5-shot、seed=42、num_concurrent=128 |

## 主要代码改动（本 task）

- `experts/cute_sm120_moe.py`：`CuteMxfp8Experts` + Gran128/32 子类、
  `requant_weight_for_cute_mxfp8`、packed a_scale 打包/repack helpers、
  **b_scale MN-major 存储修复**（parity 0/20 root cause）
- `oracle/fp8.py` + `config/kernel.py`：双 backend 注册（enum/kernel_cls/map/convert 白名单/Literal）
- tests：`test_cute_mxfp8_parity.py`（含 size-1 维 stride 陷阱修复）、`test_requant_bitequal.py`、
  `run_{gsm8k,mmlu}_v2.sh`
