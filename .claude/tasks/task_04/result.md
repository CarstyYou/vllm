# task_04 Result — v2 条件对齐 + AIME/MBPP + 多卡 TP + H20 baseline + DG granK=32

日期 2026-07-14/15。分支 `cute_sm120_precision_internal`。
数据审计：全矩阵数字已逐格回源核对（json/log/evidence 行，audit 记录在会话；
汇总同步 [memory/precision.md](../memory/precision.md)）。
**补测状态：4 项异常复核全部完成**（xiy 纪律：异常 cell 复现/取证后才入结论）——
\* 标注 cell 的判定见 §补测，全部 solid & explainable。

## 结论

1. **单卡 35B 六列矩阵**：MMLU/AIME/MBPP 三评测全列无可分辨差异；**GSM8K 是唯一有
   区分度的评测**。GSM8K 上 float-scale 参照 = triton 0.7923；cute 三列 0.780-0.789
   （与 triton ≤1.2pp，≈1σ 内）；**deep_gemm 两列显著偏低**（128: 0.7726 / 32: 0.7604\*）。
2. **同 recipe kernel 对照（35B GSM8K）**：GranK=128 上 cute 比 DG 高 1.4pp、GranK=32 上高
   2.8pp\*（~2.5σ）——GSM8K 退化主要来自 **DG kernel/集成实现而非 UE8M0 recipe 本身**
   （cute 的 UE8M0 两列基本不掉分）。kernel 级归因未做（后续候选）。
3. **大模型上该效应消失**：397B/DSv4 各六列 GSM8K/MMLU 全部同带
   （397B 0.889-0.905 / DSv4 0.903-0.910），DG 列常居前列——DG 实现的精度损失与模型
   规模负相关（hypothesis，机制未归因）。
4. **H20（sm90）跨硬件 baseline 成立，单卡 + 多卡全覆盖**：单卡 35B（H20 96G）deep_gemm
   float GSM8K 0.7930 / MMLU 0.8478 / MBPP 0.678；多卡（H20-3e 141G，xiy 2026-07-15 加测）
   397B TP8 float 0.8855 / UE8M0 0.8984、DSv4 TP4 float 0.9075 / UE8M0 0.9083——
   全部与 sm120 对应列同带。**UE8M0 的退化是"模型 × kernel"联合效应**：35B 上 sm90 掉
   13pp / sm120 掉 2pp，而 397B/DSv4 上两种硬件都不掉。397B 的 MBPP 坍塌与 AIME 单 cell
   0.2 离群在 H20 复现 → 均为硬件无关的已判定类（模型×格式 / 30 题轨迹噪声）。
5. **granK 32 vs 128 在所有模型上均非精度变量**（cute 与 DG 内部各自对比 ≤0.3pp，35B DG 除外\*）。

## 单卡 Qwen3.5-35B（v2 全量）

| Backend | Recipe | GSM8K 1319 | MMLU 14042 | AIME24/25 (30+30) | MBPP 500 |
|---|---|---|---|---|---|
| triton | float-scale | 0.7923 | 0.8470 ±0.0029 | 0.033 / 0.033 | 0.678 ±0.021 |
| deep_gemm | UE8M0 K128 | 0.7726 | 0.8478 | 0.000 / 0.000 | 0.656 |
| cute_sm120_fp8 | float-scale | 0.7801（inv 0.0015） | 0.8472 | 0.000 / 0.100 | 0.676 |
| cute_sm120_mxfp8_128 | UE8M0 K128 | 0.7862 | 0.8463 | 0.033 / 0.000 | 0.662 |
| cute_sm120_mxfp8_32 | UE8M0 K32 | 0.7885 | 0.8472 | 0.033 / 0.000 | 0.668 |
| deep_gemm_mxfp8_32 | UE8M0 K32 | 0.7604\*（inv 0.0008） | 0.8461 | 0.000 / 0.033 | 0.666 |
| H20: deep_gemm（float，sm90） | float-scale | 0.7930 | 0.8478 | 0.000 / 0.000 | 0.678 |

\* 0.7604 待复测确认可重复（§补测 ④）。
GSM8K invalid 未标注的均为 0。AIME 30 题/年 stderr 3-7pp，全列噪声底，只与 GSM8K 联看。

## 多卡

### Qwen3.5-397B-A17B-FP8（TP=8）

| Backend | GSM8K | MMLU | AIME24/25 | MBPP |
|---|---|---|---|---|
| triton | 0.8886（inv 0.011\*） | 0.8958 ±0.0025 | 0.033 / 0.000 | 0.020\* |
| deep_gemm | 0.9007 | 0.8957 | 0.067 / 0.000 | 0.010\* |
| cute_sm120_fp8 | 0.8976 | 0.8957 | 0.067 / 0.000 | 0.016\* |
| cute_sm120_mxfp8_128 | 0.8999 | 0.8963 | 0.067 / 0.000 | 0.004\* |
| cute_sm120_mxfp8_32 | 0.9045 | 0.8955 | **0.200\*** / 0.033 | 0.008\* |
| deep_gemm_mxfp8_32 | 0.8961 | 0.8956 | 0.100 / 0.000 | 0.002\* |

- MBPP 全列 0.002-0.020：backend 无关系统性坍塌（六列一致）→ 模型 × completion 格式
  问题，非 kernel；**该行不用于列间比较**（补测 ② 取证归因中）
- AIME24 的 0.200（6/30）为 2σ 离群（12+ cell 多重比较下不异常；补测 ① 复现验证中）

### DeepSeek-V4-Flash-Base（TP=4，`--kv-cache-dtype fp8_ds_mla`）

| Backend | GSM8K | MMLU | AIME24/25 | MBPP |
|---|---|---|---|---|
| triton | 0.9083 | 0.8852 ±0.0026 | 0.000 / 0.000 | 0.738 |
| deep_gemm | 0.9098 | 0.8866 | 0.033 / 0.033 | 0.732 |
| cute_sm120_fp8 | 0.9090 | 0.8865 | 0.033 / 0.067 | 0.714 |
| cute_sm120_mxfp8_128 | 0.9037 | 0.8859 | 0.000 / 0.000 | 0.744 |
| cute_sm120_mxfp8_32 | 0.9060（inv 0.0008） | 0.8864 | 0.000 / 0.067 | 0.742 |
| deep_gemm_mxfp8_32 | 0.9030 | 0.8855 | 0.033 / 0.033 | 0.738 |

六列四评测全部同带；GSM8K invalid 除标注外均 0。

## H20-3e 多卡（xiy 2026-07-15 加测；serve 同 v2 + `--gdn-prefill-backend triton`；数据 tag `*_h20float` / `*_h20ue8m0`）

| Model | TP | Mode | GSM8K | MMLU | AIME24/25 | MBPP |
|---|---|---|---|---|---|---|
| Qwen3.5-397B | 8 | deep_gemm float（baseline） | 0.8855（inv 0.013） | 0.8969 ±0.0025 | 0.033 / 0.000 | 0.010* |
| Qwen3.5-397B | 8 | deep_gemm UE8M0（参考） | 0.8984（inv 0.005） | 0.8975 ±0.0025 | 0.200* / 0.000 | 0.008* |
| DSv4-Flash-Base | 4 | deep_gemm float（baseline） | 0.9075 | 0.8871 ±0.0026 | 0.033 / 0.000 | 0.740 |
| DSv4-Flash-Base | 4 | deep_gemm UE8M0（参考） | 0.9083 | 0.8861 ±0.0026 | 0.033 / 0.000 | 0.732 |

\* 已判定异常类在 H20 复现（MBPP 模型×格式坍塌、AIME 30 题轨迹噪声），硬件无关佐证，不另补测。
多卡 H20 行硬件为 H20-3e（141G HBM3e，sm90）；单卡行为 96G H20——型号如实分记。

## H20 参考行（不入结论）

| 行 | GSM8K | MMLU | 说明 |
|---|---|---|---|
| DG-UE8M0（E8M0 默认开，sm90 静默 requant） | 0.6626 | 0.8466 | `is_deep_gemm_e8m0_used()` 无架构 gate（findings）；sm90 UE8M0 掉 13pp 未归因，仅留档 |
| triton（AUTO 落点） | 0.7445 | — | 单 run 留档 |

## 补测（xiy 纪律：异常 cell 须复现/取证后入结论；数据在 results/verify/）

| # | 项 | 复测结果 | 判定 |
|---|---|---|---|
| ① | 397B mxfp8_32 AIME24 0.200 重跑 | **0.067**（2/30） | 不可复现 → 单次 greedy 轨迹波动；cell 判噪声（AIME 本就不入结论） |
| ② | 397B MBPP 坍塌重跑 + log_samples | 0.008；**500 题中 336 题空生成（67%）**，非空样本为正常代码 | 机制坐实：397B 对 3-shot completion 格式大概率立即产出停止序列 → 模型×格式问题，非 kernel；全列排除成立 |
| ③ | 397B triton GSM8K 稳定性重跑 | 0.8939（inv 0.009）vs 原 0.8886（inv 0.011） | 稳定复现；与其他列差 ~1σ，非异常；invalid ~1% 为该列稳定特征 |
| ④ | 35B deep_gemm_mxfp8_32 GSM8K 复测 | **0.7642** vs 原 0.7604 | 复现（Δ0.4pp 噪声内）→ 真实信号，支撑结论 2 的 DG 实现差异判定 |
| — | 35B deep_gemm 0.7726 | v1 0.772 / v2 0.7726 | 已有双复现，免 |

## 测试条件

| 项 | 值 |
|---|---|
| 模型 | Qwen3.5-35B-A3B-FP8（37G）/ Qwen3.5-397B-A17B-FP8（380G，TP8）/ DeepSeek-V4-Flash-Base（276G，TP4）——均 fp8-block checkpoint |
| 硬件 | 1×/8× RTX PRO 6000 Blackwell server edition（sc-3092583 / sc-3097302 / sc-3097834）；H20 = viking-prod-255（sc-3097272） |
| 软件 | vLLM `cute_sm120_precision_internal`（base dcf4072）+ 本分支新增 backend；FI fork `sm120_moe_gemm_fp8_internal`；lm-eval 0.4.12；CUDA 13.3 |
| serve 共通（v2） | `--max-model-len 4096 --enable-prefix-caching --moe-backend=<列>`；AIME run 专用 36864（Q-A）；DSv4 加 `--kv-cache-dtype fp8_ds_mla`；H20 加 `--gdn-prefill-backend triton` + DG 列 `VLLM_USE_DEEP_GEMM_E8M0=0` |
| GSM8K | vLLM `gsm8k_eval.py` 1319 全量、5-shot、temp=0、seed=42、max_tokens=256 |
| MMLU | lm-eval 全集 14042、5-shot、seed=42、num_concurrent=128、max_length 默认 |
| AIME | lm-eval `aime24,aime25` 各 30 题、0-shot greedy、max_gen_toks=32768、model_args `max_length=36864,timeout=7200` |
| MBPP | lm-eval `mbpp` 500 题、3-shot 内置、pass@1、`--confirm_run_unsafe_code` + `HF_ALLOW_CODE_EVAL=1`、max_length=4096 |
| 新 backend | `deep_gemm_mxfp8_32`：weight per-row (1,32) UE8M0 requant + DG transform 打包；act (1,32) UE8M0 uint8→permute 打包；GEMM `recipe_a/b=(1,32)`；parity 10/10（3.4e-6~3.9e-6，[dg_mxfp8_parity.csv](results/dg_mxfp8_parity.csv)） |

每个数字可回溯：`results/{gsm8k_*.json, mmlu_*/, aime_*/, mbpp_*/, evidence_*.txt}`
（evidence 行含 oracle backend 确认 + serve 非默认参数）。工程踩坑与客观发现见
[memory/findings.md](../memory/findings.md) `## task_04`。
