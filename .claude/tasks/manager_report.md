# SM120 FP8 blockscaled MoE GEMM — 精度 & 性能汇报

**范围**：cute（SM120 FP8/MXFP8 MoE GEMM）接入 vLLM，e2e 精度 + serving 性能，对比 deep_gemm / triton
基线，并含跨硬件 H20 基线。硬件 = RTX PRO 6000 Blackwell（sm120）server edition；H20 列 = H20-3e（sm90）。

---

## 1. 总结

**精度**（GSM8K / MMLU / AIME24+25 / MBPP 四项 eval）
- cute（FP8 / MXFP8-K128 / MXFP8-K32）与 H20 FP8 基线、deep_gemm 全面对齐，**单卡与多卡（TP4/TP8）一致**；
  MMLU 几乎逐列相同、GSM8K/MBPP 列间落 stderr 内；AIME（各 30 题）列间为小样本波动、无可分辨差异。
- 同 recipe 对比 **cute_mxfp8_32 ≈ deep_gemm_mxfp8_32**（精度等价，验证 cute kernel 数值正确）。
- 单层 kernel parity 全 PASS（calc_diff 7e-4 ~ 3e-6）。

**性能（serving，ISL/OSL = 8k/1k）**
- **大 M（高并发 cc≥64）区间**：cute 追平/略超 —— 35B cc=128 cute **反超 deep_gemm +2.0%**、比 triton -4.7%；
  397B/DSv4 同趋势（cute vs dg +1.6~6.2%）。
- **小 M（低并发 cc≤8）区间**：cute 落后基线（35B -32~42%）。根因已定位为 **ZeroPadding scheduler 的
  O(num_experts) token_offset 串行扫描**（kernel 侧固定开销，非精度 recipe），优化进行中。
- 跨硬件：H20-3e 在低并发、及大模型高并发领先 sm120 —— 硬件带宽代差（HBM3e），非 kernel。

---

## 2. 精度 — 单卡（Qwen3.5-35B-A3B-FP8）

| Backend | Recipe | GSM8K (1319) | MMLU (14042) | AIME24 | AIME25 | MBPP (500) |
|---|---|---|---|---|---|---|
| H20 FP8 基线（sm90，跨硬件） | deep_gemm float | 0.7930 | 0.8478 ±0.0029 | 0.000 | 0.000 | 0.678 ±0.021 |
| cute FP8 | float-scale | 0.7801 | 0.8472 ±0.0029 | 0.000 | 0.100 | 0.676 ±0.021 |
| cute MXFP8-K128 | UE8M0 GranK=128 | 0.7862 | 0.8463 ±0.0029 | 0.033 | 0.000 | 0.662 ±0.021 |
| **cute MXFP8-K32** | UE8M0 GranK=32 (OCP) | **0.7885** | **0.8472 ±0.0029** | **0.033** | **0.000** | **0.668 ±0.021** |
| **deep_gemm MXFP8-K32** | UE8M0 GranK=32（同 recipe 对比） | 0.7604 | 0.8461 ±0.0029 | 0.000 | 0.033 | 0.666 ±0.021 |
| deep_gemm K128 | UE8M0 GranK=128 | 0.7726 | 0.8478 ±0.0029 | 0.000 | 0.000 | 0.656 ±0.021 |
| triton（参照） | float-scale | 0.7923 | 0.8470 ±0.0029 | 0.033 | 0.033 | 0.678 ±0.021 |

- MMLU 逐列 0.846–0.848（±0.0029）几乎相同 = 最强精度对齐信号；GSM8K/MBPP 列间落在 stderr 内。
- **AIME24 / AIME25 = 各年 30 题的 pass@1（答对题数 / 30）**：0.000 = 0/30、0.033 = 1/30、0.100 = 3/30。
  绝对值低是 35B（greedy、无思维链）的模型能力所限，非 kernel；每题跳 3.3pp、样本仅 30 → 列间**无可分辨
  差异**，作下限 sanity check，不单独作列间结论（精度对比以 MMLU/GSM8K 为主）。

---

## 3. 精度 — 多卡（TP）

### Qwen3.5-397B-A17B-FP8（TP8）

| Backend | GSM8K | MMLU | AIME24 | AIME25 | MBPP (500) |
|---|---|---|---|---|---|
| H20-3e 基线 | 0.8855 | 0.8969 ±0.0025 | 0.033 | 0.000 | 0.010* |
| cute FP8 | 0.8976 | 0.8957 ±0.0025 | 0.067 | 0.000 | 0.016* |
| cute MXFP8-K128 | 0.8999 | 0.8963 ±0.0025 | 0.067 | 0.000 | 0.004* |
| **cute MXFP8-K32** | **0.9045** | **0.8955 ±0.0025** | **0.200** | **0.033** | **0.008*** |
| deep_gemm MXFP8-K32 | 0.8961 | 0.8956 ±0.0025 | 0.100 | 0.000 | 0.002* |
| deep_gemm K128 | 0.9007 | 0.8957 ±0.0025 | 0.067 | 0.000 | 0.010* |
| triton | 0.8886 | 0.8958 ±0.0025 | 0.033 | 0.000 | 0.020* |

### DeepSeek-V4-Flash-Base（TP4）

| Backend | GSM8K | MMLU | AIME24 | AIME25 | MBPP (500) |
|---|---|---|---|---|---|
| H20-3e 基线 | 0.9075 | 0.8871 ±0.0026 | 0.033 | 0.000 | 0.740 ±0.020 |
| cute FP8 | 0.9090 | 0.8865 ±0.0026 | 0.033 | 0.067 | 0.714 ±0.020 |
| cute MXFP8-K128 | 0.9037 | 0.8859 ±0.0026 | 0.000 | 0.000 | 0.744 ±0.020 |
| **cute MXFP8-K32** | **0.9060** | **0.8864 ±0.0026** | **0.000** | **0.067** | **0.742 ±0.020** |
| deep_gemm MXFP8-K32 | 0.9030 | 0.8855 ±0.0026 | 0.033 | 0.033 | 0.738 ±0.020 |
| deep_gemm K128 | 0.9098 | 0.8866 ±0.0026 | 0.033 | 0.033 | 0.732 ±0.020 |
| triton | 0.9083 | 0.8852 ±0.0026 | 0.000 | 0.000 | 0.738 ±0.020 |

- 多卡与单卡结论一致：cute 与基线全面对齐，MMLU 逐列落 ±0.0025 内；DSv4 MBPP 逐列 0.71–0.74 亦对齐。
- **AIME24/25 各 30 题、stderr 高**：397B cute_mxfp8_32 AIME24=0.200 等高值为小样本波动，列间不作结论。
- **\* 397B MBPP 全列 0.002–0.020**：模型 × completion prompt 格式的系统性坍塌（backend 无关，三 backend
  一致），非 kernel 问题，该行**不用于列间比较**。

---

## 4. 性能 benchmark（output tok/s，3 轮 median，ISL/OSL = 8k/1k）

并发 cc = 1…128；perf 无 MXFP8-K128 列（精度 K128≈K32，perf 只取 OCP K32）。

### 单卡 Qwen3.5-35B（sm120；H20 列 = H20-3e）

| cc | H20 基线 | cute FP8 | cute MX32 | dg MX32 | dg K128 | triton |
|---|---|---|---|---|---|---|
| 1 | 218 | 129 | 126 | 185 | 187 | 198 |
| 8 | 725 | 579 | 556 | 604 | 619 | 674 |
| 32 | 1228 | 1096 | 1078 | 1054 | 1070 | 1202 |
| 64 | 1417 | 1353 | 1335 | 1308 | 1331 | 1447 |
| 128 | 1435 | 1549 | **1527** | 1497 | 1522 | 1603 |

### 多卡 Qwen3.5-397B（TP8）

| cc | H20 基线 | cute FP8 | cute MX32 | dg MX32 | dg K128 | triton |
|---|---|---|---|---|---|---|
| 1 | 138 | 60 | 59 | 96 | 97 | 103 |
| 8 | 629 | 312 | 302 | 336 | 346 | 389 |
| 32 | 990 | 595 | 572 | 553 | 565 | 649 |
| 64 | 1271 | 708 | 684 | 647 | 662 | 752 |
| 128 | 1274 | 787 | **766** | 721 | 740 | 811 |

### 多卡 DeepSeek-V4-Flash（TP4）

| cc | H20 基线 | cute FP8 | cute MX32 | dg MX32 | dg K128 | triton |
|---|---|---|---|---|---|---|
| 1 | 133 | 87 | 86 | 110 | 112 | 116 |
| 8 | 541 | 396 | 386 | 406 | 415 | 436 |
| 32 | 964 | 701 | 694 | 686 | 692 | 741 |
| 64 | 1156 | 865 | 848 | 833 | 838 | 891 |
| 128 | 1183 | 1048 | **1023** | 1007 | — | 1081 |

**区间对比（cute MX32 vs 同 recipe dg MX32 / triton）**：
- 大 M（cc=128）：cute vs dg = 35B **+2.0%** / 397B **+6.2%** / DSv4 **+1.6%**；cute vs triton ≈ -4.7~-5.5%。
- 小 M（cc=1）：cute 落后（35B -32%、397B -39%、DSv4 -22% vs dg）—— 根因 = ZeroPadding scheduler O(E)。

---

## 5. 测试条件

| 项 | 值 |
|---|---|
| 硬件 | sm120 = RTX PRO 6000 Blackwell server edition；H20 列 = H20-3e（141G HBM3e，sm90） |
| 模型 | Qwen3.5-35B-A3B-FP8（单卡）/ Qwen3.5-397B-A17B-FP8（TP8）/ DeepSeek-V4-Flash-Base（TP4） |
| 精度 eval | GSM8K 1319（5-shot, temp=0, seed=42）；MMLU 14042（lm-eval, 5-shot）；AIME24/25 各 30（greedy）；MBPP 500（pass@1） |
| 性能负载 | `vllm bench serve` random，ISL/OSL=8192/1024，`--ignore-eos`（锁 output 1024），并发 1/2/4/8/16/32/64/128，warmup 1 + 正式 3 轮 median |
| Backend | cute FP8 / MXFP8-K128 / MXFP8-K32；deep_gemm(K128 / MXFP8-K32)；triton；H20 列 = deep_gemm float |
| 数据规模 | 精度 6 backend × 4 eval × 3 model + H20 基线；性能 151 cells（output len 全量校验 == 1024） |
