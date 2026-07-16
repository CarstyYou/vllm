# task_05 Result — e2e serving 性能对比（cute vs baselines）

日期 2026-07-15。分支 `cute_sm120_precision_internal`。
数据审计：全部 cell 取 3 正式轮 median，`--ignore-eos` 保证 output len==1024（summarize.py 校验，
全 cell 通过）。汇总 CSV = [results/exp05_summary.csv](results/exp05_summary.csv)，逐 cell 可回溯到
`bench_<mtag>_cc<cc>_{r1,r2,r3}.json`。

## 结论

1. **cute 低并发劣势 = kernel 特有，非 recipe**：cc=1 cute_fp8/cute_mxfp8_32 明显慢于 baseline
   （35B -35%、397B -42%、DSv4 -26%）；但**同 K32 recipe 的 dg_mxfp8_32 ≈ triton**
   （35B cc1: cute_mx32 126 vs dg_mx32 185 vs triton 198）——证明小 batch 惩罚来自 cute kernel
   实现（launch/占用效率），不是 UE8M0/K32 recipe。归因移交 [task_06] nsys（cc=1/128 profile）。
2. **高并发吞吐区间惩罚收敛**：随并发上升 cute 逼近 baseline —— 35B cc=128 cute **反超**
   triton 之外全列（1549/1527，triton 1603，-3.4%/-4.7%）；397B/DSv4 cc=128 cute 与 triton
   差 -3~6%。**同 recipe cute vs dg：高并发 cute ≥ dg**（35B cc128 cute_mx32 1527 > dg_mx32 1497；
   DSv4 cc128 1023 > 1007）——精度不输（task_04）+ 高并发吞吐不输，劣势仅在低并发延迟区间。
3. **跨硬件 H20-3e baseline**：低并发 H20-3e 最快（HBM3e 带宽）；大模型（397B/DSv4）高并发
   H20-3e 仍显著领先 sm120（397B cc128 H20 1274 vs sm120 triton 811）——硬件代差，非 kernel。
4. **已知失败点**：dg_ue8m0 DSv4 TP4 cc=128 可复现 CUDA IMA（8k×512 高并发，findings 记录），
   非 cute 列、非数据缺口。

## 数据表（output tok/s median of 3 rounds；完整 TTFT/TPOT 见 [exp05_summary.csv](results/exp05_summary.csv)）

列顺序（聚类，同 precision.md）：H20 baseline → cute(fp8/mx32) → dg(mx32/ue8m0) → triton。
（perf 无 cute_mxfp8_128 列——精度 128/32 等价，perf 只取 OCP K32。）

### 单卡 Qwen3.5-35B（sm120 6K Pro；H20 列 = H20-3e）

| cc | H20 base | cute_fp8 | cute_mx32 | dg_mx32 | dg_ue8m0 | triton |
|---|---|---|---|---|---|---|
| 1 | 218 | 129 | 126 | 185 | 187 | 198 |
| 2 | 370 | 217 | 213 | 271 | 276 | 294 |
| 4 | 561 | 372 | 364 | 430 | 438 | 477 |
| 8 | 725 | 579 | 556 | 604 | 619 | 674 |
| 16 | 962 | 819 | 797 | 829 | 841 | 917 |
| 32 | 1228 | 1096 | 1078 | 1054 | 1070 | 1202 |
| 64 | 1417 | 1353 | 1335 | 1308 | 1331 | 1447 |
| 128 | 1435 | 1549 | 1527 | 1497 | 1522 | 1603 |

### 多卡 Qwen3.5-397B TP8

| cc | H20 base | cute_fp8 | cute_mx32 | dg_mx32 | dg_ue8m0 | triton |
|---|---|---|---|---|---|---|
| 1 | 138 | 60 | 59 | 96 | 97 | 103 |
| 2 | 251 | 106 | 104 | 154 | 156 | 163 |
| 4 | 412 | 189 | 183 | 239 | 242 | 260 |
| 8 | 629 | 312 | 302 | 336 | 346 | 389 |
| 16 | 798 | 458 | 438 | 447 | 455 | 513 |
| 32 | 990 | 595 | 572 | 553 | 565 | 649 |
| 64 | 1271 | 708 | 684 | 647 | 662 | 752 |
| 128 | 1274 | 787 | 766 | 721 | 740 | 811 |

### 多卡 DeepSeek-V4-Flash TP4

| cc | H20 base | cute_fp8 | cute_mx32 | dg_mx32 | dg_ue8m0 | triton |
|---|---|---|---|---|---|---|
| 1 | 133 | 87 | 86 | 110 | 112 | 116 |
| 2 | 234 | 151 | 149 | 176 | 178 | 189 |
| 4 | 380 | 256 | 249 | 275 | 279 | 298 |
| 8 | 541 | 396 | 386 | 406 | 415 | 436 |
| 16 | 772 | 544 | 539 | 546 | 552 | 598 |
| 32 | 964 | 701 | 694 | 686 | 692 | 741 |
| 64 | 1156 | 865 | 848 | 833 | 838 | 891 |
| 128 | 1183 | 1048 | 1023 | 1007 | **IMA✗** | 1081 |

## 测试条件

| 项 | 值 |
|---|---|
| 模型 | Qwen3.5-35B-A3B-FP8（单卡）/ Qwen3.5-397B-A17B-FP8（TP8）/ DeepSeek-V4-Flash-Base（TP4） |
| 硬件 | sm120 = RTX PRO 6000 Blackwell server edition；H20 列 = H20-3e（141G HBM3e，sm90） |
| 负载 | ISL/OSL = 8192/1024（单场景）；并发 1/2/4/8/16/32/64/128（`--max-concurrency`） |
| Backend | triton / deep_gemm(UE8M0 K128) / cute_sm120_fp8 / cute_sm120_mxfp8_32 / deep_gemm_mxfp8_32；H20 列 = deep_gemm float（`E8M0=0`）。dg-float sm120 不支持（DG kernel 只吃 UE8M0 packed scale），故 sm120 无 dg-float 列 |
| serve | `--max-model-len 10240 --no-enable-prefix-caching`（perf 专用：关 prefix caching 防固定 seed 多轮 prefill 全命中）；cudagraph/compile 默认开；H20 列加 `--gdn-prefill-backend triton` + `VLLM_USE_DEEP_GEMM_E8M0=0`；DSv4 加 `--kv-cache-dtype fp8_ds_mla` |
| bench | `vllm bench serve` random dataset seed=42、`--ignore-eos`（锁 output=1024，apples-to-apple）；每 cell warmup 1 轮 + 正式 3 轮取 median；num_prompts=max(4×cc,16) |
| GDN 控制点 | sm120 列用 FI GDN、H20 列用 triton GDN；补 35B triton triton-GDN 单列 bound 跨硬件 GDN 路径差异（tag `tritongdnctl`） |

## 数据完成度

全部完成：35B 单卡 6 列 + GDN 控制点、397B TP8 5 列、DSv4 TP4 5 列、H20-3e 三列 baseline。
151 cells，summarize.py 校验 output len 全 == 1024（唯一失败点 dg_ue8m0 DSv4 cc128 已剔除）。

## Risks 命中

- dg-float sm120 不支持 → sm120 无该列（goal 指令：不支持则放弃）——已确认，非遗漏
- 单卡 1-GPU alloc 在共享节点不隔离 GPU → OOM → 改单 alloc 串行（findings 记录）
