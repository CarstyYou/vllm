# task_06 followup — cc=128 全链路 kernel breakdown（为什么 GEMM 不慢，cute 却 e2e 慢于 triton）

承接 task_06（cc=1 归因 = ZeroPadding scheduler O(E)；cc=128 三臂 MoE GEMM 收敛 ~385μs）。
本 followup 复用 task_06 的 **cc=128 nsys traces**（`trace_{cutemx32,dgmx32,triton}_cc128.nsys-rep`），
把**全部 kernel**（非只 MoE GEMM）按 demangled 分组，回答：cc=128 GEMM 不慢，为什么 cute e2e 仍慢于 triton。

## 结论（先行）

1. **cc=128 GEMM 不是瓶颈**：cute MoE GEMM ≈ dg ≈ triton fused GEMM（归一后 235 / 237 / 246 μs/anchor）。
2. **三臂皆 chain，融合结构相近**：都是 `GEMM(FC1) → 独立 SiLU → 独立 quant → GEMM(FC2) → 独立 reduce`。
   **SiLU 在三家都不融进 FC1**（各有独立 act kernel）。triton 的 `fused_moe_kernel` 融的是「单 GEMM 内部的
   gather + weight-dequant + GEMM」，不是把整个 MoE 熔成一个 kernel。
3. **cute 唯一的 outlier = `FillFunctor<int>`（int-buffer zero-fill）**：归一 **234 μs/anchor、146K 次调用，
   是 dg/triton 的 ~10×**（dg/triton 仅 24μs）。这是 cute 的 **vLLM 集成层**每次 forward 反复清零 int 缓冲
   （scale-storage / m_indptr / offset）的纯 setup 开销，**dg/triton 都没有**。**非 cute kernel 本身问题，是集成层 buffer 管理。**
4. **可优化**：cute 集成路径预分配 + 复用这些 int 缓冲，消除 per-forward 的 146K 次 fill。

## 方法

- 数据：task_06 cc=128 三 trace（35B 单卡 sm120，nsys cuda-13.3，`--ignore-eos`，cudagraph 生产路径）。
- 提取：`veloq stats <trace> --group-by demangled --device 0`（全 kernel）。
- **窗口归一化**（三 trace 抓的步数不同）：用**共享 dense kernel** `cutlass_3x_gemm_fp8_blockwise`（非 MoE 的
  qkv/o 投影，三臂逐步一致）的调用次数当锚点 —— cute 46708 / dg 45341 / triton 44160。每 kernel 的 total 除以
  各自 anchor 次数 → per-anchor μs，消除窗口差。

## 全 kernel breakdown（归一 per-anchor）

| 环节 | cute_mx32 | dg_mx32 | triton | 说明 |
|---|---|---|---|---|
| MoE GEMM（FC1+FC2） | 235 μs | 237 μs | 246 μs | cute/dg 为 grouped GEMM；triton `fused_moe_kernel` 含 gather+dequant |
| SiLU+mul（act） | 31 μs | 38 μs | 25 μs | 三家皆独立 kernel（cute/dg 与 quant 合并；triton `act_and_mul`）|
| per_token_group_quant | 39 μs | 39 μs | 41 μs | FC2 输入量化，三家等价 |
| **FillFunctor\<int\>** | **234 μs** | **24 μs** | **24 μs** | **cute 独有的 10× 开销** |
| FillFunctor 调用次数/anchor | **3.14** | 0.64 | 0.35 | cute 每单位 work 多 ~5× int-fill |

（原始：cute FillFunctor 10935ms/146599 次；dg 1079ms/29224；triton 1071ms/15351。MoE GEMM 原始 cute 10988ms、
dg 10764ms、triton fused 10857ms —— 印证 task_06「GEMM 收敛」。）

## 关键归因：`FillFunctor<int>`（cute-specific）

- **是什么**：`void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, ...>` = int 张量
  zero-fill/填充（`torch.zeros` / `.zero_()` on int32）。**不是** quant/scale 数学。
- **归因铁证**：三次 run **唯一变量是 `--moe-backend`**，非 MoE 部分（flash attention / GDN chunk / RMSNorm /
  dense linear = anchor）三臂逐 kernel 一致。故 FillFunctor 的 10× 超额**只能来自 cute 的 MoE backend**。
- **源头**（cute vLLM 集成 `experts/cute_sm120_moe.py`）：每次 forward 构造 ZeroPadding 元数据时的 int32
  `torch.zeros`——scale-storage 重排缓冲 `_repack_packed_a_scale_for_fi`（`(k_align, m_padded)` int32）、
  `m_indptr` / `expert_ids` / offset 等，逐层逐步反复清零。cc=128 下 m_padded 大 → 单次 fill 也不小，
  叠加 146K 次 → 归一 234μs/anchor。

## 融合结构对比（三臂皆 chain）

vLLM `fused_experts_impl`（`fused_moe.py:1730-1791`）主流程，三 backend 共享：
```
FC1 GEMM → SiLU+mul(独立) → quant(独立) → FC2 GEMM → moe_sum(独立)
```
- triton `fused_moe_kernel`（每层 2 次 = FC1/FC2）融「gather(sorted_token_ids) + weight-dequant + GEMM」；
  SiLU / 中间 quant / reduce 均独立。
- cute / dg 为 grouped GEMM（m_indices/token_offset 做 grouping，dequant 融入 GEMM）；SiLU+quant 各有一个融合
  kernel（`silu_mul_(per_token_group_)quant`）；reduce/gather 独立。
- **即融合深度三家相近**，差距不在此，而在 cute 的 FillFunctor。

## 可优化

- cute 集成层：**预分配 + 复用** scale-storage / m_indptr / offset 的 int 缓冲，避免每 forward `torch.zeros`；
  或把重排/清零并入已有 kernel。消除后 cute 的 per-anchor 开销应从 FillFunctor 234μs 降到 ~24μs 量级（对齐 dg/triton）。

## 诚实边界

- **归一化基于共享 anchor**，消除了窗口差；FillFunctor 的 10× 是窗口无关比值，稳。
- **GPU-kernel-time 组成清楚，但不 1:1 映射 e2e**：cute e2e 仅 -4.7% vs triton（1527 vs 1603），小于 FillFunctor
  的 GPU-time 占比 —— 146K 个 tiny kernel 部分与其他 kernel **overlap**，未全额传导到 e2e。FillFunctor 是 cute
  最大的 backend-specific GPU-time 浪费 + launch 开销，但精确 e2e 收益需消除后实测。
- `FillFunctor<int>` 的具体 op 由 functor 名（`FillFunctor<int>`）+ 唯一变量归因确定；未逐 call site 做 NVTX
  关联到具体 `torch.zeros` 行（如需可加 nsys NVTX 或 correlation）。

## Artifacts

- traces：`results/trace_{cutemx32,dgmx32,triton}_cc128.nsys-rep`（+ `.veloq` 缓存，gitignore）。
- 提取命令：`veloq stats <trace> --group-by demangled --device 0`。
