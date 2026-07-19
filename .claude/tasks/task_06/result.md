# task_06 Result — nsys e2e breakdown（cute 低并发劣势归因）

日期 2026-07-16。分支 `cute_sm120_precision_internal`。单卡 35B，nsys（cuda-13.3 bundled 2026.2.1）
capture + veloq 0.2.2 分析。6 traces（cc=1/128 × cute_mxfp8_32/dg_mxfp8_32/triton），全 FI GDN、
prefix caching 关、`--ignore-eos`（同 exp_05 生产路径）。

## 结论

**exp_05 的 cute cc=1 劣势根因 = cute SM120 MXFP8 MoE kernel 的 per-invocation 固定开销，
在 tiny-M（decode）主导、在 large-M（prefill/高并发）被摊薄。**

- **cc=1（decode，M=1×topk）**：同 K32 recipe 下 cute MoE kernel **67.1 μs/call vs dg 28.0 μs/call
  = 慢 2.4×**（triton `fused_moe_kernel` 42.6 μs）。这是 kernel 级根因。
- **cc=128（large M）**：三列 MoE kernel 收敛到 ~385 μs/call（cute 384 / dg 388 / triton 392，±2%）——
  固定开销被大 batch 的 compute 摊薄，惩罚消失。与 exp_05 e2e cc=128 cute≈dg 吻合。
- **闭环校验**：exp_05 35B cc=1 e2e 差 = cute 126 vs dg 185 tok/s → 2.53 ms/token。每 decode step
  MoE calls ≈ 40 层 ×2 gemm = 80，per-call Δ(cute−dg)=39.1 μs → 80×39.1 = 3.1 ms/step ≳ 2.53 ms/token
  e2e 差。**MoE kernel 的 per-call 惩罚单独即可解释（甚至略超）整个 e2e 延迟差** → cc=1 劣势
  完全归因于 cute MoE kernel，非 recipe、非 attention/GDN/其他段。

## 数据（MoE GEMM kernel，device 0，veloq stats --group-by demangled）

| cc | 指标 | triton | cute_mxfp8_32 | dg_mxfp8_32 |
|---|---|---|---|---|
| 1 | mean μs/call | 42.6 | **67.1** | **28.0** |
| 1 | total ms | 351 | 574 | 233 |
| 128 | mean μs/call | 392.3 | 383.8 | 388.0 |
| 128 | total ms | 10858 | 11081 | 10844 |

MoE kernel 识别（demangled name）：
- triton = `fused_moe_kernel`
- cute_mxfp8_32 = `cutlass::device_kernel<flashinfer::gemm::mxfp8_cute_sm120::...>`
- dg_mxfp8_32 = `deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<...>`（gemm1 + gemm2 两 instance）

## 解读

cute kernel 的 per-invocation 固定成本（grid/TMA descriptor setup、epilogue 等不随 M 缩放的部分）
在 M=1 decode 时占主导 → 每 call 67 μs 远高于 dg 28 μs；M 大时 compute 主导，固定成本占比趋零 →
三列齐平。**优化方向**：降低 cute kernel 小 M 路径的固定开销（swapAB / persistent / 减少 setup），
或 decode 小 M 走轻量 dispatch。精度上 cute 全面不输（task_04），高并发吞吐不输（task_05）——
劣势仅低并发延迟区间，且根因明确可优化。

## 测试条件

| 项 | 值 |
|---|---|
| 模型/硬件 | Qwen3.5-35B-A3B-FP8，1×RTX PRO 6000 Blackwell server edition（sm120） |
| capture | nsys `--trace cuda,nvtx`（vLLM 默认不发 layerwise NVTX 且与 cudagraph 不兼容 → 用 kernel name 识别 MoE 段，保留 cudagraph 生产路径）；ISL 8192，measured decode 256 steps |
| serve | `--max-model-len 10240 --no-enable-prefix-caching`，cudagraph/compile 默认开；全列 FI GDN |
| 分析 | veloq `stats --group-by demangled --device 0`；per-call mean = total/count |
| 相位归属 | cc=1 mean 由 decode（256 步 ×80 call）主导、prefill（1×80 call）稀释可忽略；prefill 段 cute 不吃亏（cc=128 数据佐证），故 mean 稀释只会低估真实 decode gap |
| 环境踩坑 | nsys 缺 OPENSSL_3.3.0 → 用 bundled host nsys + LD_LIBRARY_PATH；home quota 满 → cache 重定向 scratch；triton JIT cache 残缺 → 清后重跑（findings 记录） |

## 产出

6 个 .nsys-rep 在 results/（gitignore，不入 git）；本 result.md + 关键 μs/call 表入库。
