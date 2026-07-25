# exp_00 vLLM MoE 三臂 Benchmark

## 结论

- 5KP 与 6KP 的趋势一致：低并发时 DeepGEMM MXFP8-32 更快；从
  `cc=32` 开始，CuTe FP8 转为最快，领先 DeepGEMM 约 `1.1%–2.9%`。
- CuTe MXFP8-32 在全部 14 个平台×并发点均慢于 CuTe FP8，差距约
  `1.2%–3.7%`。
- 高并发时 CuTe MXFP8-32 与 DeepGEMM 基本持平：`cc=64/128` 上领先约
  `0.2%–0.3%`；低并发仍明显落后。
- 数据稳定性较好：两轮 output throughput 的 CV 最大为 `2.39%`，其余绝大多数
  低于 `1%`。

## 测试背景

| 项目 | 配置 |
|---|---|
| 模型 | Qwen3.5-35B-A3B-FP8 |
| 框架 | vLLM，TP=1 |
| 请求 | ISL=8000，OSL=1000 |
| 并发 | 1、4、8、16、32、64、128 |
| 5KP | 110 SM |
| 6KP | 188 SM |
| 三臂 | CuTe FP8 `(1,128,128)`；CuTe MXFP8 GranK=32；DeepGEMM MXFP8 GranK=32 |
| 汇总口径 | r1/r3 中位数；每个请求 manifest 均经真实 tokenizer 回放验证 |

## 5KP

单位：output tokens/s；括号内为相对 DeepGEMM 的差异。

| 并发 | CuTe FP8 | CuTe MXFP8-32 | DeepGEMM MXFP8-32 |
|---:|---:|---:|---:|
| 1 | 140.6 (-17.2%) | 136.4 (-19.8%) | 169.9 |
| 4 | 364.8 (-7.2%) | 352.7 (-10.3%) | 393.1 |
| 8 | 546.0 (-3.8%) | 532.5 (-6.2%) | 567.5 |
| 16 | 767.1 (-3.3%) | 745.9 (-6.0%) | 793.3 |
| 32 | 970.3 (+2.2%) | 957.8 (+0.9%) | 949.2 |
| 64 | 1147.0 (+2.1%) | 1126.4 (+0.3%) | 1123.3 |
| 128 | 1163.1 (+2.9%) | 1134.6 (+0.3%) | 1130.7 |

## 6KP

单位：output tokens/s；括号内为相对 DeepGEMM 的差异。

| 并发 | CuTe FP8 | CuTe MXFP8-32 | DeepGEMM MXFP8-32 |
|---:|---:|---:|---:|
| 1 | 148.3 (-21.2%) | 142.9 (-24.1%) | 188.3 |
| 4 | 405.2 (-10.9%) | 392.4 (-13.7%) | 454.8 |
| 8 | 625.2 (-6.2%) | 609.0 (-8.7%) | 666.7 |
| 16 | 901.4 (-3.0%) | 890.4 (-4.2%) | 929.6 |
| 32 | 1192.1 (+1.1%) | 1171.9 (-0.6%) | 1178.9 |
| 64 | 1443.2 (+2.3%) | 1413.5 (+0.2%) | 1410.3 |
| 128 | 1597.3 (+2.3%) | 1566.0 (+0.3%) | 1561.6 |

## 数据资格与限制

- 正式 sweep 产生 3 轮数据，但两个平台的 r2 均出现正式测量期间的
  compile/JIT 活动；r2 整轮拒绝，不进入汇总。最终使用两平台共同完整通过的
  r1/r3，共 84 条原始记录。
- 84 条 JSON 均通过 checksum、请求 shape、duration/throughput 算术、源码和
  运行身份校验；7 个请求 manifest 经模型绑定的 tokenizer 验证为逐请求
  `8000/1000` tokens。
- 每个平台内部，CuTe FP8 与 CuTe MXFP8 使用同一个 ELF；5KP 与 6KP 的 CuTe
  ELF 不同，因此本文只做平台内三臂比较，不把跨平台差异解释为硬件收益。
- 两个平台全部有效 arm 的实际 benchmark 脚本 SHA 一致。6KP root harness
  manifest 留有过期的准备阶段脚本 SHA；实际执行 SHA 已从 arm marker 覆盖的
  evidence 中独立校验并记录。
- CuTe FP8/MXFP8 correctness gate 均通过。vLLM 对 Qwen3.5 Blackwell 的
  DeepGEMM E8M0 给出 accuracy degradation 警告；本报告只把 DeepGEMM 作为
  性能参考，不宣称模型级精度等价。

## 产物

- [benchmark.csv](results/benchmark.csv)：14 个平台×并发的直接三臂对比。
- [benchmark_medians.csv](results/benchmark_medians.csv)：42 个后端中位数。
- [benchmark_rounds.csv](results/benchmark_rounds.csv)：84 条有效轮次数据。
- [request_manifest_audit.csv](results/request_manifest_audit.csv)：请求 shape
  与 manifest hash 审计。
- [artifact_identity.csv](results/artifact_identity.csv)：实际 ELF 与 benchmark
  harness 身份。
- `results/{5kp,6kp}/raw/`：仅保留 r1/r3 的原始 JSON、checksum 和 arm evidence。
