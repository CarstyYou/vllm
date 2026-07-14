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
  与 vLLM 官方 auto-disable 理由方向一致；数据 link: [task_01/result.md](../task_01/result.md)。
  冒烟 100 题波动大（0.900/0.810），不可用于结论。
- **MMLU 全量持平**（triton 0.8465 vs dg 0.8478，< 1 stderr）：UE8M0 退化只在长 CoT 累积显形，
  MMLU 型单点判别评测检不出。

## task_02: cute FP8 experts 接入 (2026-07-13)

FI entry 精确 contract（a_scale 4 列对齐 MN-major、b_scale K-first 等）与 vLLM 对接事实的
ground truth 见 [task_02/sub_task_0_findings.md](../task_02/sub_task_0_findings.md)，不在此复制。

### 环境踩坑

- **FI editable (0.6.14-dev) 与 vLLM venv 自带 flashinfer-cubin (0.6.13) 版本检查冲突**：
  `import flashinfer` 直接 raise。解法：`FLASHINFER_DISABLE_VERSION_CHECK=1`（我们只用自建 JIT
  的 grouped_mm，不消费 cubin 包）。所有 serve/eval/smoke 脚本必须带。
- FI JIT cache 走 `FLASHINFER_WORKSPACE_BASE=/home/scratch.xiy_gpu`（home 太小）。
- **裸机 venv 跑 FI JIT 的完整依赖清单**：`nvidia-cutlass-dsl[cu13]>=4.5` + `ninja`（pip 包，
  binary 落 venv/bin，PATH 要含 venv/bin）+ CUDA_HOME(nvcc)。缺 ninja 时 test 0.5s 失败
  `FileNotFoundError: 'ninja'`。
- **ts 任务里 `pytest | tail && echo OK` 会假阳性**（pipefail 经 ts 的 bash -c 包装后失效）：
  必须 `pytest > log; RC=$?; ...; exit $RC` 显式传递退出码。

### 实测

- **vLLM 融合 SiLU+quant kernel（colmajor）硬性要求 M%128==0**（fp8_utils.py:445）——
  exact packing (align=1) 直接喂会 assert 崩；解法 = quant 阶段用 128-pad zero-fill 视图，
  GEMM 仍吃 exact 切片（review B1，pad 行经 silu 产 0、scale 被 eps clamp，无 NaN）。
- `topk_ids` 在无 expert_map 时也可能含 -1（`VLLM_MOE_SKIP_PADDING`，默认关）——
  exact-packing 类 experts 必须防（init 期 assert env 关闭）。

### oracle 注册实测（sub-task 4）

- **`convert_to_fp8_moe_kernel_format` 默认分支是白名单 + raise，不是 pass-through**
  （oracle/fp8.py:558-570）：新 backend 必须加进白名单，否则权重加载期即崩——
  注册实际是 **5 处**（enum / priority / kernel_cls / map / convert 白名单）+ Literal。
- AUTO 落点分析：sm120 + deep_gemm 可用时 AUTO 仍落 DEEPGEMM（qwen3_5 的 auto-disable
  是 runtime shape 级、不影响 oracle 选择）→ cute 只能显式 `--moe-backend=cute_sm120_fp8`；
  `VLLM_USE_DEEP_GEMM=0` 时 AUTO 从 TRITON 变为 CUTE_SM120_FP8（本分支的有意行为变化）。
- **禁止原地编辑排队/运行中 ts 任务正在执行的脚本**：bash 按字节偏移惰性读脚本，
  重写后偏移错位（实锤 `sleep`→`leep` command not found, exit 127）。改脚本前先确认
  队列里没有引用它的 pending/running 任务，或复制新名。
