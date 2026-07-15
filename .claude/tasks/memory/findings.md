# Shared Task Findings（vLLM cute_sm120_precision）

记客观发现 / bug / 踩坑 / 性能，不记主观设计决策。风格同 FI `tasks/memory/findings.md`。

## task_03: cute MXFP8 3a/3b 接入 (2026-07-14)

### Kernel-level bug 诊断（parity r2 0/20 root cause）

- **FI MXFP8 kernel 的 b_scale 是 MN-major 存储 contract，shape check 挡不住存储序错误**：
  `deduce_sfb_layout`（FI `sf_mxfp8_tma_load.cuh:112-126`）stride `(1, scale_n, scale_n*scale_k)`；
  binding（`cute_sm120_mxfp8_op.cu`）只取 `data_ptr()` 不传 stride，shape check 只验
  `(E, align(n,4), k_align)` → row-major contiguous 的 b_scale 静默给出转置 scale 读取，
  全 cells 系统性 calc_diff 7e-2~4.5e-1（granK=32 更大：scale 列数 4×、per-row 方差大）。
  修复 = `requant_weight_for_cute_mxfp8` 存储 `(E, k_align, N)` contiguous + 返回 `.transpose(1,2)` 视图。
  FI upstream test 之所以 PASS：其打包器 `get_col_major_tma_aligned_packed_tensor` 天然产 MN-major。
- **torch size-1 维 stride 陷阱**：`(E, N, k_align=1)` 的 transpose 视图 `is_contiguous()==True`
  （size-1 维 stride 不参与判定）→ `.contiguous()` 不 copy、原 stride 保留 →
  `view(torch.uint8)` 要求 stride(-1)==1 直接 raise。修复 = `clone(memory_format=torch.contiguous_format)`。

### 实测

- parity 20/20 PASS + e2e 与既有列同带、granK 32≈128 → [task_03/result.md](../task_03/result.md)。

## task_04: v2 对齐 + AIME/MBPP + 多卡 + H20 baseline (2026-07-14)

### 环境踩坑

- **ssh-gw 按 GPU 型号自动推导 partition 会拿错节点**（复发，task_01 已有先例）：
  `--gpu h20` 无显式 partition 时落到 GH200（aarch64）节点。多 partition 逗号串
  `-p p1,p2` Slurm 接受。8 卡 6K Pro 的 8gpu partition 实名：
  `rtx-pro-6000-blackwell-server-edition@{cr+mp,qs1}/x13degoa/8gpu-224cpu-2048gb`。
- **DSv4-Flash serve 必须 `--kv-cache-dtype fp8_ds_mla`**：默认 auto 时 worker 全灭
  `AssertionError: DeepseekV4 fp8_ds_mla layout only supports fp8 kv-cache, got auto`
  （`vllm/models/deepseek_v4/attention.py:83`）；`fp8_ds_mla` 是 cache.py 合法枚举。
- **lm-eval local-completions 的 `max_length` 默认 2048，与 `max_gen_toks` 冲突时静默截 prompt**：
  AIME（max_gen_toks=32768）prompt 预算变负 → prompt 截成空 → vLLM 400
  "The decoder prompt cannot be empty"，全 backend 齐挂。修复 = model_args 加
  `max_length=<serve max-model-len>`。MMLU（loglikelihood 路径）不受影响。
- **`is_deep_gemm_e8m0_used()` 无架构 gate（deep_gemm.py:103-121，只看 env 默认 1）**：
  H20（sm90）DG 路径静默做 UE8M0 requant。实测 GSM8K：H20 DG-UE8M0 0.6626 vs
  H20 triton 0.7445（同 FLA GDN 条件，-8.2pp）——sm90 上 UE8M0 退化远重于 sm120 的 -2pp，
  是否含 kernel 级 scale 处理问题未归因。float-scale 基线需显式 `VLLM_USE_DEEP_GEMM_E8M0=0`。
- **H20 上 FI GDN prefill 首跑产全空输出**：r1 GSM8K exit 0 但 accuracy 0.000 / invalid 1.000 /
  0 output tokens——eval 脚本不因 invalid fail，**exit 0 ≠ 数据有效**，收数必查 invalid_rate。
- **H20（sm90）上 vLLM GDN prefill 默认选 FlashInfer 路径，我们 FI fork 的 sm90 CuTeDSL
  编译在该 venv 组合下炸**（`cute.compile[...]` → `'function' object is not subscriptable`，
  `delta_rule_sm90.py:2462`）→ 首个触发 shape 的 batch 直接 EngineCore fatal。修复 =
  serve 加 `--gdn-prefill-backend triton`（vLLM 原生 FLA 路径；对 H20 native baseline 语义更纯）。
  sm120 不走该 kernel 所以从未暴露。
- **lm-eval api_models 客户端 request timeout 默认 300s**：AIME 单请求 32k greedy 生成
  >5min → asyncio cancel → `ServerDisconnectedError`（serve 侧无 crash，log 干净 shutdown 是
  trap 杀的，别误判 server 挂）。修复 = model_args 加 `timeout=7200`。短生成评测不受影响。
- **MBPP 的 HF `code_eval` metric 有独立安全闸**：`--confirm_run_unsafe_code` 只过 lm-eval
  自己的 gate，metric 计算期还要 env `HF_ALLOW_CODE_EVAL=1`，否则 `ValueError(_WARNING)`。
- lm-eval 0.4.12：`aime24`/`aime25` 各 30 题 0-shot greedy `max_gen_toks=32768`（serve ctx 必须 ≥33k）；
  `mbpp` 500 题 3-shot 内置、代码在 lm-eval 客户端进程执行、CLI 必须 `--confirm_run_unsafe_code`
  （yaml `unsafe_code: true`）。

- **ssh-gw 客户端异常退出/被杀时 salloc 会脱管存活成孤儿 job**（实锤两例：报 "job disappeared"
  的 3105055 实际 granted 空占 8 卡数小时；TaskStop 客户端后 3106200 仍 granted 占 1 卡）。
  停掉申请客户端或客户端报错后，必须 `squeue -u $USER` 对账并 scancel 脱管 job。
- **ssh-gw 高频轮询会触发 computelab 登录节点 sshd 连接限速**：数小时 `task wait`/exec
  轮询后，新 SSH 一律 `ssh_exchange_identification: Connection closed`（30 次重试全拒，
  已有多路复用连接不受影响）。恢复只能冷却等待；长等待场景应拉长轮询间隔或改用
  一次性 wait，不要叠多个 watcher 各自轮询。
- **ts 串行任务间的 serve 显存释放竞态**：前一 task 的 trap kill 只杀外层 api_server，
  EngineCore 子进程释放 90G 需要时间甚至成僵尸（实锤 task 80→81 连锁：80 的 serve 20min
  没 health 成僵尸，81 起服 "Free memory 6.19 GiB" 直接挂）。修复 = v2 脚本加 drain guard
  （轮询 nvidia-smi 到 free>80G 再起服，5min 上限）；僵尸需手动 kill。
- **vLLM DeepGEMM warmup 按 isinstance(DeepGemmExperts) 抓 warmup 对象且硬编码默认 recipe**
  （`deep_gemm_warmup.py:189`）：DG 子类若换 scale 布局（如 (1,32) packed），engine 启动期
  warmup GEMM 直接 layout assert 挂——serve 级问题，单层 parity 测不到。修复 = warmup gate
  排除 `DeepGemmMxfp8Gran32Experts`（首 batch JIT 代价可接受）。

### 实测

- 同 UE8M0 recipe 下 cute 比 DG 高 1.4-2.8pp（35B GSM8K，dg-32 双 run 复现）、大模型上效应消失、
  v1→v2 config 变化 ≤0.5pp → 全矩阵与补测判定见 [task_04/result.md](../task_04/result.md)。
- **397B MBPP 坍塌机制**（log_samples 取证）：67% 空生成、非空样本为正常代码 → 模型对 3-shot
  completion prompt 首 token 即停止序列；backend 无关；DSv4-Base 同格式正常 → 与 instruct/thinking
  训练风格相关。
- **AIME 列间差异无信息量的实证**：397B AIME24 0.200 复测 0.067 不可复现——32k greedy 长 CoT 的
  run-to-run 轨迹波动可让 30 题 cell 摆动 4/30。

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
- UE8M0 GSM8K -2.5pp 而 MMLU 持平（退化只在长 CoT 累积显形；冒烟 100 题波动大不可用于结论）
  → 数字见 [task_01/result.md](../task_01/result.md)。

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
