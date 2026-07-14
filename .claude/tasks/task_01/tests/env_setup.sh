#!/bin/bash
set -euo pipefail
cd /home/scratch.xiy_gpu/mega_inference/vllm

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export UV_CACHE_DIR=/home/scratch.xiy_gpu/.uv_cache

[ -d .venv ] || uv venv --python 3.12
source .venv/bin/activate

VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto

python - << 'EOF'
import torch, vllm
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cc:", torch.cuda.get_device_capability(0))
print("vllm", vllm.__version__)
from vllm.utils.import_utils import has_deep_gemm
print("has_deep_gemm:", has_deep_gemm())
EOF
echo ENV_SETUP_DONE
