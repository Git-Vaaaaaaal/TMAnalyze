#!/usr/bin/env bash
# Fix incompatibilite torch / NCCL (undefined symbol: ncclCommResume)
# A executer sur calymprdube, dans /data/TMAnalyze, venv active.
set -e

# Le disque systeme (/tmp, ~/.cache) est sature : on redirige le cache pip
# et les fichiers temporaires vers /data (qui a de la place) pour l'install.
export TMPDIR="/data/TMAnalyze/tmp_pip"
export PIP_CACHE_DIR="/data/TMAnalyze/pip_cache"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

echo "==> Desinstallation de torch et du stack nvidia-*"
pip uninstall -y torch torchvision torchaudio \
  nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 \
  nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-cufile-cu12 nvidia-curand-cu12 \
  nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cusparselt-cu12 \
  nvidia-nccl-cu12 nvidia-nvjitlink-cu12 nvidia-nvshmem-cu12 nvidia-nvtx-cu12

echo "==> Purge du cache pip"
pip cache purge

echo "==> Reinstallation propre de torch (pip resout lui-meme les versions nvidia-*)"
pip install --cache-dir "$PIP_CACHE_DIR" torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo "==> Verification que torch charge bien CUDA"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

echo "==> Relance du pipeline"
python pipeline_calym.py

echo "==> Versions figees a reporter dans requirements.txt :"
pip freeze | grep -iE "^torch|^nvidia-"
