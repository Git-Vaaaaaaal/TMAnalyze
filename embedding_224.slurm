#!/bin/bash
#SBATCH --job-name=musk_224     # nom du job
#SBATCH --output=musk_224.out  # fichier de sortie (%j = job ID)
#SBATCH --error=musk_224.err   # fichier d'erreur (%j = job ID)
#SBATCH --constraint=h100               # demander des GPU H100 80 Go
#SBATCH --nodes=1                       # reserver 1 noeud
#SBATCH --ntasks=1                      # 1 tache shell (les shards sont des sous-processus)
#SBATCH --gres=gpu:1                    # reserver 1 GPU
#SBATCH --cpus-per-task=8               # reserver 8 CPU
#SBATCH --time=20:00:00                 # temps maximal d'allocation "(HH:MM:SS)"
#SBATCH --hint=nomultithread            # desactiver l'hyperthreading
#SBATCH --account=ehe@h100              # comptabilite H100

module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0

source /lustre/fsn1/projects/rech/ehe/udq27fb/TMAnalyze/tmanalyze_venv/bin/activate

# Cache HuggingFace — pointe vers les poids pre-telecharges
export HF_HOME=/lustre/fsn1/projects/rech/ehe/udq27fb/cache/huggingface
export HUGGINGFACE_HUB_CACHE=/lustre/fsn1/projects/rech/ehe/udq27fb/cache/huggingface/hub
export TRANSFORMERS_CACHE=/lustre/fsn1/projects/rech/ehe/udq27fb/cache/huggingface/hub
# Mode offline : interdit tout appel reseau vers huggingface.co
export HF_HUB_OFFLINE=1

# Cache PyTorch / torchvision (resnet34 pour TMAx, etc.)
export TORCH_HOME=/lustre/fsn1/projects/rech/ehe/udq27fb/cache/torch

set -x

python /lustre/fsn1/projects/rech/ehe/udq27fb/TMAnalyze/pipeline_offline.py
