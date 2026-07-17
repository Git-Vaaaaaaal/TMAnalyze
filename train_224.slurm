#!/bin/bash
#SBATCH --job-name=train_224     # nom du job
#SBATCH --output=train_224.out  # fichier de sortie (%j = job ID)
#SBATCH --error=train_224.err   # fichier d'erreur (%j = job ID)
#SBATCH --constraint=h100               # demander des GPU H100 80 Go
#SBATCH --nodes=1                       # reserver 1 noeud
#SBATCH --ntasks=1                      # 1 tache shell (les shards sont des sous-processus)
#SBATCH --gres=gpu:4                    # reserver 4 GPU par noeud
#SBATCH --cpus-per-task=48              # 12 CPU par shard x 4 shards
#SBATCH --time=20:00:00                 # temps maximal d'allocation "(HH:MM:SS)"
#SBATCH --hint=nomultithread            # desactiver l'hyperthreading
#SBATCH --account=ehe@h100              # comptabilite H100

module purge
conda deactivate
module load arch/h100
module load pytorch-gpu/py3/2.4.0

source /lustre/fsn1/projects/rech/ehe/udq27fb/TMAnalyze_old/tmanalyze_venv/bin/activate
export PYTHONPATH=/lustre/fsn1/projects/rech/ehe/udq27fb/TMAnalyze_old/tmanalyze_venv/lib/python3.11/site-packages:$PYTHONPATH

# Cache HuggingFace — pointe vers les poids pre-telecharges
export HF_HOME=/lustre/fsn1/projects/rech/ehe/udq27fb/cache/huggingface
export HUGGINGFACE_HUB_CACHE=/lustre/fsn1/projects/rech/ehe/udq27fb/cache/huggingface/hub
export TRANSFORMERS_CACHE=/lustre/fsn1/projects/rech/ehe/udq27fb/cache/huggingface/hub

set -x

SCRIPT=/lustre/fsn1/projects/rech/ehe/udq27fb/TMAnalyze/train_new.py
NUM_SHARDS=4
NUM_WORKERS=10

# Lance un processus par GPU, chacun traite son sous-ensemble de combinaisons
for SHARD_ID in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$SHARD_ID \
    SHARD_ID=$SHARD_ID \
    NUM_SHARDS=$NUM_SHARDS \
    NUM_WORKERS=$NUM_WORKERS \
    python $SCRIPT > /lustre/fsn1/projects/rech/ehe/udq27fb/TMAnalyze/training_jz/shard_${SHARD_ID}.out 2>&1 &
done

wait  # attend que tous les shards terminent
echo "Tous les shards termines."
