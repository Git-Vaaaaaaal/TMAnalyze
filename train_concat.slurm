#!/bin/bash
#SBATCH --job-name=training_torchmil
#SBATCH --constraint=v100-32g
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --qos=qos_gpu-t3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --output=train.out
#SBATCH --error=logs.out
#SBATCH -A ehe@v100
module purge # nettoyer les modules herites par defaut
module load python
module load cuda/12.0.0
module load arch/a100 # selectionner les modules compiles pour les A100
source /lustre/fsn1/projects/rech/ehe/udq27fb/TMAnalyze/tmanalyze_venv/bin/activate
export PYTHONPATH=/lustre/fsn1/projects/rech/ehe/udq27fb/TMAnalyze/tmanalyze_venv/lib/python3.12.7/site-packages:$PYTHONPATH
set -x # activer l’echo des commandes

#File to run 
srun python /lustre/fsn1/projects/rech/ehe/udq27fb/TMAnalyze/train_concatenation.py # script pour embedding des tuiles 224
