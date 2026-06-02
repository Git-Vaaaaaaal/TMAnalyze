#!/bin/bash
#SBATCH --job-name=TravailGPU # nom du job
#SBATCH --output=TravailGPU%j.out # fichier de sortie (%j = job ID)
#SBATCH --error=TravailGPU%j.err # fichier d’erreur (%j = job ID)
#SBATCH --constraint=a100 # demander des GPU A100 80 Go
#SBATCH --nodes=1 # reserver 1 nœud
#SBATCH --ntasks=1 # reserver 1 tache (processus unique)
#SBATCH --gres=gpu:1 # reserver 1 GPU
#SBATCH --cpus-per-task=8 # reserver 8 CPU par tache (et memoire associee)
#SBATCH --time=20:00:00 # temps maximal d’allocation "(HH:MM:SS)"
#SBATCH --hint=nomultithread # desactiver l’hyperthreading
#SBATCH --account=ehe@a100 # comptabilite A100

module purge # nettoyer les modules herites par defaut
conda deactivate # desactiver les environnements herites par defaut
module load arch/a100 # selectionner les modules compiles pour les A100
module load pytorch-gpu/py3/2.3.0 # charger les modules
source /lustre/fsn1/projects/rech/ehe/udq27fb/TMAnalyze/tmanalyze_venv/bin/activate
export PYTHONPATH=/lustre/fsn1/projects/rech/ehe/udq27fb/TMAnalyze/tmanalyze_venv/lib/python3.12.7/site-packages:$PYTHONPATH
set -x # activer l’echo des commandes


#File to run
python /lustre/fsn1/projects/rech/ehe/udq27fb/TMAnalyze/train_concatenation.py


