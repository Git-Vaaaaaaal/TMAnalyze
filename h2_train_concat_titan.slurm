#!/bin/bash
#SBATCH --job-name=h100_titan # nom du job
#SBATCH --output=h100_titan.out # fichier de sortie (%j = job ID)
#SBATCH --error=h100_titan.err # fichier d’erreur (%j = job ID)
#SBATCH --constraint=h100 # demander des GPU H100 80 Go
#SBATCH --nodes=1 # reserver 2 nœuds
#SBATCH --ntasks=1 # reserver 8 taches (ou processus)
#SBATCH --gres=gpu:4 # reserver 4 GPU par noeud
#SBATCH --cpus-per-task=24 # reserver 24 CPU par tache (et memoire associee)
#SBATCH --time=20:00:00 # temps maximal d’allocation "(HH:MM:SS)"
#SBATCH --hint=nomultithread # desactiver l’hyperthreading
#SBATCH --account=ehe@h100 # comptabilite H100

module purge # nettoyer les modules herites par defaut
conda deactivate # desactiver les environnements herites par defaut
module load arch/h100 # selectionner les modules compiles pour les H100
module load pytorch-gpu/py3/2.4.0 # charger les modules

source /lustre/fsn1/projects/rech/ehe/udq27fb/TMAnalyze/tmanalyze_venv/bin/activate
export PYTHONPATH=/lustre/fsn1/projects/rech/ehe/udq27fb/TMAnalyze/tmanalyze_venv/lib/python3.12.7/site-packages:$PYTHONPATH

set -x # activer l’echo des commandes


#File to run 
srun python /lustre/fsn1/projects/rech/ehe/udq27fb/TMAnalyze/train_concatenation_h100.py # script pour embedding des tuiles 224

