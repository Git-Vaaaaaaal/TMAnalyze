#!/bin/bash
#SBATCH --job-name=grid_search_rf        # Nom du job
#SBATCH --nodes=1
#SBATCH --ntasks=4                 # Nombre total de processus MPI
#SBATCH --hint=nomultithread       # 1 processus MPI par coeur physique (pas d'hyperthreading)
#SBATCH --time=00:20:00            # Temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=unzip.out   # Nom du travail
#SBATCH --error=unzip.out    # Nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --qos=qos_cpu-dev
#SBATCH --account=ehe@cpu

module purge
conda deactivate

source /lustre/fsn1/projects/rech/ehe/udq27fb/TMAnalyze/venv/bin/activate
export PYTHONPATH=/lustre/fsn1/projects/rech/ehe/udq27fb/TMAnalyze/venv/lib/python3.12.7/site-packages:$PYTHONPATH



