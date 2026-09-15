#!/bin/bash
#SBATCH --job-name=ml_kfold        # Nom du job
#SBATCH --nodes=1
#SBATCH --ntasks=4                 # Nombre total de processus MPI
#SBATCH --hint=nomultithread       # 1 processus MPI par coeur physique (pas d'hyperthreading)
#SBATCH --time=00:10:00            # Temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=ml_kfold.out   # Nom du travail
#SBATCH --error=ml_kfold.out    # Nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --qos=qos_cpu-dev
#SBATCH --account=ehe@cpu

module purge

source /lustre/fsn1/projects/rech/ehe/udq27fb/TMAnalyze/tmanalyze_venv/bin/activate
PYVER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
export PYTHONPATH=/lustre/fsn1/projects/rech/ehe/udq27fb/TMAnalyze/tmanalyze_venv/lib/python${PYVER}/site-packages:$PYTHONPATH

export MPLCONFIGDIR="$TMPDIR/matplotlib"
mkdir -p "$MPLCONFIGDIR"

set -x

cd /lustre/fsn1/projects/rech/ehe/udq27fb/TMAnalyze
python survival_ml/ml_slide.py
python survival_ml/ml_kfold.py