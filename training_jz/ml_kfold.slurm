#!/bin/bash
#SBATCH --job-name=ml_kfold
#SBATCH --nodes=1
#SBATCH --ntasks=8                 # Nombre de coeurs (n_jobs=-1 dans RSF les utilisera tous)
#SBATCH --hint=nomultithread       # 1 processus par coeur physique
#SBATCH --time=00:20:00            # Temps max (HH:MM:SS)
#SBATCH --output=ml_kfold_%j.out
#SBATCH --error=ml_kfold_%j.out
#SBATCH --qos=qos_cpu-dev
#SBATCH --account=ehe@cpu

module purge
conda deactivate

source /lustre/fsn1/projects/rech/ehe/udq27fb/TMAnalyze/tmanalyze_venv/bin/activate
export PYTHONPATH=/lustre/fsn1/projects/rech/ehe/udq27fb/TMAnalyze/tmanalyze_venv/lib/python3.12.7/site-packages:$PYTHONPATH

export MPLCONFIGDIR="$TMPDIR/matplotlib"
mkdir -p "$MPLCONFIGDIR"

set -x

cd /lustre/fsn1/projects/rech/ehe/udq27fb/TMAnalyze
python survival_ml/ml_slide.py
python survival_ml/ml_kfold.py