#!/bin/bash
#SBATCH --job-name=ml_kfold
#SBATCH --partition=cpu_dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=00:20:00
#SBATCH --output=ml_kfold.out
#SBATCH --error=ml_kfold.err
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