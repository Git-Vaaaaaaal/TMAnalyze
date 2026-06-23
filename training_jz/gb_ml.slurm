#!/bin/bash
#SBATCH --job-name=gb_ml        # Nom du job
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
module load arch/h100
module load pytorch-gpu/py3/2.4.0

source /lustre/fsn1/projects/rech/ehe/udq27fb/TMAnalyze/tmanalyze_venv/bin/activate
export PYTHONPATH=/lustre/fsn1/projects/rech/ehe/udq27fb/TMAnalyze/tmanalyze_venv/lib/python3.12.7/site-packages:$PYTHONPATH

export MPLCONFIGDIR="$TMPDIR/matplotlib"
mkdir -p "$MPLCONFIGDIR"

# Remove system cuDNN from LD_LIBRARY_PATH so PyTorch uses its bundled version
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v 'cudnn' | tr '\n' ':' | sed 's/:$//')

# Disable torch.compile (Triton JIT fails on this cluster due to missing libcuda.so.1 in gcc -L paths)
export TORCH_COMPILE_DISABLE=1

set -x

cd 
python /lustre/fsn1/projects/rech/ehe/udq27fb/TMAnalyze/ml_.py


