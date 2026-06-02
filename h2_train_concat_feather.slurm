#!/bin/bash
#SBATCH --job-name=h100_feather # nom du job
#SBATCH --output=h100_feather.out # fichier de sortie (%j = job ID)
#SBATCH --error=h100_feather.err # fichier d’erreur (%j = job ID)
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

cd /lustre/fsn1/projects/rech/ehe/udq27fb/TMAnalyze

# --- Sharding : 1 process Python par GPU, chacun isole sur 1 GPU --------
NUM_SHARDS=4
# ~24 CPU / 4 process = 6 workers par process pour le DataLoader
export NUM_WORKERS=6
export NUM_SHARDS

pids=""
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    CUDA_VISIBLE_DEVICES=$i SHARD_ID=$i \
        python /lustre/fsn1/projects/rech/ehe/udq27fb/TMAnalyze/train_concatenation_h100_feather.py &
    pids="$pids $!"
done

# Attendre la fin des 4 process ; echouer si l'un d'eux echoue
rc=0
for p in $pids; do
    wait "$p" || rc=1
done
exit $rc

