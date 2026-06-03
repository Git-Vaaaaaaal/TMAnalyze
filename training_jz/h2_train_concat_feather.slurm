#!/bin/bash
#SBATCH --job-name=h100_feather # nom du job
#SBATCH --output=h100_feather.slurm.log # log brut/verbeux (set -x, modules, traces python)
#SBATCH --error=h100_feather.slurm.err # erreurs brutes
# NB : h100_feather.out est reserve au tableau d'avancement, ecrit par le moniteur.
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

RESULT_DIR=Result_h100_feather
PROG_DIR=$RESULT_DIR/.progress
DASH=h100_feather.out          # tableau d'avancement (seule chose dans ce fichier)
mkdir -p "$PROG_DIR"

# --- Moniteur : agrege les .progress/*.status -> tableau dans $DASH ------
render_dashboard() {
    local tmp="$DASH.tmp" sum_e=0 sum_t=0
    {
        echo "Grid search feather — $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
        for f in $(ls "$PROG_DIR"/*.status 2>/dev/null | sort); do
            IFS=$'\t' read -r label epoch total state < "$f"
            [ -z "$total" ] && continue
            local idx pct tag
            idx=$((10#$(basename "$f" .status)))
            pct=0; [ "$total" -gt 0 ] && pct=$((100 * epoch / total))
            tag=""
            case "$state" in
                done)    tag="  [termine]" ;;
                pending) tag="  [en attente]" ;;
                skip)    tag="  [ignore]" ;;
            esac
            printf "%2d  %-30s : Epoch %d/%d, %d %%%s\n" \
                   "$idx" "$label" "$epoch" "$total" "$pct" "$tag"
            sum_e=$((sum_e + epoch)); sum_t=$((sum_t + total))
        done
        echo ""
        local tot=0; [ "$sum_t" -gt 0 ] && tot=$((100 * sum_e / sum_t))
        echo "Total progress = $tot %"
    } > "$tmp"
    mv "$tmp" "$DASH"   # remplacement atomique
}

# --- Sharding : 1 process Python par GPU, chacun isole sur 1 GPU --------
NUM_SHARDS=4
# ~24 CPU / 4 process = 6 workers par process pour le DataLoader
export NUM_WORKERS=6
export NUM_SHARDS

pids=""
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    CUDA_VISIBLE_DEVICES=$i SHARD_ID=$i \
        python /lustre/fsn1/projects/rech/ehe/udq27fb/TMAnalyze/train_concatenation_h100_feather.py \
        > "$RESULT_DIR/shard_${i}.out" 2>&1 &
    pids="$pids $!"
done

# Rafraichir le tableau toutes les 15 s tant que des process tournent.
while :; do
    running=0
    for p in $pids; do kill -0 "$p" 2>/dev/null && running=1; done
    render_dashboard
    [ "$running" -eq 0 ] && break
    sleep 15
done

# Attendre la fin des 4 process ; echouer si l'un d'eux echoue.
rc=0
for p in $pids; do
    wait "$p" || rc=1
done
render_dashboard   # rendu final
exit $rc

