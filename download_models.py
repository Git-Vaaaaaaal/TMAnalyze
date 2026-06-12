"""
download_models.py — Télécharge localement tous les poids des modèles de fondation
utilisés dans pipeline.py. À lancer une seule fois avant les jobs SLURM.

Usage :
    python download_models.py

Les poids sont sauvegardés dans le dossier défini par HF_HOME
(par défaut ~/.cache/huggingface, ou $HF_HOME si la variable est définie).
Sur Jean-Zay, pointer HF_HOME vers $SCRATCH ou $STORE pour un cache persistant :
    export HF_HOME=/lustre/fsn1/projects/rech/ehe/udq27fb/hf_cache
    python download_models.py
"""

import os
from huggingface_hub import login, snapshot_download
from env import hf_token

login(hf_token)

# Répertoire de cache — HF utilise automatiquement HF_HOME/hub si HF_HOME est défini.
# Ne pas passer cache_dir explicitement pour éviter le mismatch avec hf_hub_download.
hf_home = os.environ.get("HF_HOME", None)
cache_dir = None  # laisse HF résoudre via HF_HOME → HF_HOME/hub
if hf_home:
    print(f"HF_HOME : {hf_home}  →  cache effectif : {hf_home}/hub")
else:
    print("Cache HuggingFace : défaut (~/.cache/huggingface/hub)")

# ------------------------------------------------------------------
# Table : nom trident → repo HuggingFace
# ------------------------------------------------------------------
PATCH_ENCODER_REPOS = {
    "gpfm":         "mahmoodlab/GPFM",
    "hibou_l":      "histai/hibou-L",
    "musk":         "xiangjx/musk",
    "openmidnight": "kaiko-ai/midnight",
    "virchow2":     "paige-ai/Virchow2",
    "virchow":      "paige-ai/Virchow",      # utilisé par prism
    "conch_v15":    "MahmoodLab/conch_v1_5", # utilisé par titan / feather
}

SLIDE_ENCODER_REPOS = {
    "titan":   "MahmoodLab/TITAN",
    "prism":   "paige-ai/Prism",
    "feather": "MahmoodLab/feather",
}

# Modeles de segmentation (TMAx — utilise dans run_segmentation_job)
SEGMENTATION_REPOS = {
    "tmax": "Vaaaal/TMAs",
}

# ------------------------------------------------------------------
# Téléchargement
# ------------------------------------------------------------------
all_repos = {**PATCH_ENCODER_REPOS, **SLIDE_ENCODER_REPOS, **SEGMENTATION_REPOS}

for name, repo_id in all_repos.items():
    print(f"\n[{name}] Téléchargement depuis {repo_id} ...")
    try:
        path = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            ignore_patterns=["*.msgpack", "*.ot", "flax_model*", "tf_model*"],
        )
        print(f"  OK → {path}")
    except Exception as e:
        print(f"  ERREUR : {e}")

print("\nTéléchargements terminés.")
