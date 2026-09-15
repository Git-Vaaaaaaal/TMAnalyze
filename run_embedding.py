from src.embedding import running_patch_embedding
from src.embedding import running_slide_embedding
from src.class_embedding import Processor
import torch
import os

from env import hf_token, HF_HOME, HF_HUB_CACHE

#if internet connection not available use download_models.py
# to download models and set internet_connection = False
internet_connection = True

# Parameters of the datasets using differente marker encoders and slide encoders
list_marker = ["MYC"]#"BCL2", "BCL6", "CD10", "HE", "MUM1", "MYC"
list_encoder = ["gpfm", "openmidnight", "virchow2", "prism", "feather"]# "hoptimus1" prism", "feather", "titan"]
slide_list = ["prism",  "feather"]


if internet_connection == False :
    # Mode offline : on configure le cache AVANT tout appel réseau, et on ne fait
    # PAS appel à login() — login() interroge le Hub (whoami) et planterait sans
    # accès internet. HF_HOME/HF_HUB_CACHE viennent de env.py pour pointer vers le
    # même cache que celui rempli par download_models.py (à adapter dans env.py si
    # vous n'êtes pas sur Jean Zay).
    os.environ["HF_TOKEN"]             = hf_token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
    os.environ["HF_HUB_OFFLINE"]       = "1"
    os.environ["TRANSFORMERS_OFFLINE"]  = "1"
    if HF_HOME:
        os.environ["HF_HOME"] = HF_HOME
    if HF_HUB_CACHE:
        os.environ["HF_HUB_CACHE"] = HF_HUB_CACHE
else :
    # Mode online : on peut valider le token auprès du Hub.
    from huggingface_hub import login
    login(hf_token)


for marker in list_marker :
    for slide_encoder in list_encoder :

        # Path to the dataset
        path = os.path.join("data_224_reborn", f"{slide_encoder}")

        #Output directory for embeddings
        job_dir = os.path.join(path, f"{marker}")

        wsi_source = os.path.join(path, f"{marker}", "wsi_source")

        # Image extensions to consider
        wsi_ext = [".tiff"]
        wsi_cache = "cache/"
        skip_errors = True
        custom_list_of_wsis = None
        max_workers = None
        reader_type = None
        search_nested = False


        # Image parameters
        magnification = 40.0
        patch_size = 224

        GPU = 0
        device = f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        embedding = Processor(
                job_dir=job_dir,
                wsi_source=wsi_source,
                wsi_ext=wsi_ext,
                wsi_cache=wsi_cache,
                skip_errors=skip_errors,
                custom_list_of_wsis=custom_list_of_wsis,
                max_workers=max_workers,
                search_nested=search_nested,
                mpp=0.2535,
            )

        encoder_name = f"{slide_encoder}"
        if slide_encoder in slide_list :
            running_slide_embedding(embedding, encoder_name, magnification, patch_size)
        else :
            running_patch_embedding(embedding, encoder_name, magnification, patch_size)