from src.embedding import running_patch_embedding
from src.embedding import running_slide_embedding
from src.class_embedding import Processor
import torch
import os

from env import hf_token

# Sur les noeuds sans internet (Jean-Zay GPU), login() crashe.
# On passe le token via variable d'environnement et on force le mode offline.
os.environ["HF_TOKEN"]             = hf_token
os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
os.environ["HF_HUB_OFFLINE"]       = "1"
os.environ["HF_HUB_CACHE"]         = "/lustre/fsn1/projects/rech/ehe/udq27fb/cache/huggingface/hub"

list_marker = ["HE", "MUM1", "MYC"] #
list_slide_encoder = ["prism", "feather"] #["musk", "gpfm",  "hibou_l", "openmidnight", "virchow2"] "hoptimus1" prism", "feather", "titan"]
slide_list = ["titan", "prism", "feather"]


for marker in list_marker :
    for slide_encoder in list_slide_encoder :
        path = os.path.join("data_224_reborn", f"{slide_encoder}")
        #Partie 1 : Embeddings
        job_dir = os.path.join(path, f"{marker}")
        wsi_source = os.path.join(path, f"{marker}", "wsi_source")
        wsi_ext = [".tiff"]
        wsi_cache = "cache/"
        skip_errors = True
        custom_list_of_wsis = None
        max_workers = None
        reader_type = None
        search_nested = False

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
        running_slide_embedding(embedding, encoder_name, magnification, patch_size)
        #running_patch_embedding(embedding, encoder_name, magnification, patch_size)