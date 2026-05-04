from src.embedding import running_patch_embedding
from src.embedding import running_slide_embedding
from src.class_embedding import Processor
import torch
import os

from huggingface_hub import login
from env import hf_token

login(hf_token)

list_marker = ["BCL2", "BCL6", "CD10", "HE", "MUM1", "MYC"]
list_slide_encoder = ["titan", "feather","prism"]

for marker in list_marker :
    for slide_encoder in list_slide_encoder :
        path = os.path.join("data", f"{slide_encoder}")
        #Partie 1 : Embeddings
        job_dir = os.path.join(path, f"{marker}")
        wsi_source = os.path.join(path, f"{marker}", "wsi_source")
        wsi_ext = [".tiff"]
        wsi_cache = "cache/"
        skip_errors = True
        custom_mpp_keys = {}
        custom_list_of_wsis = None
        max_workers = None
        reader_type = None
        search_nested = False

        magnification = 20.0
        patch_size = 64

        GPU = 0
        device = f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        embedding = Processor(
                job_dir=job_dir,
                wsi_source=wsi_source,
                wsi_ext=wsi_ext,
                wsi_cache=wsi_cache,
                skip_errors=skip_errors,
                custom_mpp_keys=custom_mpp_keys,
                custom_list_of_wsis=custom_list_of_wsis,
                max_workers=max_workers,
                search_nested=search_nested,
            )

        encoder_name = f"{slide_encoder}"
        running_slide_embedding(embedding, encoder_name, magnification, patch_size)