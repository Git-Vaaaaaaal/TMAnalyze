from src.embedding import running_patch_embedding
from src.embedding import running_slide_embedding
from src.class_embedding import Processor
import torch
import os

from huggingface_hub import login
from env import hf_token

login(hf_token)

#list_slide_encoder = ["prism"] #"prism"
list_slide_encoder = ["gpfm", "openmidnight", "virchow2", "prism", "feather", "hibou_l", "musk", ]# "hoptimus1" prism", "feather", "titan"]
slide_list = ["prism",  "feather"]

embed_path = "embedding"
os.makedirs(embed_path, exist_ok=True)

for slide_encoder in list_slide_encoder :
    #img_path = os.path.join("data", "TMAnalyze", "img")
    img_path = "/data/TMAnalyze/img"
    path = os.path.join(embed_path, f"{slide_encoder}")
    os.makedirs(path, exist_ok=True)
    #Partie 1 : Embeddings
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
            job_dir=path,
            wsi_source=img_path,
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