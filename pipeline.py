from src.embedding import running_patch_embedding
from src.embedding import running_slide_embedding
from src.class_embedding import Processor

#Partie 1 : Embeddings
job_dir = r"MUM1/job_dir"
wsi_source = r"MUM1/wsi_source"
wsi_ext = ".tiff"
wsi_cache = "cache/"
skip_errors = True
custom_mpp_keys = {}
custom_list_of_wsis = None
max_workers = None
reader_type = None
search_nested = False

magnification = 20.0
patch_size = 64

embedding = Processor(
        job_dir=job_dir,
        wsi_source=wsi_source,
        wsi_ext=wsi_ext,
        wsi_cache=wsi_cache,
        skip_errors=skip_errors,
        custom_mpp_keys=custom_mpp_keys,
        custom_list_of_wsis=custom_list_of_wsis,
        max_workers=max_workers,
        reader_type=reader_type,
        search_nested=search_nested,
    )

encoder_name = "prism"
running_slide_embedding(embedding, encoder_name, magnification, patch_size)