from trident import Processor

def initialize_processor():
    return Processor(
        job_dir=JOB_DIR,
        wsi_source=WSI_DIR,
        wsi_ext=WSI_EXT,
        wsi_cache=WSI_CACHE,
        skip_errors=SKIP_ERRORS,
        custom_mpp_keys=CUSTOM_MPP_KEYS,
        custom_list_of_wsis=CUSTOM_LIST_OF_WSIS,
        max_workers=MAX_WORKERS,
        reader_type=READER_TYPE,
        search_nested=SEARCH_NESTED,
    )
