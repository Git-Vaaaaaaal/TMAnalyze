import os
import torch
import segmentation_models_pytorch
from trident.patch_encoder_models import encoder_registry as patch_encoder_registry
from trident.slide_encoder_models import encoder_registry as slide_encoder_registry
from trident.patch_encoder_models.load import encoder_factory

def running_patch_embedding(processor, model_name, magnification, patch_size):
    gpu = 0 # == cuda

    # Segmentation task
    processor.run_segmentation_job()

    # Patching task
    processor.run_patching_job(
        target_magnification=magnification,
        patch_size=patch_size,
        overlap=0,
        min_tissue_proportion=0.0
    )

    # Embedding task
    encoder = encoder_factory(model_name)
    processor.run_patch_feature_extraction_job(
        coords_dir=f'{magnification}x_{patch_size}px_{0}px_overlap',
        patch_encoder=encoder,
        patch_size=patch_size,
        target_mag=magnification,
        device=f'cuda:{gpu}'
    )


def running_slide_embedding(processor, slide_model, magnification, patch_size):
    gpu = 0 # == cuda

    slide_to_patch_encoder_name = {
        'threads': 'conch_v15',
        'titan': 'conch_v15',
        'tcga': 'conch_v15',
        'prism': 'virchow',
        'chief': 'ctranspath',
        'gigapath': 'gigapath',
        'madeleine': 'conch_v1',
        'feather': 'conch_v15'
    }
        
    # Segmentation task
    processor.run_segmentation_job()

    # Patching task
    processor.run_patching_job(
        target_magnification=magnification,
        patch_size=patch_size,
        overlap=0,
        min_tissue_proportion=0.0
    )

    # Embedding patch task
    model_name = slide_to_patch_encoder_name.get(slide_model)
    encoder = encoder_factory(model_name)
    processor.run_patch_feature_extraction_job(
        coords_dir=f'{magnification}x_{patch_size}px_{0}px_overlap',
        patch_encoder=encoder,
        patch_size=patch_size,
        target_mag=magnification,
        device=f'cuda:{gpu}'
    )

    # Embedding slide task 
    df_result = pd.DataFrame()
    encoder = encoder_factory(slide_model)
    processor.run_slide_feature_extraction_job(
        slide_encoder=encoder,
        coords_dir=f'{magnification}x_{patch_size}px_{0}px_overlap',
        device=f'cuda:{gpu}',
        results_df=df_result
    )