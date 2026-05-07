from __future__ import annotations
import os
import sys
import torch
from tqdm import tqdm
from typing import Optional, List, Dict, Any
from inspect import signature
import geopandas as gpd
import pandas as pd 
import numpy as np
import TMAx
import cv2
import json

from trident.IO import create_lock, remove_lock, is_locked, update_log, collect_valid_slides
from trident.Maintenance import deprecated
from trident.wsi_objects.WSIFactory import OPENSLIDE_EXTENSIONS, PIL_EXTENSIONS, SDPC_EXTENSIONS
from src.class_wsi_claude import WSI  # ← ta classe personnalisée
import openslide

import os
import time
import logging
import traceback
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from inspect import signature

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

class Processor:

    def __init__(
        self,
        job_dir: str,
        wsi_source: str,
        wsi_ext: List[str] = None,
        wsi_cache: Optional[str]= None,
        clear_cache: bool = False,
        skip_errors: bool = False,
        custom_mpp_keys: Optional[List[str]] = None,
        custom_list_of_wsis: Optional[str] = None,
        max_workers: Optional[int] = None,
        search_nested: bool = False, 
    ) -> None:
        """
        The `Processor` class handles all preprocessing steps starting from whole-slide images (WSIs). 
    
        Available methods:
            - `run_segmentation_job`: Performs tissue segmentation on all slides managed by the processor.
            - `run_patching_job`: Extracts patch coordinates from the segmented tissue regions of slides.
            - `run_patch_feature_extraction_job`: Extracts patch-level features using a specified patch encoder.
                - Deprecated alias: `run_feature_extraction_job`
            - `run_slide_feature_extraction_job`: Extracts slide-level features using a specified slide encoder.
            
        Parameters:
            job_dir (str): 
                Directory where image and results are stored
            wsi_source (str): 
                The directory containing the WSIs to be processed. This can either be a local directory 
                or a network-mounted drive. All slides in this directory matching the specified file 
                extensions will be considered for processing.
            wsi_ext (List[str]): 
                A list of accepted WSI file extensions, such as ['.ndpi', '.svs']. This allows for 
                filtering slides based on their format. If set to None, a default list of common extensions 
                will be used. Defaults to None.
            wsi_cache (str, optional): 
                [DEPRECATED as of v0.2.0] An optional directory for caching WSIs locally. If specified, slides will be copied 
                from the source directory to this local directory before processing, improving performance 
                when the source is a network drive. Defaults to None.
            clear_cache (bool, optional):
                [DEPRECATED as of v0.2.0] A flag indicating whether slides in the cache should be deleted after processing. 
                This helps manage storage space. Defaults to False. 
            skip_errors (bool, optional): 
                A flag specifying whether to continue processing if an error occurs on a slide. 
                If set to False, the process will stop on the first error. Defaults to False.
            custom_mpp_keys (List[str], optional): 
                A list of custom keys in the slide metadata for retrieving the microns per pixel (MPP) value. 
                If not provided, standard keys will be used. Defaults to None.
            custom_list_of_wsis (str, optional): 
                Path to a csv file with a custom list of WSIs to process in a field called 'wsi' (including extensions). If provided, only 
                these slides will be considered for processing. Defaults to None, which means all 
                slides matching the wsi_ext extensions will be processed.
                Note: If `custom_list_of_wsis` is provided, any names that do not match the available slides will be ignored, and a warning will be printed.
            max_workers (int, optional):
                Maximum number of workers for data loading. If None, the default behavior will be used.
                Defaults to None.
            reader_type (WSIReaderType, optional):
                Force the image reader engine to use. Options are are ["openslide", "image", "cucim"]. Defaults to None
                (auto-determine the right engine based on image extension).
            search_nested (bool, optional):  
                If True, the processor will recursively search for WSIs within all subdirectories of `wsi_source`.
                All matching files (based on `wsi_ext`) found at any depth within the directory  
                tree will be included. Each slide will be identified by its relative path to `wsi_source`, but only  
                the filename (excluding directory structure) will be used for downstream outputs (e.g., segmentation filenames).  
                If False, only files directly inside `wsi_source` will be considered.  
                Defaults to False.


        Returns:
            None: This method initializes the class instance and sets up the environment for processing.

        Example
        -------
        Initialize the `Processor` for a directory of WSIs:

        >>> processor = Processor(
        ...     job_dir="results/",
        ...     wsi_source="data/slides/",
        ...     wsi_ext=[".svs", ".ndpi"],
        ... )
        >>> print(f"Processor initialized for {len(processor.wsis)} slides.")

        Raises:
            AssertionError: If `wsi_ext` is not a list or if any extension does not start with a period.
        """
        
        if not (sys.version_info.major >= 3 and sys.version_info.minor >= 9):
            raise EnvironmentError("Trident requires Python 3.9 or above. Python 3.10 is recommended.")

        self.job_dir = job_dir
        self.wsi_source = os.path.join(job_dir, wsi_source)
        self.wsi_ext = wsi_ext or (list(PIL_EXTENSIONS) + list(OPENSLIDE_EXTENSIONS) + list(SDPC_EXTENSIONS))
        self.skip_errors = skip_errors
        self.custom_mpp_keys = custom_mpp_keys
        self.max_workers = max_workers

        # Validate extensions
        assert isinstance(self.wsi_ext, list), f'wsi_ext must be a list, got {type(self.wsi_ext)}'
        for ext in self.wsi_ext:
            assert ext.startswith('.'), f'Invalid extension: {ext} (must start with a period)'

        # === Collect slide paths and relative paths ===
        full_paths, rel_paths = collect_valid_slides(
            wsi_dir=wsi_source,
            custom_list_path=custom_list_of_wsis,
            wsi_ext=wsi_ext,
            search_nested=search_nested,
            max_workers=max_workers,
            return_relative_paths=True
        )

        self.wsi_rel_paths = rel_paths if custom_list_of_wsis else None

        # === Extract mpp column if provided ===
        if custom_list_of_wsis is not None:
            wsi_df = pd.read_csv(custom_list_of_wsis)
            valid_mpps = (
                wsi_df['mpp'].dropna().tolist()
                if 'mpp' in wsi_df.columns else None
            )
        else:
            valid_mpps = None

        print(f'[PROCESSOR] Found {len(full_paths)} valid slides in {wsi_source}.')

        # === Initialize WSIs ===
        self.wsis = []
        for wsi_idx, abs_path in enumerate(full_paths):
            name = os.path.basename(abs_path)
            tissue_seg_path = os.path.join(
                self.job_dir, 'contours_geojson',
                f'{os.path.splitext(name)[0]}.geojson'
            )
            if not os.path.exists(tissue_seg_path):
                tissue_seg_path = None

            slide = WSI(
                slide_path=abs_path,
                name=name,
                tissue_seg_path=tissue_seg_path,
                custom_mpp_keys=self.custom_mpp_keys,
                mpp=valid_mpps[wsi_idx] if valid_mpps is not None else None,
                max_workers=self.max_workers,
                lazy_init=True,
            )
            self.wsis.append(slide)

    def run_segmentation_job(self) -> str:
        """
        The `run_segmentation_job` function performs tissue segmentation on all slides managed by the processor. 
        It uses a machine learning model to identify tissue regions and saves the resulting segmentations to the 
        output directory. This function is essential for workflows that require detailed tissue delineation.

        Parameters:
            segmentation_model (torch.nn.Module): 
                A pre-trained PyTorch model that performs the tissue segmentation. This model should be compatible 
                with the expected input data format of WSIs.
            seg_mag (int, optional): 
                The magnification level at which segmentation is performed. For example, a value of 10 indicates 
                10x magnification. Defaults to 10.
            holes_are_tissue (bool, optional): 
                Specifies whether to treat holes within tissue regions as part of the tissue. Defaults to False.
            batch_size (int, optional): 
                The batch size for segmentation. Defaults to 16.
            artifact_remover_model (torch.nn.Module, optional): 
                A pre-trained PyTorch model that can remove artifacts from an existing segmentation. Defaults to None.
            device (str): 
                The computation device to use (e.g., 'cuda:0' for GPU or 'cpu' for CPU).

        Returns:
            str: Absolute path to where directory containing contours is saved.

        Example
        -------
        Run a segmentation job with a pre-trained model:

        >>> from segmentation.models import TissueSegmenter
        >>> model = TissueSegmenter()
        >>> processor.run_segmentation_job(segmentation_model=model, seg_mag=20)
        """

        def get_ring(cnt):
            """Approximation, formatage des coordonnées et fermeture du polygone."""
            approx = cv2.approxPolyDP(cnt, epsilon, closed=True) if epsilon > 0 else cnt
            coords = approx.reshape(-1, 2).tolist()
            if coords[0] != coords[-1]: coords.append(coords[0])
            return coords

        saveto = os.path.join(self.job_dir, 'geojson_contours')
        os.makedirs(saveto, exist_ok=True)
        for wsi in self.wsis:
            wsi_mask = TMAx.predict_mask(wsi)
            binary = np.where(wsi_mask > 0, 255, 0).astype(np.uint8)
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            
            features = []
            min_area, epsilon = 100.0, 1.0
            if hierarchy is not None:
                for i, (cnt, h) in enumerate(zip(contours, hierarchy[0])):
                    # Ignorer les trous (h[3] != -1) et les zones trop petites
                    if h[3] != -1 or cv2.contourArea(cnt) < min_area: 
                        continue

                    rings = [get_ring(cnt)]

                    # Traitement des trous
                    child = h[2]
                    while child != -1:
                        if cv2.contourArea(contours[child]) >= min_area:
                            rings.append(get_ring(contours[child]))
                        child = hierarchy[0][child][0]

                    features.append({
                        "type": "Feature",
                        "properties": {"tissue_id": len(features) + 1},
                        "geometry": {"type": "Polygon", "coordinates": rings},
                    })

            # Enregistrement
            filename = os.path.join(saveto, getattr(wsi, 'name', f"wsi_{id(wsi)}") + ".geojson")
            with open(filename, 'w') as f:
                json.dump({
                    "type": "FeatureCollection",
                    "name": filename,
                    "features": features
                }, f)
            wsi.tissue_seg_path = filename

        # Return the directory where the contours are saved
        return saveto

    def run_patching_job(
        self, 
        target_magnification: int, 
        patch_size: int, 
        overlap: int = 0, 
        min_tissue_proportion: float = 0.,
    ) -> str:


        self.target_magnification = target_magnification

        logger.info("🚀 Starting patching job")
        logger.info(f"Total WSIs: {len(self.wsis)}")
        logger.info(f"Magnification: {target_magnification}, Patch size: {patch_size}, Overlap: {overlap}")

        os.makedirs(os.path.join(self.job_dir, 'patches'), exist_ok=True)

        self.loop = tqdm(self.wsis, desc=f'Saving tissue coordinates to {self.job_dir}', total=len(self.wsis))

        for wsi in self.loop:

            csv_path = os.path.join(self.job_dir, 'patches', f'{wsi.name}_patches.csv')

            try:
                logger.info("🔒 Creating lock")

                logger.info("⚙️ Running extract_tissue_coords...")
                start = time.time()

                coords = wsi.extract_tissue_coords(
                    target_mag=target_magnification,
                    patch_size=patch_size,
                    save_coords=os.path.join(self.job_dir, "patches"),
                    overlap=overlap,
                    min_tissue_proportion=min_tissue_proportion,
                )
                df_coords = pd.DataFrame(coords, columns=['x', 'y'])
                df_coords.to_csv(csv_path, index=False)

                elapsed = time.time() - start
                
            except:
                try:
                    wsi.release()
                except Exception:
                    pass

                if self.skip_errors:
                    continue
                else:
                    raise

        return os.path.join(self.job_dir, "patches")

    @deprecated
    def run_feature_extraction_job(
        self, 
        coords_dir: str, 
        patch_encoder: torch.nn.Module, 
        device: str,  
        batch_limit: int = 512, 
    ) -> str:
        self.run_patch_feature_extraction_job(
            coords_dir, 
            patch_encoder, 
            device, 
            batch_limit, 
        )
        
    def run_patch_feature_extraction_job(
        self,
        patch_encoder: torch.nn.Module,
        patch_size: int,
        target_mag: int,
        device: str,
        batch_limit: int = 512,
    ) -> str:

        save_dir = os.path.join(self.job_dir, f'features_{patch_encoder.enc_name}')
        os.makedirs(save_dir, exist_ok=True)

        self.loop = tqdm(
            self.wsis,
            desc='Extracting patch features',
            total=len(self.wsis)
        )

        for wsi in self.loop:

            csv_path = os.path.join(save_dir, f'{wsi.name}.csv')

            if os.path.exists(csv_path) and not is_locked(csv_path):
                self.loop.set_postfix_str(f'{wsi.name}: already done')
                continue

            coords_path = os.path.join(
                self.job_dir,
                'patches',
                f'{wsi.name}_patches.csv'
            )

            if not os.path.exists(coords_path):
                self.loop.set_postfix_str(f'{wsi.name}: coords not found')
                continue

            if is_locked(csv_path):
                self.loop.set_postfix_str(f'{wsi.name}: locked')
                continue

            try:
                self.loop.set_postfix_str(f'{wsi.name}: extracting')
                create_lock(csv_path)

                features = wsi.extract_patch_features(
                    patch_encoder=patch_encoder,
                    coords_path=coords_path,
                    save_features=save_dir,
                    patch_size=patch_size,
                    target_mag=target_mag,
                    device=device,
                    batch_limit=batch_limit,
                )

                coords = pd.read_csv(coords_path)[['x', 'y']].values
                feat_cols = [f'feat_{i}' for i in range(features.shape[1])]

                df = pd.DataFrame(
                    np.hstack([coords, features]),
                    columns=['x', 'y'] + feat_cols,
                )

                df.to_csv(csv_path, index=False)

                remove_lock(csv_path)
                wsi.release()

            except Exception as e:
                remove_lock(csv_path)
                try:
                    wsi.release()
                except Exception:
                    pass
                raise e

        return save_dir

    def run_slide_feature_extraction_job(
        self,
        slide_encoder: torch.nn.Module,
        patch_size: int,
        target_mag: int,
        slide_model: str,
        device: str = 'cuda',
        batch_limit: int = 512,
        saveto: str | None = None,
    ) -> str:
        
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

        patch_model_name = slide_to_patch_encoder_name.get(slide_model)
        patch_features_dir = f'features_{patch_model_name}'


        from trident.slide_encoder_models.load import slide_to_patch_encoder_name

        #Slide encoder name
        if slide_encoder.enc_name.startswith('mean-'):
            slide_to_patch_encoder_name[slide_encoder.enc_name] = slide_encoder.enc_name.split('mean-')[1]


        if saveto is None:
            saveto = f'slide_features_{slide_encoder.enc_name}'
        os.makedirs(os.path.join(self.job_dir, saveto), exist_ok=True)

        already_processed = []
        if os.path.isdir(os.path.join(self.job_dir, patch_features_dir)):
            already_processed = [
                os.path.splitext(x)[0]
                for x in os.listdir(os.path.join(self.job_dir, patch_features_dir))
                if x.endswith('.csv')  # ← était saveas / '.h5'
            ]
            wsi_names = [slide.name for slide in self.wsis]
            already_processed = [x for x in already_processed if x in wsi_names]

        if len(already_processed) < len(self.wsis):
            print(f"[PROCESSOR] Some patch features haven't been extracted in {len(already_processed)}/{len(self.wsis)} WSIs. Starting extraction.")
            from trident.patch_encoder_models.load import encoder_factory
            patch_encoder = encoder_factory(slide_to_patch_encoder_name[slide_encoder.enc_name])
            self.run_patch_feature_extraction_job(
                patch_encoder=patch_encoder,
                patch_size=patch_size, 
                target_mag=target_mag,
                device=device,
                batch_limit=batch_limit,
                # ← saveas supprimé : run_patch_feature_extraction_job produit toujours du CSV
            )

        sig = signature(self.run_slide_feature_extraction_job)
        local_attrs = {k: v for k, v in locals().items() if k in sig.parameters}

        collected_rows = [] 

        self.loop = tqdm(self.wsis, desc=f'Extracting slide features using {slide_encoder.enc_name}', total=len(self.wsis))
        for wsi in self.loop:

            # ← Fichier de sortie slide en CSV
            slide_feature_path = os.path.join(self.job_dir, saveto, f'{wsi.name}.csv')

            if os.path.exists(slide_feature_path) and not is_locked(slide_feature_path):
                self.loop.set_postfix_str(f'Slide features already extracted for {wsi.name}. Skipping...')
                #update_log(log_fp, f'{wsi.name}{wsi.ext}', 'Slide features extracted.')
                # ← Relire le CSV existant pour l'intégrer dans results_df

                row = pd.read_csv(slide_feature_path)
                collected_rows.append(row)
                continue

            # ← Patch features en CSV au lieu de H5
            patch_features_path = os.path.join(self.job_dir, patch_features_dir, f'{wsi.name}.csv')
            if not os.path.exists(patch_features_path):
                self.loop.set_postfix_str(f'Patch features not found for {wsi.name}. Skipping...')
                #update_log(log_fp, f'{wsi.name}{wsi.ext}', 'Patch features not found.')
                continue

            if is_locked(slide_feature_path):
                self.loop.set_postfix_str(f'{wsi.name} is locked. Skipping...')
                continue

            try:
                self.loop.set_postfix_str(f'Extracting slide features for {wsi.name}{wsi.ext}')
                create_lock(slide_feature_path)
                #update_log(log_fp, f'{wsi.name}{wsi.ext}', 'LOCKED. Extracting slide features...')

                # ← Charge le CSV de patch features (x, y, feat_0, ..., feat_D)
                patch_df = pd.read_csv(patch_features_path)
                coords  = patch_df[['x', 'y']].values                          # (N, 2)
                features = patch_df.drop(columns=['x', 'y']).values            # (N, D)

                # extract_slide_features reçoit coords + features séparément
                slide_features = wsi.extract_slide_features(
                    coords=coords,
                    features=features,
                    slide_encoder=slide_encoder,
                    device=device,
                    save_features=os.path.join(self.job_dir, saveto),
                    patch_size=patch_size,
                    target_mag=target_mag,
                )

                # ← Sauvegarde CSV : une ligne = un WSI, colonnes wsi_name + slide features
                n_feats = slide_features.shape[0] if slide_features.ndim == 1 else slide_features.shape[1]
                feat_cols = [f'slide_feat_{i}' for i in range(n_feats)]
                row_data = np.atleast_1d(slide_features).reshape(1, -1)
                row_df = pd.DataFrame(row_data, columns=feat_cols)
                row_df.insert(0, 'wsi_name', wsi.name)

                collected_rows.append(row_df)

                remove_lock(slide_feature_path)
                wsi.release()

            except Exception as e:
                remove_lock(slide_feature_path)
                try:
                    wsi.release()
                except Exception:
                    pass
                print(f"[ERROR] Failed on {wsi.name}: {e}")


        if len(collected_rows) == 0:
            print("[WARNING] No slide features were collected. Nothing will be saved.")
            return os.path.join(self.job_dir, saveto)

        results_df = pd.concat(collected_rows, ignore_index=True)
        save_df = results_df.to_csv(
            os.path.join(self.job_dir, saveto, f"{slide_model}_encoder.csv"),
            index=False
        )

        return save_df

    def release(self) -> None:
        """
        Release all resources tied to the WSIs held by this Processor instance.
        Frees memory, closes file handles, and clears GPU memory.
        Should be called after processing is complete to avoid memory leaks.
        """
        if hasattr(self, "wsis"):
            for wsi in self.wsis:
                try:
                    wsi.release()
                except Exception:
                    pass
            self.wsis.clear()

        # Also clear loop references (e.g., tqdm)
        if hasattr(self, "loop"):
            self.loop = None

        # Explicit garbage collection and CUDA cache release
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()

