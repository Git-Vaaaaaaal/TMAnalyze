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

from trident import load_wsi, WSIReaderType
from trident.IO import create_lock, remove_lock, is_locked, update_log, collect_valid_slides
from trident.Maintenance import deprecated
from trident.wsi_objects.WSIFactory import OPENSLIDE_EXTENSIONS, PIL_EXTENSIONS, SDPC_EXTENSIONS


class Processor:

    def __init__(
        self,
        job_dir: str,
        wsi_source: str,
        wsi_ext: List[str] = None,
        wsi_cache: Optional[str] = None,
        clear_cache: bool = False,
        skip_errors: bool = False,
        custom_mpp_keys: Optional[List[str]] = None,
        custom_list_of_wsis: Optional[str] = None,
        max_workers: Optional[int] = None,
        reader_type: Optional[WSIReaderType] = None,
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
                The directory where the results of processing, including segmentations, patches, and extracted features, 
                will be saved. This should be an existing directory with sufficient storage.
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
        self.wsi_source = wsi_source
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
            wsi_ext=self.wsi_ext,
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

            slide = load_wsi(
                slide_path=abs_path,
                name=name,
                tissue_seg_path=tissue_seg_path,
                custom_mpp_keys=self.custom_mpp_keys,
                mpp=valid_mpps[wsi_idx] if valid_mpps is not None else None,
                max_workers=self.max_workers,
                reader_type=reader_type,
            )
            self.wsis.append(slide)

    def run_segmentation_job(
        self, 
        segmentation_model: torch.nn.Module, 
        seg_mag: int = 10, 
        holes_are_tissue: bool = False,
        batch_size: int = 16,
        artifact_remover_model: torch.nn.Module = None,
        device: str = 'cuda:0', 
    ) -> str:
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
            binary = np.where(wsi_mask == 0, 255, 0).astype(np.uint8)
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
            filename = getattr(wsi, 'name', f"wsi_{id(wsi)}") + ".geojson"
            with open(filename, 'w') as f:
                json.dump({
                    "type": "FeatureCollection",
                    "name": filename,
                    "features": features
                }, f)
                
        # Return the directory where the contours are saved
        return saveto

    def run_patching_job(
        self, 
        target_magnification: int, 
        patch_size: int, 
        overlap: int = 0, 
        saveto: str | None = None, 
        min_tissue_proportion: float = 0.,
    ) -> str:
        if saveto is None:
            saveto = f"{target_magnification}x_{patch_size}px_{overlap}px_overlap"

        self.target_magnification = target_magnification


        os.makedirs(os.path.join(self.job_dir, saveto, 'patches'), exist_ok=True)

        sig = signature(self.run_patching_job)
        local_attrs = {k: v for k, v in locals().items() if k in sig.parameters}

        self.loop = tqdm(self.wsis, desc=f'Saving tissue coordinates to {saveto}', total=len(self.wsis))
        for wsi in self.loop:

            # ← CSV au lieu du H5
            csv_path = os.path.join(self.job_dir, saveto, 'patches', f'{wsi.name}_patches.csv')

            if os.path.exists(csv_path):
                self.loop.set_postfix_str(f'Patch coords already generated for {wsi.name}. Skipping...')
                update_log(os.path.join(self.job_dir, saveto, '_logs_coords.txt'), f'{wsi.name}{wsi.ext}', 'Coords generated')
                continue

            if is_locked(csv_path):
                self.loop.set_postfix_str(f'{wsi.name} is locked. Skipping...')
                continue

            if wsi.tissue_seg_path is None or not os.path.exists(wsi.tissue_seg_path):
                self.loop.set_postfix_str(f'GeoJSON not found for {wsi.name}. Skipping...')
                update_log(os.path.join(self.job_dir, saveto, '_logs_coords.txt'), f'{wsi.name}{wsi.ext}', 'GeoJSON not found.')
                continue

            gdf = gpd.read_file(wsi.tissue_seg_path, rows=1)
            if gdf.empty:
                self.loop.set_postfix_str(f'Empty GeoDataFrame for {wsi.name}. Skipping...')
                update_log(os.path.join(self.job_dir, saveto, '_logs_coords.txt'), f'{wsi.name}{wsi.ext}', 'Empty GeoDataFrame.')
                continue

            try:
                self.loop.set_postfix_str(f'Generating patch coords for {wsi.name}{wsi.ext}')
                update_log(os.path.join(self.job_dir, saveto, '_logs_coords.txt'), f'{wsi.name}{wsi.ext}', 'LOCKED. Generating coords...')
                create_lock(csv_path)

                # extract_tissue_coords doit retourner un np.ndarray (N, 2) avec [x, y]
                coords = wsi.extract_tissue_coords(
                    target_mag=target_magnification,
                    patch_size=patch_size,
                    save_coords=os.path.join(self.job_dir, saveto),
                    overlap=overlap,
                    min_tissue_proportion=min_tissue_proportion,
                )

                pd.DataFrame(coords, columns=['x', 'y']).to_csv(csv_path, index=False)


                wsi.release() 

            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    remove_lock(csv_path)
                try:
                    wsi.release()
                except Exception:
                    pass

        return os.path.join(self.job_dir, saveto)

    @deprecated
    def run_feature_extraction_job(
        self, 
        coords_dir: str, 
        patch_encoder: torch.nn.Module, 
        device: str,  
        batch_limit: int = 512, 
        saveto: str | None = None
    ) -> str:
        self.run_patch_feature_extraction_job(
            coords_dir, 
            patch_encoder, 
            device, 
            batch_limit, 
            saveto,
        )
        
    def run_patch_feature_extraction_job(
        self, 
        coords_dir: str, 
        patch_encoder: torch.nn.Module, 
        patch_size: int,    
        target_mag: int,   #ADDED
        device: str, 
        batch_limit: int = 512, 
        saveto: str | None = None
    ) -> str:
        # Le paramètre saveas est supprimé : la sortie est toujours CSV
        if saveto is None:
            saveto = os.path.join(coords_dir, f'features_{patch_encoder.enc_name}')

        os.makedirs(os.path.join(self.job_dir, saveto), exist_ok=True)

        sig = signature(self.run_patch_feature_extraction_job)
        local_attrs = {k: v for k, v in locals().items() if k in sig.parameters}

        #log_fp = os.path.join(self.job_dir, coords_dir, f'_logs_feats_{patch_encoder.enc_name}.txt')
        self.loop = tqdm(self.wsis, desc=f'Extracting patch features from coords in {coords_dir}', total=len(self.wsis))
        for wsi in self.loop:

            csv_path = os.path.join(self.job_dir, saveto, f'{wsi.name}.csv')

            if os.path.exists(csv_path) and not is_locked(csv_path):
                self.loop.set_postfix_str(f'Features already extracted for {wsi}. Skipping...')
                #update_log(log_fp, f'{wsi.name}{wsi.ext}', 'Features extracted.')
                continue

            # ← Lecture des coordonnées depuis le CSV produit par run_patching_job
            coords_path = os.path.join(self.job_dir, coords_dir, 'patches', f'{wsi.name}_patches.csv')
            if not os.path.exists(coords_path):
                self.loop.set_postfix_str(f'Coords not found for {wsi.name}. Skipping...')
                #update_log(log_fp, f'{wsi.name}{wsi.ext}', 'Coords not found.')
                continue

            if is_locked(csv_path):
                self.loop.set_postfix_str(f'{wsi.name} is locked. Skipping...')
                continue

            try:
                self.loop.set_postfix_str(f'Extracting features from {wsi.name}{wsi.ext}')
                create_lock(csv_path)
                #update_log(log_fp, f'{wsi.name}{wsi.ext}', 'LOCKED. Extracting features...')

                # extract_patch_features doit retourner un np.ndarray (N, D)
                features = wsi.extract_patch_features(
                    patch_encoder=patch_encoder,
                    coords_path=coords_path,
                    save_features=os.path.join(self.job_dir, saveto),
                    patch_size=patch_size,
                    target_mag=target_mag,
                    device=device,
                    batch_limit=batch_limit,
                )

                # ← Lecture des coordonnées x, y pour les coller en tête du CSV
                coords = pd.read_csv(coords_path)[['x', 'y']].values
                feat_cols = [f'feat_{i}' for i in range(features.shape[1])]
                df = pd.DataFrame(
                    data=np.hstack([coords, features]),
                    columns=['x', 'y'] + feat_cols,
                )
                df.to_csv(csv_path, index=False)

                remove_lock(csv_path)
                #update_log(log_fp, f'{wsi.name}{wsi.ext}', 'Features extracted.')
                wsi.release()

            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    remove_lock(csv_path)
                try:
                    wsi.release()
                except Exception:
                    pass
                """ if self.skip_errors:
                    #update_log(log_fp, f'{wsi.name}{wsi.ext}', f'ERROR: {e}')
                    continue
                else:
                    raise e """

        return os.path.join(self.job_dir, saveto)

    def run_slide_feature_extraction_job(
        self,
        slide_encoder: torch.nn.Module,
        coords_dir: str,
        device: str = 'cuda',
        batch_limit: int = 512,
        saveto: str | None = None,
        results_df: pd.DataFrame | None = None,  # ← nouveau paramètre
    ) -> str:
        """
        ... (docstring inchangée) ...
        
        Args:
            ...
            results_df (pd.DataFrame | None): DataFrame optionnel dans lequel les features
                slide extraites sont accumulées. Chaque ligne correspond à un WSI, avec
                les colonnes 'wsi_name' suivies des features. Si None, aucun DataFrame
                n'est rempli. Le DataFrame est modifié in-place ET retourné.
        """
        from trident.slide_encoder_models.load import slide_to_patch_encoder_name

        if slide_encoder.enc_name.startswith('mean-'):
            slide_to_patch_encoder_name[slide_encoder.enc_name] = slide_encoder.enc_name.split('mean-')[1]

        # Setting I/O
        mustbe_patch_encoder = slide_to_patch_encoder_name[slide_encoder.enc_name]
        patch_features_dir = os.path.join(coords_dir, f'features_{mustbe_patch_encoder}')
        if saveto is None:
            saveto = os.path.join(coords_dir, f'slide_features_{slide_encoder.enc_name}')
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
                coords_dir=coords_dir,
                patch_encoder=patch_encoder,
                device=device,
                batch_limit=batch_limit,
                # ← saveas supprimé : run_patch_feature_extraction_job produit toujours du CSV
            )

        sig = signature(self.run_slide_feature_extraction_job)
        local_attrs = {k: v for k, v in locals().items() if k in sig.parameters}
        """ self.save_config(
            saveto=os.path.join(self.job_dir, coords_dir, f'_config_slide_features_{slide_encoder.enc_name}.json'),
            local_attrs=local_attrs,
            ignore=['loop', 'valid_slides', 'wsis', 'results_df']  # ← results_df ignoré dans le config
        ) """

        #log_fp = os.path.join(self.job_dir, coords_dir, f'_logs_slide_features_{slide_encoder.enc_name}.txt')
        collected_rows = []  # ← accumulateur local avant merge dans results_df

        self.loop = tqdm(self.wsis, desc=f'Extracting slide features using {slide_encoder.enc_name}', total=len(self.wsis))
        for wsi in self.loop:

            # ← Fichier de sortie slide en CSV
            slide_feature_path = os.path.join(self.job_dir, saveto, f'{wsi.name}.csv')

            if os.path.exists(slide_feature_path) and not is_locked(slide_feature_path):
                self.loop.set_postfix_str(f'Slide features already extracted for {wsi.name}. Skipping...')
                #update_log(log_fp, f'{wsi.name}{wsi.ext}', 'Slide features extracted.')
                # ← Relire le CSV existant pour l'intégrer dans results_df
                if results_df is not None:
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
                )

                # ← Sauvegarde CSV : une ligne = un WSI, colonnes wsi_name + slide features
                n_feats = slide_features.shape[0] if slide_features.ndim == 1 else slide_features.shape[1]
                feat_cols = [f'slide_feat_{i}' for i in range(n_feats)]
                row_data = np.atleast_1d(slide_features).reshape(1, -1)
                row_df = pd.DataFrame(row_data, columns=feat_cols)
                row_df.insert(0, 'wsi_name', wsi.name)
                row_df.to_csv(slide_feature_path, index=False)

                if results_df is not None:
                    collected_rows.append(row_df)

                remove_lock(slide_feature_path)
                #update_log(log_fp, f'{wsi.name}{wsi.ext}', 'Slide features extracted.')
                wsi.release()

            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    remove_lock(slide_feature_path)
                try:
                    wsi.release()
                except Exception:
                    pass
                """ if self.skip_errors:
                    update_log(log_fp, f'{wsi.name}{wsi.ext}', f'ERROR: {e}')
                    continue
                else:
                    raise e """

        # ← Merge de toutes les lignes collectées dans le DataFrame fourni en entrée
        if results_df is not None and collected_rows:
            new_rows = pd.concat(collected_rows, ignore_index=True)
            for col in new_rows.columns:
                results_df[col] = results_df.get(col, pd.NA)
            results_df = pd.concat([results_df, new_rows], ignore_index=True)

        return os.path.join(self.job_dir, saveto), results_df

    """ def save_config(
        self,
        saveto: str,
        local_attrs: Optional[Dict[str, Any]] = None,
        ignore: List[str] = ['valid_slides']
    ) -> None:
        The `save_config` function saves the current configuration of the `Processor` instance to a JSON file. 
        This configuration includes attributes of the instance as well as optional additional parameters 
        provided via the `local_attrs` argument.

        The function filters out attributes specified in the `ignore` list and ensures that only JSON-serializable 
        attributes are included. This makes it ideal for saving configurations in a structured format that can 
        later be reloaded or inspected for reproducibility.

        Parameters:
            saveto (str): 
                The path to the file where the configuration will be saved. This should include the file extension 
                (e.g., "config.json").
            local_attrs (dict, optional): 
                A dictionary of additional attributes to include in the configuration. This can be used to add 
                method-specific parameters or runtime settings. Defaults to None.
            ignore (list, optional): 
                A list of attribute names to exclude from the configuration. This is useful for omitting large 
                or non-serializable objects. Defaults to ['valid_slides'].

        Returns:
            None: The function saves the configuration to the specified file and does not return any value.

        Example
        -------
        Save the current processor configuration to a file:

        >>> processor.save_config(saveto="output/config.json")
        >>> # Check the saved configuration
        >>> with open("output/config.json", "r") as f:
        ...     config = json.load(f)
        ...     print(config)
        from trident.IO import JSONsaver

        def serialize_safe(obj):
            try:
                return json.loads(json.dumps(obj))  # Ensure the object is JSON-serializable
            except (TypeError, OverflowError):
                return None

        # Merge instance attributes and local_attrs, filtering ignored and unserializable items
        config = {
            k: serialize_safe(v)
            for attr_dict in [vars(self), local_attrs or {}]
            for k, v in attr_dict.items()
            if k not in ignore and serialize_safe(v) is not None
        }

        # Save the combined configuration to the specified file
        with open(saveto, 'w') as f:
            json.dump(config, f, indent=4, cls=JSONsaver) """

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
