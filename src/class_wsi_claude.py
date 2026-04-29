from __future__ import annotations
import os
import warnings
import numpy as np
import torch
import openslide
from PIL import Image
from typing import List, Tuple, Optional, Literal, Union
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import geopandas as gpd

from trident.segmentation_models.load import SegmentationModel
from trident.wsi_objects.WSIPatcher import WSIPatcher
from trident.wsi_objects.WSIPatcherDataset import WSIPatcherDataset
from trident.IO import (
    mask_to_gdf, overlay_gdf_on_thumbnail, get_num_workers
)

ReadMode = Literal['pil', 'numpy']


class WSI:
    """
    Whole Slide Image class with OpenSlide backend integrated directly.
    No dependency on trident's load_wsi or OpenSlideWSI.

    Attributes
    ----------
    slide_path : str
        Absolute path to the WSI file.
    name : str
        Slide name without extension.
    ext : str
        Slide file extension.
    tissue_seg_path : str or None
        Path to the GeoJSON tissue segmentation file.
    gdf_contours : gpd.GeoDataFrame or None
        Loaded tissue contours (None until available).
    mpp : float or None
        Microns per pixel.
    mag : int or None
        Estimated magnification level.
    """

    def __init__(
        self,
        slide_path: str,
        name: Optional[str] = None,
        tissue_seg_path: Optional[str] = None,
        custom_mpp_keys: Optional[List[str]] = None,
        lazy_init: bool = True,
        mpp: Optional[float] = None,
        max_workers: Optional[int] = None,
    ):
        self.slide_path = slide_path

        raw_name = name if name is not None else os.path.basename(slide_path)
        self.name, self.ext = os.path.splitext(raw_name)

        self.tissue_seg_path = tissue_seg_path
        self.custom_mpp_keys = custom_mpp_keys
        self.mpp = mpp
        self.mag = None
        self.max_workers = max_workers
        self._initialized = False

        # Placeholders — set during _lazy_initialize
        self.img = None
        self.width = None
        self.height = None
        self.dimensions = None
        self.level_count = None
        self.level_downsamples = None
        self.level_dimensions = None
        self.properties = None

        # Load GeoJSON contours immediately if available
        self.gdf_contours = self._load_gdf(tissue_seg_path)

        if not lazy_init:
            self._lazy_initialize()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_gdf(self, path: Optional[str]) -> Optional[gpd.GeoDataFrame]:
        """Load a GeoJSON file into a GeoDataFrame, or return None."""
        if path and os.path.exists(path):
            try:
                return gpd.read_file(path)
            except Exception as e:
                warnings.warn(f"Could not load GeoJSON at {path}: {e}")
        return None

    def _lazy_initialize(self) -> None:
        """Open the slide with OpenSlide and populate all metadata attributes."""
        if self._initialized:
            return
        try:
            self.img = openslide.OpenSlide(self.slide_path)
            self.dimensions = self.img.dimensions          # (width, height)
            self.width, self.height = self.dimensions
            self.level_count = self.img.level_count
            self.level_downsamples = self.img.level_downsamples
            self.level_dimensions = self.img.level_dimensions
            self.properties = self.img.properties

            if self.mpp is None:
                self.mpp = self._fetch_mpp(self.custom_mpp_keys)
            self.mag = self._fetch_magnification(self.custom_mpp_keys)

            # Reload GeoJSON now if it was not available at construction time
            if self.gdf_contours is None and self.tissue_seg_path:
                self.gdf_contours = self._load_gdf(self.tissue_seg_path)

            self._initialized = True

        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize WSI '{self.slide_path}' with OpenSlide: {e}"
            ) from e

    def _fetch_mpp(self, custom_mpp_keys: Optional[List[str]] = None) -> float:
        """Extract microns-per-pixel from OpenSlide metadata."""
        mpp_keys = [
            openslide.PROPERTY_NAME_MPP_X,
            'openslide.mirax.MPP',
            'aperio.MPP',
            'hamamatsu.XResolution',
            'openslide.comment',
        ]
        if custom_mpp_keys:
            mpp_keys.extend(custom_mpp_keys)

        for key in mpp_keys:
            if key in self.img.properties:
                try:
                    return round(float(self.img.properties[key]), 4)
                except ValueError:
                    continue

        x_res = self.img.properties.get('tiff.XResolution')
        unit = self.img.properties.get('tiff.ResolutionUnit')
        if x_res and unit:
            try:
                if unit.lower() == 'centimeter':
                    return round(10000 / float(x_res), 4)
                elif unit.upper() == 'INCH':
                    return round(25400 / float(x_res), 4)
            except ValueError:
                pass

        raise ValueError(
            f"Unable to extract MPP from slide metadata: '{self.slide_path}'.\n"
            "Provide `custom_mpp_keys` or set `mpp` explicitly in the constructor."
        )

    def _fetch_magnification(self, custom_mpp_keys: Optional[List[str]] = None) -> int:
        """Estimate magnification from MPP or OpenSlide metadata."""
        # Try from MPP first
        mpp_x = self.mpp
        if mpp_x is None:
            try:
                mpp_x = self._fetch_mpp(custom_mpp_keys)
            except ValueError:
                mpp_x = None

        if mpp_x is not None:
            if mpp_x < 0.16:
                return 80
            elif mpp_x < 0.2:
                return 60
            elif mpp_x < 0.3:
                return 40
            elif mpp_x < 0.6:
                return 20
            elif mpp_x < 1.2:
                return 10
            elif mpp_x < 2.4:
                return 5
            else:
                raise ValueError(
                    f"MPP value {mpp_x} is unexpectedly large for a WSI. "
                    "Check your slide metadata."
                )

        # Fallback: OpenSlide objective-power property
        meta_mag = self.img.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
        if meta_mag is not None:
            try:
                return int(meta_mag)
            except ValueError:
                pass

        raise ValueError(
            f"Unable to determine magnification for: '{self.slide_path}'"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self._initialized:
            return (
                f"<WSI name={self.name}, width={self.width}, height={self.height}, "
                f"mpp={self.mpp}, mag={self.mag}>"
            )
        return f"<WSI name={self.name} (not yet initialized)>"

    def __enter__(self) -> "WSI":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.release()
        return False

    def get_dimensions(self) -> Tuple[int, int]:
        """Return (width, height) of the slide at level 0."""
        self._lazy_initialize()
        return self.img.dimensions

    def get_thumbnail(self, size: Tuple[int, int]) -> Image.Image:
        """Return an RGB PIL thumbnail of the requested size."""
        self._lazy_initialize()
        return self.img.get_thumbnail(size).convert('RGB')

    def read_region(
        self,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
        read_as: ReadMode = 'pil',
    ) -> Union[Image.Image, np.ndarray]:
        """
        Read a rectangular region from the slide.

        Parameters
        ----------
        location : (x, y)
            Top-left corner in level-0 coordinates.
        level : int
            Pyramid level.
        size : (width, height)
            Region size in pixels at the requested level.
        read_as : {'pil', 'numpy'}
            Output format.

        Returns
        -------
        PIL.Image.Image or np.ndarray (H, W, 3)
        """
        self._lazy_initialize()
        try:
            region = self.img.read_region(location, level, size).convert('RGB')
        except openslide.lowlevel.OpenSlideError as e:
            warnings.warn(
                f"Corrupt region at {location}, level {level}: {e}. "
                "Re-initializing OpenSlide and attempting fallback."
            )
            self.img = openslide.OpenSlide(self.slide_path)
            try:
                if level > 0:
                    downsample = self.level_downsamples[level]
                    fallback_size = (
                        max(1, round(size[0] * downsample)),
                        max(1, round(size[1] * downsample)),
                    )
                    region = self.img.read_region(location, 0, fallback_size).convert('RGB')
                    resample = (
                        Image.Resampling.LANCZOS
                        if hasattr(Image, 'Resampling') else Image.LANCZOS
                    )
                    region = region.resize(size, resample=resample)
                else:
                    region = Image.new('RGB', size, (255, 255, 255))
            except Exception as fallback_e:
                warnings.warn(
                    f"Fallback read failed at {location}, level {level}: {fallback_e}. "
                    "Returning blank white image."
                )
                region = Image.new('RGB', size, (255, 255, 255))

        if read_as == 'pil':
            return region
        elif read_as == 'numpy':
            return np.array(region)
        else:
            raise ValueError(f"Invalid read_as='{read_as}'. Must be 'pil' or 'numpy'.")

    def get_best_level_and_custom_downsample(
        self,
        downsample: float,
        tolerance: float = 0.01,
    ) -> Tuple[int, float]:
        """Return the best pyramid level and residual custom downsample factor."""
        self._lazy_initialize()
        level_downsamples = self.level_downsamples

        for level, ld in enumerate(level_downsamples):
            if abs(ld - downsample) <= tolerance:
                return level, 1.0

        if downsample >= level_downsamples[0]:
            closest_level, closest_ld = None, None
            for level, ld in enumerate(level_downsamples):
                if ld <= downsample:
                    closest_level, closest_ld = level, ld
                else:
                    break
            if closest_level is not None:
                return closest_level, downsample / closest_ld

        for level, ld in enumerate(level_downsamples):
            if ld >= downsample:
                return level, ld / downsample

        raise ValueError(f"No suitable pyramid level found for downsample={downsample}.")

    # ------------------------------------------------------------------
    # Patcher
    # ------------------------------------------------------------------

    def create_patcher(
        self,
        patch_size: int,
        src_pixel_size: Optional[float] = None,
        dst_pixel_size: Optional[float] = None,
        src_mag: Optional[int] = None,
        dst_mag: Optional[int] = None,
        overlap: int = 0,
        mask: Optional[gpd.GeoDataFrame] = None,
        coords_only: bool = False,
        custom_coords: Optional[np.ndarray] = None,
        threshold: float = 0.15,
        pil: bool = False,
    ) -> WSIPatcher:
        """Create a WSIPatcher for iterating over patches of this slide."""
        return WSIPatcher(
            self, patch_size, src_pixel_size, dst_pixel_size, src_mag, dst_mag,
            overlap, mask, coords_only, custom_coords, threshold, pil,
        )

    # ------------------------------------------------------------------
    # Coordinate extraction
    # ------------------------------------------------------------------

    def extract_tissue_coords(
        self,
        target_mag: int,
        patch_size: int,
        save_coords: str,
        overlap: int = 0,
        min_tissue_proportion: float = 0.,
    ) -> pd.DataFrame:
        """
        Extract patch coordinates from tissue regions and save them as CSV.

        Parameters
        ----------
        target_mag : int
            Target magnification for the patches.
        patch_size : int
            Patch size in pixels at target_mag.
        save_coords : str
            Directory where the CSV will be saved as ``{name}_patches.csv``.
        overlap : int
            Overlap between adjacent patches in pixels.
        min_tissue_proportion : float
            Minimum fraction of the patch that must be tissue (0 = keep all).

        Returns
        -------
        pd.DataFrame
            Columns: 'x', 'y' (level-0 coordinates).
        """
        self._lazy_initialize()

        patcher = self.create_patcher(
            patch_size=patch_size,
            src_mag=self.mag,
            dst_mag=target_mag,
            mask=self.gdf_contours,
            coords_only=True,
            overlap=overlap,
            threshold=min_tissue_proportion,
        )

        coords = [(x, y) for x, y in patcher]
        df = pd.DataFrame(coords, columns=['x', 'y'])

        os.makedirs(save_coords, exist_ok=True)
        csv_path = os.path.join(save_coords, f'{self.name}_patches.csv')
        df.to_csv(csv_path, index=False)

        return df

    # ------------------------------------------------------------------
    # Segmentation helpers (kept for compatibility)
    # ------------------------------------------------------------------

    def _segment_semantic(
        self,
        segmentation_model: SegmentationModel,
        target_mag: int,
        verbose: bool,
        device: str,
        batch_size: int,
        collate_fn,
        num_workers: Optional[int],
        inference_fn,
    ):
        destination_mpp = 10 / target_mag
        patcher = self.create_patcher(
            patch_size=segmentation_model.input_size,
            src_pixel_size=self.mpp,
            dst_pixel_size=destination_mpp,
            mask=self.gdf_contours,
        )
        precision = segmentation_model.precision
        eval_transforms = segmentation_model.eval_transforms
        dataset = WSIPatcherDataset(patcher, eval_transforms)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=(
                get_num_workers(batch_size, max_workers=self.max_workers)
                if num_workers is None else num_workers
            ),
            pin_memory=False,
        )

        mpp_reduction_factor = self.mpp / destination_mpp
        width, height = self.get_dimensions()
        width = int(round(width * mpp_reduction_factor))
        height = int(round(height * mpp_reduction_factor))
        predicted_mask = np.zeros((height, width), dtype=np.uint8)

        if verbose:
            dataloader = tqdm(dataloader)

        for batch in dataloader:
            with torch.autocast(
                device_type=device.split(':')[0],
                dtype=precision,
                enabled=(precision != torch.float32),
            ):
                if collate_fn is not None:
                    if 'xcoords' not in batch or 'ycoords' not in batch:
                        raise ValueError(
                            "collate_fn must return 'xcoords' and 'ycoords'."
                        )
                    xcoords = torch.tensor(batch['xcoords'])
                    ycoords = torch.tensor(batch['ycoords'])
                    imgs = batch.get('img')
                else:
                    imgs, (xcoords, ycoords) = batch

                if inference_fn is not None:
                    preds = inference_fn(segmentation_model, batch, device).cpu().numpy()
                else:
                    imgs = imgs.to(device, dtype=precision)
                    preds = segmentation_model(imgs).cpu().numpy()

            x_starts = np.clip(
                np.round(xcoords.numpy() * mpp_reduction_factor).astype(int), 0, width - 1
            )
            y_starts = np.clip(
                np.round(ycoords.numpy() * mpp_reduction_factor).astype(int), 0, height - 1
            )
            x_ends = np.clip(x_starts + segmentation_model.input_size, 0, width)
            y_ends = np.clip(y_starts + segmentation_model.input_size, 0, height)

            for i in range(len(preds)):
                xs, xe = x_starts[i], x_ends[i]
                ys, ye = y_starts[i], y_ends[i]
                if xs >= xe or ys >= ye:
                    continue
                predicted_mask[ys:ye, xs:xe] += preds[i][:ye - ys, :xe - xs]

        return predicted_mask, mpp_reduction_factor

    @torch.inference_mode()
    def segment_tissue(
        self,
        segmentation_model: SegmentationModel,
        target_mag: int = 10,
        holes_are_tissue: bool = True,
        job_dir: Optional[str] = None,
        batch_size: int = 16,
        device: str = 'cuda:0',
        verbose: bool = False,
        num_workers: Optional[int] = None,
    ) -> Union[str, gpd.GeoDataFrame]:
        self._lazy_initialize()
        segmentation_model.to(device)

        max_dim = 1000
        if self.width > self.height:
            tw, th = max_dim, int(max_dim * self.height / self.width)
        else:
            th, tw = max_dim, int(max_dim * self.width / self.height)
        thumbnail = self.get_thumbnail((tw, th))

        predicted_mask, mpp_reduction_factor = self._segment_semantic(
            segmentation_model, target_mag, verbose, device, batch_size,
            None, num_workers, None,
        )
        predicted_mask = (predicted_mask > 0).astype(np.uint8) * 255

        gdf_contours = mask_to_gdf(
            mask=predicted_mask,
            max_nb_holes=0 if holes_are_tissue else 20,
            min_contour_area=1000,
            pixel_size=self.mpp,
            contour_scale=1 / mpp_reduction_factor,
        )

        if job_dir is not None:
            thumbnail_path = os.path.join(job_dir, 'thumbnails', f'{self.name}.jpg')
            os.makedirs(os.path.dirname(thumbnail_path), exist_ok=True)
            thumbnail.save(thumbnail_path)

            gdf_path = os.path.join(job_dir, 'contours_geojson', f'{self.name}.geojson')
            os.makedirs(os.path.dirname(gdf_path), exist_ok=True)
            gdf_contours.set_crs('EPSG:3857', inplace=True)
            gdf_contours.to_file(gdf_path, driver='GeoJSON')
            self.gdf_contours = gdf_contours
            self.tissue_seg_path = gdf_path

            contours_path = os.path.join(job_dir, 'contours', f'{self.name}.jpg')
            annotated = np.array(thumbnail)
            overlay_gdf_on_thumbnail(
                gdf_contours, annotated, contours_path, tw / self.width
            )
            return gdf_path

        return gdf_contours

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def extract_patch_features(
        self,
        patch_encoder: torch.nn.Module,
        coords_path: str,
        save_features: str,
        patch_size: int,
        target_mag: int,
        device: str = 'cuda:0',
        batch_limit: int = 512,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Extract patch-level feature embeddings.

        Parameters
        ----------
        patch_encoder : torch.nn.Module
        coords_path : str
            Path to the CSV produced by extract_tissue_coords (columns: x, y).
        save_features : str
            Directory where the feature CSV will be written.
        patch_size : int
        target_mag : int
        device : str
        batch_limit : int
        verbose : bool

        Returns
        -------
        np.ndarray of shape (N, D)
        """
        self._lazy_initialize()
        patch_encoder.to(device)
        patch_encoder.eval()
        precision = getattr(patch_encoder, 'precision', torch.float32)
        patch_transforms = patch_encoder.eval_transforms

        coords_df = pd.read_csv(coords_path)
        coords = coords_df[['x', 'y']].values  # (N, 2)

        patcher = self.create_patcher(
            patch_size=patch_size,
            src_mag=self.mag,
            dst_mag=target_mag,
            custom_coords=coords,
            coords_only=False,
            pil=True,
        )

        dataset = WSIPatcherDataset(patcher, patch_transforms)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_limit,
            num_workers=get_num_workers(batch_limit, max_workers=self.max_workers),
            pin_memory=False,
        )
        if verbose:
            dataloader = tqdm(dataloader)

        features = []
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            with torch.autocast(
                device_type='cuda', dtype=precision,
                enabled=(precision != torch.float32),
            ):
                batch_features = patch_encoder(imgs)
            features.append(batch_features.cpu().numpy())

        return np.concatenate(features, axis=0)  # (N, D)

    @torch.inference_mode()
    def extract_slide_features(
        self,
        coords: np.ndarray,
        features: np.ndarray,
        slide_encoder: torch.nn.Module,
        save_features: str,
        device: str = 'cuda',
    ) -> np.ndarray:
        """
        Encode patch features into a single slide-level feature vector.

        Parameters
        ----------
        coords : np.ndarray  (N, 2)
        features : np.ndarray  (N, D)
        slide_encoder : torch.nn.Module
        save_features : str
        device : str

        Returns
        -------
        np.ndarray of shape (K,)
        """
        slide_encoder.to(device)
        slide_encoder.eval()

        patch_features = torch.from_numpy(features).float().to(device).unsqueeze(0)
        coords_tensor = torch.from_numpy(coords).to(device).unsqueeze(0)

        batch = {'features': patch_features, 'coords': coords_tensor}

        with torch.autocast(
            device_type='cuda',
            enabled=(slide_encoder.precision != torch.float32),
        ):
            slide_features = slide_encoder(batch, device)

        return slide_features.float().cpu().numpy().squeeze()  # (K,)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def release(self) -> None:
        """Close the OpenSlide handle and free memory."""
        if self.img is not None:
            try:
                self.img.close()
            except Exception:
                pass
            self.img = None

        self.gdf_contours = None
        self._initialized = False
