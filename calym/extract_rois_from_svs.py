"""
Extrait, pour chaque paire (svs, geojson) de meme nom, une image .tiff par
region d'interet (feature) presente dans le geojson.

Hypotheses sur les fichiers :
  - un fichier "<id_lame>.svs" et un fichier "<id_lame>.geojson" partagent le
    meme nom de base (ex: "13901.svs" / "13901.geojson") ;
  - le geojson est une FeatureCollection dont chaque Feature delimite UNE
    region d'interet (ex: un noyau de TMA) via un polygone en coordonnees
    pixel niveau 0 de la lame (meme convention que les geojson de contours
    tissulaires deja produits par src/class_embedding.py) ;
  - chaque Feature porte un identifiant (utilise comme nom de fichier .tiff)
    sous l'une de ces cles, testees dans l'ordre : "id" (au niveau Feature,
    standard GeoJSON), puis dans "properties" : "id", "name",
    "classification.name", "tissue_id".

La magnification/mpp de la lame source est lue une seule fois (meme logique
que _fetch_mpp/_fetch_magnification dans src/class_wsi_claude.py, qui gere
deja le contournement du bug de metadonnees TIFF documente dans
SITUATION_MPP.txt) puis reecrite explicitement dans les tags de resolution
TIFF (XResolution/YResolution/ResolutionUnit) de chaque crop, pour que ce
parametre ne se perde pas au decoupage.

Ce script n'utilise que openslide (pas de dependance a torch/trident), pour
pouvoir tourner meme sur une machine ou l'installation PyTorch/CUDA est
cassee.

Usage : renseigner les variables ci-dessous puis lancer
    python calym/extract_rois_from_svs.py
"""

import json
import os

import numpy as np
import openslide
import tifffile
from PIL import Image

# ======================================================================
# Parametres a renseigner
# ======================================================================

SVS_DIR      = r"gold"       # dossier contenant les .svs
GEOJSON_DIR  = r"gold"   # dossier contenant les .geojson
OUTPUT_DIR   = r"data\TMAnalyze8"    # dossier ou stocker les .tiff extraits

MARGIN       = 0          # marge en pixels autour de chaque bbox
TILE_SIZE    = 256        # taille des tuiles du TIFF de sortie
COMPRESSION  = "deflate"  # "deflate" (sans perte), "jpeg" ou "none"
JPEG_QUALITY = 90         # utilise seulement si COMPRESSION == "jpeg"

# ======================================================================

# Autorise les tres grands crops (lames medicales) sans avertissement PIL.
Image.MAX_IMAGE_PIXELS = None


def fetch_mpp(slide: openslide.OpenSlide, custom_mpp_keys: list[str] | None = None) -> float:
    """Extrait le mpp (microns/pixel) des metadonnees OpenSlide.

    Meme logique que WSI._fetch_mpp dans src/class_wsi_claude.py (dont le
    fallback tiff.XResolution/ResolutionUnit contourne le bug documente
    dans SITUATION_MPP.txt), reimplementee ici sans dependre de torch.
    """
    mpp_keys = [
        openslide.PROPERTY_NAME_MPP_X,
        "openslide.mirax.MPP",
        "aperio.MPP",
        "hamamatsu.XResolution",
        "openslide.comment",
    ]
    if custom_mpp_keys:
        mpp_keys.extend(custom_mpp_keys)

    for key in mpp_keys:
        if key in slide.properties:
            try:
                return round(float(slide.properties[key]), 4)
            except ValueError:
                continue

    x_res = slide.properties.get("tiff.XResolution")
    unit = slide.properties.get("tiff.ResolutionUnit")
    if x_res and unit:
        try:
            if unit.lower() == "centimeter":
                return round(10_000 / float(x_res), 4)
            elif unit.upper() == "INCH":
                return round(25_400 / float(x_res), 4)
        except ValueError:
            pass

    raise ValueError(
        "Impossible d'extraire le mpp des metadonnees de la lame. "
        "Fournir custom_mpp_keys ou corriger les tags de la lame."
    )


def fetch_magnification(slide: openslide.OpenSlide, mpp: float | None) -> int:
    """Estime la magnification a partir du mpp (ou, a defaut, des metadonnees OpenSlide)."""
    if mpp is not None:
        if mpp < 0.16:
            return 80
        elif mpp < 0.2:
            return 60
        elif mpp < 0.3:
            return 40
        elif mpp < 0.6:
            return 20
        elif mpp < 1.2:
            return 10
        elif mpp < 2.4:
            return 5
        else:
            raise ValueError(f"Valeur de mpp inattendue (trop grande) : {mpp}")

    meta_mag = slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
    if meta_mag is not None:
        try:
            return int(meta_mag)
        except ValueError:
            pass

    raise ValueError("Impossible de determiner la magnification de la lame.")


# Ordre de priorite pour trouver l'identifiant d'une region. Les exports
# QuPath (annotations TMA) mettent le nom lisible dans properties.name
# (ex: "13/704") ; le "id" au niveau Feature est un UUID auto-genere, pas
# pense pour servir de nom de fichier, donc il passe en tout dernier recours.
ID_PROPERTY_CANDIDATES = ["name", "classification.name", "tissue_id", "id"]

_UNSAFE_FILENAME_CHARS = '/\\:*?"<>|'


def sanitize_filename(value: str) -> str:
    """Remplace les caracteres interdits/ambigus dans un nom de fichier (ex: '13/704' -> '13-704')."""
    for ch in _UNSAFE_FILENAME_CHARS:
        value = value.replace(ch, "-")
    return value.strip()


def resolve_feature_id(feature: dict, fallback: str) -> str:
    """Cherche un identifiant pour ce Feature dans un ordre de cles connu."""
    props = feature.get("properties", {}) or {}
    for key in ID_PROPERTY_CANDIDATES:
        if key == "classification.name":
            value = (props.get("classification") or {}).get("name")
        else:
            value = props.get(key)
        if value not in (None, ""):
            return sanitize_filename(str(value))

    if "id" in feature and feature["id"] not in (None, ""):
        return sanitize_filename(str(feature["id"]))

    return fallback


def feature_bbox(feature: dict) -> tuple[float, float, float, float] | None:
    """Renvoie (xmin, ymin, xmax, ymax) en coordonnees pixel niveau 0."""
    geom = feature.get("geometry")
    if geom is None:
        return None

    coords = []

    def _collect(node):
        if not node:
            return
        if isinstance(node[0], (int, float)):
            coords.append(node)
        else:
            for child in node:
                _collect(child)

    _collect(geom.get("coordinates"))
    if not coords:
        return None

    arr = np.asarray(coords, dtype=np.float64)
    x_min, y_min = arr[:, 0].min(), arr[:, 1].min()
    x_max, y_max = arr[:, 0].max(), arr[:, 1].max()
    return float(x_min), float(y_min), float(x_max), float(y_max)


def find_matching_slide(svs_dir: str, stem: str, exts=(".svs", ".SVS")) -> str | None:
    for ext in exts:
        candidate = os.path.join(svs_dir, stem + ext)
        if os.path.isfile(candidate):
            return candidate
    return None


def save_region_as_tiff(
    region: np.ndarray,
    out_path: str,
    mpp: float,
    magnification: int,
    tile_size: int,
    compression: str,
    jpeg_quality: int,
) -> None:
    """Ecrit le crop en TIFF tuile, avec les tags de resolution derives du mpp reel."""
    px_per_cm = 10_000.0 / mpp  # 1 cm = 10 000 micrometres

    compression_kwargs = {}
    if compression == "jpeg":
        compression_kwargs = dict(compression="jpeg", compressionargs={"level": jpeg_quality})
    elif compression == "deflate":
        compression_kwargs = dict(compression="deflate")
    elif compression == "none":
        compression_kwargs = {}
    else:
        raise ValueError(f"Compression inconnue: {compression}")

    tifffile.imwrite(
        out_path,
        region,
        photometric="rgb",
        tile=(tile_size, tile_size),
        resolution=(px_per_cm, px_per_cm),
        resolutionunit="CENTIMETER",
        description=f"mpp={mpp:.4f};magnification={magnification}x",
        **compression_kwargs,
    )


def process_pair(
    svs_path: str,
    geojson_path: str,
    output_dir: str,
    margin: int,
    tile_size: int,
    compression: str,
    jpeg_quality: int,
) -> int:
    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    features = data.get("features", [])
    if not features:
        print(f"[SKIP] {geojson_path} : aucune feature")
        return 0

    name = os.path.splitext(os.path.basename(svs_path))[0]
    slide = openslide.OpenSlide(svs_path)
    slide_w, slide_h = slide.dimensions
    mpp = fetch_mpp(slide)
    mag = fetch_magnification(slide, mpp)
    print(
        f"[{name}] {slide_w}x{slide_h}px  mpp={mpp}  "
        f"magnification={mag}x  ({len(features)} region(s) a extraire)"
    )

    n_saved = 0
    seen_ids: dict[str, int] = {}
    try:
        for idx, feature in enumerate(features):
            bbox = feature_bbox(feature)
            if bbox is None:
                print(f"  [SKIP] feature {idx} : geometrie manquante/invalide")
                continue

            roi_id = resolve_feature_id(feature, fallback=f"{name}_{idx}")

            # Meme id reutilise (ex: cores repliques d'un meme patient) -> on suffixe
            # pour ne pas ecraser le fichier precedent.
            seen_ids[roi_id] = seen_ids.get(roi_id, 0) + 1
            if seen_ids[roi_id] > 1:
                print(f"  [INFO] identifiant '{roi_id}' deja utilise dans ce geojson, "
                      f"suffixe _{seen_ids[roi_id]} ajoute")
                roi_id = f"{roi_id}_{seen_ids[roi_id]}"

            x_min, y_min, x_max, y_max = bbox
            x = max(0, int(x_min) - margin)
            y = max(0, int(y_min) - margin)
            x_end = min(slide_w, int(np.ceil(x_max)) + margin)
            y_end = min(slide_h, int(np.ceil(y_max)) + margin)
            w, h = x_end - x, y_end - y

            if w <= 0 or h <= 0:
                print(f"  [SKIP] '{roi_id}' : bbox degeneree ou hors limites")
                continue

            region = np.array(slide.read_region((x, y), 0, (w, h)).convert("RGB"))

            out_path = os.path.join(output_dir, f"{roi_id}.tiff")
            save_region_as_tiff(
                region, out_path, mpp, mag, tile_size, compression, jpeg_quality
            )
            print(f"  [OK] '{roi_id}' -> {out_path}  ({w}x{h}px)")
            n_saved += 1
    finally:
        slide.close()

    return n_saved


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    geojson_files = sorted(
        f for f in os.listdir(GEOJSON_DIR) if f.lower().endswith(".geojson")
    )
    if not geojson_files:
        print(f"Aucun .geojson trouve dans {GEOJSON_DIR}")
        return

    total_saved, total_pairs = 0, 0
    for geojson_name in geojson_files:
        stem = os.path.splitext(geojson_name)[0]
        svs_path = find_matching_slide(SVS_DIR, stem)
        if svs_path is None:
            print(f"[SKIP] pas de .svs correspondant a '{stem}' dans {SVS_DIR}")
            continue

        geojson_path = os.path.join(GEOJSON_DIR, geojson_name)
        try:
            total_saved += process_pair(
                svs_path,
                geojson_path,
                OUTPUT_DIR,
                MARGIN,
                TILE_SIZE,
                COMPRESSION,
                JPEG_QUALITY,
            )
            total_pairs += 1
        except Exception as e:
            print(f"[ERREUR] {stem} : {e}")

    print(f"\nTermine : {total_saved} region(s) extraite(s) depuis {total_pairs} lame(s).")


if __name__ == "__main__":
    main()
