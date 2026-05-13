import json
import numpy as np
import cv2


GEOJSON_PATH = "data/prism/CD10/geojson_contours/13906.geojson"
OUTPUT_PATH  = "output_mask.png"
MARGIN       = 20


def geojson_to_png(geojson_path: str, output_path: str, margin: int = 20):
    with open(geojson_path) as f:
        data = json.load(f)

    features = data.get("features", [])
    if not features:
        print(f"[SKIP] aucun contour dans {geojson_path}")
        return

    all_coords = []
    for feat in features:
        for ring in feat["geometry"]["coordinates"]:
            all_coords.extend(ring)
    arr = np.array(all_coords)
    x_min, y_min = arr[:, 0].min(), arr[:, 1].min()
    x_max, y_max = arr[:, 0].max(), arr[:, 1].max()

    width  = int(x_max - x_min) + 2 * margin + 1
    height = int(y_max - y_min) + 2 * margin + 1

    canvas = np.full((height, width), 255, dtype=np.uint8)  # fond blanc

    for feat in features:
        for ring in feat["geometry"]["coordinates"]:
            pts = np.array(ring, dtype=np.float32)
            pts[:, 0] -= x_min - margin
            pts[:, 1] -= y_min - margin
            pts = pts.astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(canvas, [pts], color=0)  # ROI noir

    cv2.imwrite(output_path, canvas)
    print(f"Sauvegardé → {output_path}  ({width}x{height} px)")


if __name__ == "__main__":
    geojson_to_png(GEOJSON_PATH, OUTPUT_PATH, MARGIN)
