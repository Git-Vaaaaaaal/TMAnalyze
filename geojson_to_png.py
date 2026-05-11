import os
import json
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np
import cv2


def geojson_to_mask(geojson_path: str, margin: int = 20) -> np.ndarray:
    with open(geojson_path) as f:
        data = json.load(f)

    features = data.get("features", [])
    if not features:
        return None

    all_coords = []
    for feat in features:
        for ring in feat["geometry"]["coordinates"]:
            all_coords.extend(ring)
    arr = np.array(all_coords)
    x_min, y_min = arr[:, 0].min(), arr[:, 1].min()
    x_max, y_max = arr[:, 0].max(), arr[:, 1].max()

    width  = int(x_max - x_min) + 2 * margin + 1
    height = int(y_max - y_min) + 2 * margin + 1

    canvas = np.zeros((height, width), dtype=np.uint8)
    for feat in features:
        for ring in feat["geometry"]["coordinates"]:
            pts = np.array(ring, dtype=np.float32)
            pts[:, 0] -= x_min - margin
            pts[:, 1] -= y_min - margin
            pts = pts.astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(canvas, [pts], color=255)

    return canvas


class GeoJSONViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GeoJSON Mask Viewer")
        self.geometry("900x700")
        self.configure(bg="#1e1e1e")

        self._build_toolbar()
        self._build_canvas()
        self._build_statusbar()

        self.current_files = []
        self.current_index = 0
        self._photo = None

    def _build_toolbar(self):
        bar = tk.Frame(self, bg="#2d2d2d", pady=6)
        bar.pack(fill=tk.X)

        btn_style = dict(bg="#3c8dbc", fg="white", relief=tk.FLAT,
                         padx=12, pady=4, cursor="hand2", font=("Helvetica", 10))

        tk.Button(bar, text="Charger fichier", command=self._load_file, **btn_style).pack(side=tk.LEFT, padx=6)
        tk.Button(bar, text="Charger dossier", command=self._load_dir,  **btn_style).pack(side=tk.LEFT, padx=2)

        nav_style = dict(bg="#555", fg="white", relief=tk.FLAT,
                         padx=10, pady=4, cursor="hand2", font=("Helvetica", 10))
        tk.Button(bar, text="◀", command=self._prev, **nav_style).pack(side=tk.LEFT, padx=(16, 2))
        tk.Button(bar, text="▶", command=self._next, **nav_style).pack(side=tk.LEFT, padx=2)

        self.lbl_counter = tk.Label(bar, text="", bg="#2d2d2d", fg="#aaa", font=("Helvetica", 10))
        self.lbl_counter.pack(side=tk.LEFT, padx=10)

        save_style = dict(bg="#27ae60", fg="white", relief=tk.FLAT,
                          padx=12, pady=4, cursor="hand2", font=("Helvetica", 10))
        tk.Button(bar, text="Enregistrer PNG", command=self._save_png, **save_style).pack(side=tk.RIGHT, padx=6)

    def _build_canvas(self):
        frame = tk.Frame(self, bg="#1e1e1e")
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(frame, bg="#111", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_resize)

    def _build_statusbar(self):
        self.status = tk.Label(self, text="Aucun fichier chargé", bg="#2d2d2d",
                               fg="#aaa", anchor=tk.W, padx=8, font=("Helvetica", 9))
        self.status.pack(fill=tk.X, side=tk.BOTTOM)

    # --- file loading ---

    def _load_file(self):
        path = filedialog.askopenfilename(
            title="Choisir un fichier GeoJSON",
            filetypes=[("GeoJSON", "*.geojson"), ("JSON", "*.json"), ("Tous", "*.*")]
        )
        if path:
            self.current_files = [path]
            self.current_index = 0
            self._show_current()

    def _load_dir(self):
        folder = filedialog.askdirectory(title="Choisir un dossier")
        if not folder:
            return
        files = []
        for dirpath, _, fnames in os.walk(folder):
            for f in fnames:
                if f.endswith((".geojson", ".json")):
                    files.append(os.path.join(dirpath, f))
        if not files:
            self.status.config(text=f"Aucun fichier GeoJSON trouvé dans {folder}")
            return
        self.current_files = sorted(files)
        self.current_index = 0
        self._show_current()

    # --- navigation ---

    def _prev(self):
        if self.current_files:
            self.current_index = (self.current_index - 1) % len(self.current_files)
            self._show_current()

    def _next(self):
        if self.current_files:
            self.current_index = (self.current_index + 1) % len(self.current_files)
            self._show_current()

    # --- display ---

    def _show_current(self):
        if not self.current_files:
            return
        path = self.current_files[self.current_index]
        mask = geojson_to_mask(path)
        if mask is None:
            self.status.config(text=f"[VIDE] {os.path.basename(path)}")
            self.canvas.delete("all")
            self._current_mask = None
            return

        self._current_mask = mask
        self._render_mask()

        total = len(self.current_files)
        idx   = self.current_index + 1
        self.lbl_counter.config(text=f"{idx} / {total}")
        self.status.config(text=path)

    def _render_mask(self):
        if self._current_mask is None:
            return
        cw = self.canvas.winfo_width()  or 860
        ch = self.canvas.winfo_height() or 600

        mask = self._current_mask
        mh, mw = mask.shape
        scale = min(cw / mw, ch / mh)
        new_w, new_h = int(mw * scale), int(mh * scale)

        resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        img = Image.fromarray(resized)
        self._photo = ImageTk.PhotoImage(img)

        self.canvas.delete("all")
        x0 = (cw - new_w) // 2
        y0 = (ch - new_h) // 2
        self.canvas.create_image(x0, y0, anchor=tk.NW, image=self._photo)

    def _on_resize(self, event):
        self._render_mask()

    # --- save ---

    def _save_png(self):
        if self._current_mask is None:
            return
        default = os.path.splitext(os.path.basename(self.current_files[self.current_index]))[0] + ".png"
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=default,
            filetypes=[("PNG", "*.png")]
        )
        if path:
            cv2.imwrite(path, self._current_mask)
            self.status.config(text=f"Enregistré → {path}")


if __name__ == "__main__":
    app = GeoJSONViewer()
    app.mainloop()
