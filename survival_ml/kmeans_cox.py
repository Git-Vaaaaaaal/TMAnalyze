import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import pandas as pd
import os

marker_list  = ["BCL2", "BCL6", "CD10", "HE", "MUM1", "MYC"]
encoder_list = ["prism", "feather"]
ENCODER_CFG = {
    "prism":   dict(slide_subdir="slide_features_prism",   slide_csv="prism_encoder.csv"),
    "titan":   dict(slide_subdir="slide_features_titan",   slide_csv="titan_encoder.csv"),
    "feather": dict(slide_subdir="slide_features_feather", slide_csv="feather_encoder.csv"),
}

os.makedirs("kmeans", exist_ok=True)
for encoder in encoder_list:
    enc = ENCODER_CFG[encoder]
    for marker in marker_list:
        candidate = os.path.join("data_224", encoder, marker, enc["slide_subdir"], enc["slide_csv"])
        df_marker = pd.read_csv(candidate)
        df_marker = df_marker.drop(columns=["wsi_name"])
        X = df_marker.values
        # On peut le transformer en DataFrame : 
        X = pd.DataFrame(X)

        intertia_list = [ ]
        # Notre liste de nombres de clusters : 
        k_list = range(1, 15)

        # Pour chaque nombre de clusters : 
        for k in k_list : 
            
            # On instancie un k-means pour k clusters
            kmeans = KMeans(n_clusters=k)
            
            # On entraine
            kmeans.fit(X)
            
            # On enregistre l'inertie obtenue : 
            intertia_list.append(kmeans.inertia_)

        # Méthode du coude : on cherche le k où la 2ème dérivée est maximale
        inertia_arr = np.array(intertia_list)
        deltas = np.diff(inertia_arr)
        second_deriv = np.diff(deltas)
        elbow_k = int(k_list[np.argmax(second_deriv) + 2])
        print(f"[{encoder}][{marker}] Coude détecté à k={elbow_k}")

        # Silhouette score sur le k du coude
        kmeans_final = KMeans(n_clusters=elbow_k, random_state=42)
        labels = kmeans_final.fit_predict(X)
        sil_avg = silhouette_score(X, labels)
        sil_samples = silhouette_samples(X, labels)
        print(f"[{encoder}][{marker}] Silhouette moyen (k={elbow_k}) : {sil_avg:.4f}")
        for c in range(elbow_k):
            c_sil = sil_samples[labels == c].mean()
            print(f"  Cluster {c} ({(labels == c).sum()} items) : silhouette = {c_sil:.4f}")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Courbe d'inertie
        axes[0].set_ylabel("inertia")
        axes[0].set_xlabel("n_cluster")
        axes[0].plot(k_list, intertia_list)
        axes[0].axvline(x=elbow_k, color="red", linestyle="--", label=f"coude k={elbow_k}")
        axes[0].legend()
        axes[0].set_title("Méthode du coude")

        # Silhouette par cluster
        y_lower = 10
        for c in range(elbow_k):
            c_vals = np.sort(sil_samples[labels == c])
            y_upper = y_lower + len(c_vals)
            axes[1].barh(range(y_lower, y_upper), c_vals, height=1.0)
            axes[1].text(-0.05, (y_lower + y_upper) / 2, str(c))
            y_lower = y_upper + 10
        axes[1].axvline(x=sil_avg, color="red", linestyle="--", label=f"moy = {sil_avg:.3f}")
        axes[1].set_xlabel("Silhouette")
        axes[1].set_ylabel("Cluster")
        axes[1].set_title(f"Silhouette (k={elbow_k})")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join("kmeans", f"{marker}_{encoder}.png"))