import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.cluster import KMeans
import pandas as pd
import os

marker_list  = ["BCL2", "BCL6", "CD10", "HE", "MUM1", "MYC"]
encoder_list = ["prism", "feather"]
ENCODER_CFG = {
    "prism":   dict(slide_subdir="slide_features_prism",   slide_csv="prism_encoder.csv"),
    "titan":   dict(slide_subdir="slide_features_titan",   slide_csv="titan_encoder.csv"),
    "feather": dict(slide_subdir="slide_features_feather", slide_csv="feather_encoder.csv"),
}
# Une liste vide pour enregistrer les inerties :  
intertia_list = [ ]

for encoder in encoder_list:
    enc = ENCODER_CFG[encoder]
    for marker in marker_list:
        candidate = os.path.join("data_224", encoder, marker, enc["slide_subdir"], enc["slide_csv"])
        df_marker = pd.read_csv(candidate)

        X = df_marker.values
        # On peut le transformer en DataFrame : 
        X = pd.DataFrame(X)

        # Cela permet d'appliquer la méthode .head : 
        print(X.head())

        """ # Notre liste de nombres de clusters : 
        k_list = range(1, 15)

        # Pour chaque nombre de clusters : 
        for k in k_list : 
            
            # On instancie un k-means pour k clusters
            kmeans = KMeans(n_clusters=k)
            
            # On entraine
            kmeans.fit(X)
            
            # On enregistre l'inertie obtenue : 
            intertia_list.append(kmeans.inertia_)

        fig, ax = plt.subplots(1,1,figsize=(12,6))

        ax.set_ylabel("intertia")
        ax.set_xlabel("n_cluster")

        ax = plt.plot(k_list, intertia_list) """