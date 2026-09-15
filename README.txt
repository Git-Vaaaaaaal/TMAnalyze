kaplan.py = plot les courbes de survie selon kaplan meiers
run_embedding.py = run les embeddings dans le dataset d'image
mil_train.py = entrainements des modeles mil
ml_algorithms.py = entrainement sur algorithme de machine learning classique a partir des deonnées des slides encoders
table_population.py = code pour plot les différentes catégories pour les marqueurs (apercus des volumes)

Dossier : 
data_analysis = script python pour analyser les data cliniques des patients, les embeddings
src = dossier avec toutes les sous dépendances nécessaires a faire fonctionner les autres codes
survival_ml = dossier contenant les codes d execution de machine learning survival
    - rfs_gridsearch : rfs couplé a une fonction gridsearch 
    - rfs_kfold : rfs avec kfold (prend en entrée les outputs de rfs_gridsearch afin d'optimiser les parametres)
    - rfs_pca : pca puis random forest survival classique
    - gbs : gradient boosting survival 
    - svms : support vector machine survival
sh_slurm_script = dossier de tout les fichiers slurms pour jeanzay
    - cpu_works_jz : fichier de job sur cpu (pour ml)
    - embedding_jz : lance le fichier run_embedding.py sur jeanzay