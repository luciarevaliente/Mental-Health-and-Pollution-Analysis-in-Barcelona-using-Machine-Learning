"""
Antes:
    1. Eliminar valores null, escalar los datos y seleccionar las variables.
    2. Determinar número óptimo de clusters
    3. Aplicar algoritmo de clústering
"""
# IMPORTACIÓ
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from yellowbrick.cluster import KElbowVisualizer
from mpl_toolkits.mplot3d import Axes3D
from preprocess import preprocess
from models_clustering import ClusteringModel

# VARIABLES CONSTANTS
PATH_DATASET = "data/cleaned_dataset.pkl"  # Dataset natejat
ALGORITHMS = ['kmeans', 'spectral', 'agglo', 'gmm']  # Algoritmes de clústering a testejar
TARGET = 'estres'
VARIABLES_RELLEVANTS = []

if __name__=="__main__":
    # 1 i 2. Codificació i escalat
    preprocessed_df = preprocess(PATH_DATASET, TARGET) # Carreguem les dades i les preprocessem
    preprocessed_df = preprocessed_df.drop(TARGET, axis=1)  # Eliminem la variable a partir de la qual volem fer clústering

    # 3. Si hi ha variables rellevants, reduïm la dimensió del dataset
    if VARIABLES_RELLEVANTS: 
        preprocessed_df = preprocessed_df[VARIABLES_RELLEVANTS]  # Modifiquem el dataset

    # 4. Elecció de l'algoritme de clústering: Inicialitzem la classe i provem
    for algoritme in ALGORITHMS:
        if algoritme == 'kmeans':
            clustering_kmeans = ClusteringModel(preprocessed_df, algorithm='kmeans')  
            clustering_kmeans.elbow_method(max_clusters=50)
            clustering_kmeans.fit()
            clustering_kmeans.plot_clusters_PCA_2d()
            clustering_kmeans.plot_clusters_PCA_3d()
            clustering_kmeans.plot_clusters_TSNE_2d()
            clustering_kmeans.plot_clusters_TSNE_3d()
            clustering_kmeans.evaluate()

            # reduced_dataset['cluster'] = clustering_kmeans.get_labels() 
            # cluster_stats = reduced_dataset.groupby('cluster').mean()
            # # cluster_stats[:,0]
            # for col in cluster_stats:
            #     print(cluster_stats[col])

        elif algoritme == 'spectral':
            print('en desenvolupament')
            # clustering_spectral = ClusteringModel(df, n_clusters=3, algorithm='spectral')
            # clustering_spectral.fit()
            # clustering_spectral.evaluate()

        elif algoritme == 'agglo':
            print('en desenvolupament')
            # clustering_agglo = ClusteringModel(df, n_clusters=3, algorithm='agglo')
            # clustering_agglo.fit()
            # clustering_agglo.evaluate()

        elif algoritme == 'gmm':
            print('en desenvolupament')
            # clustering_gmm = ClusteringModel(df, n_clusters=3, algorithm='gmm')
            # clustering_gmm.fit()
            # clustering_gmm.evaluate()

        break