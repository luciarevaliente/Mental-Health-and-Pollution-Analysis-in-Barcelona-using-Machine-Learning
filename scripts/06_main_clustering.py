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
COMPONENTS = ["Component 1", "Component 2", "Component 3"]
THRESHOLD = 0.6

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
            # clustering_kmeans = ClusteringModel(preprocessed_df, algorithm='kmeans')  
            # clustering_kmeans.elbow_method(max_clusters=50)
            # clustering_kmeans.fit()
            # clustering_kmeans.plot_clusters_PCA_2d()
            # clustering_kmeans.plot_clusters_PCA_3d()
            # clustering_kmeans.plot_clusters_TSNE_2d()
            clustering_kmeans = ClusteringModel(preprocessed_df, n_clusters=4,algorithm='kmeans')
            clustering_kmeans.fit()

            # Grups segons els centroides
            centroides, caracteristicas_relevantes = clustering_kmeans.analisi_components_centroides(preprocessed_df)

            # Ver los centroides
            print("Centroides de los clusters:")
            print(centroides)

            # Ver las características más relevantes por cluster
            print("Característiques segons centroides:")
            for cluster, data in caracteristicas_relevantes.items():
                print(f"Cluster {cluster}:")
                print(f"  Variables más altas: {data['top']}")
                print(f"  Variables más bajas: {data['low']}")
            print()

            # Visualitzem el cluster
            reduced_data = clustering_kmeans.plot_clusters_TSNE_3d()  # Obtenim les característiques que més afecten a cada component (correlació )--> top 5
            
            # Grups segons les correlacions de cada dimensió
            dic_correlacions = clustering_kmeans.analisi_components_tsne_correlacio(reduced_data)  

            # Ver las características más relevantes por cluster
            print("Característiques segons components:")
            for comp in dic_correlacions.keys():
                print(f'Component {comp}:') 
                print(f"  Variables más altas: {dic_correlacions[comp][0]}")
                print(f"  Variables más bajas: {dic_correlacions[comp][1]}")
            print()    

            # Evaluació final del model
            clustering_kmeans.evaluate()


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