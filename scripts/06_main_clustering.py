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
MAX_CLUSTERS = 50
COMPONENTS = ["Component 1", "Component 2", "Component 3"]

if __name__=="__main__":
    # 1 i 2. Codificació i escalat
    preprocessed_df = preprocess(PATH_DATASET, TARGET) # Carreguem les dades i les preprocessem
    preprocessed_df = preprocessed_df.drop(TARGET, axis=1)  # Eliminem la variable a partir de la qual volem fer clústering

    # 3. Si hi ha variables rellevants, reduïm la dimensió del dataset
    if VARIABLES_RELLEVANTS: 
        preprocessed_df = preprocessed_df[VARIABLES_RELLEVANTS]  # Modifiquem el dataset

    # 4. Elecció de l'algoritme de clústering: Inicialitzem la classe i provem
    for algoritme in ALGORITHMS:
        print(f'\nModel {algoritme}')
        model = ClusteringModel(preprocessed_df, algorithm=algoritme)
        model.elbow_method(max_clusters=MAX_CLUSTERS)
        model.fit()

        # Visualitzacions:
        # clustering_kmeans.plot_clusters_PCA_2d()
        # clustering_kmeans.plot_clusters_PCA_3d()
        # clustering_kmeans.plot_clusters_TSNE_2d()
        reduced_data = clustering_kmeans.plot_clusters_TSNE_3d() 

        model.evaluate()
