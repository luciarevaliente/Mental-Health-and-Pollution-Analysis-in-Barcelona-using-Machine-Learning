"""
Antes:
    1. Eliminar valores null, escalar los datos y seleccionar las variables.
    2. Determinar número óptimo de clusters
    3. Aplicar algoritmo de clústering
"""
# IMPORTACIÓ
from preprocess import preprocess
from models_clustering import ClusteringModel

# VARIABLES CONSTANTS
PATH_DATASET = "data/cleaned_dataset.pkl"  # Dataset natejat
ALGORITHMS = ['kmeans', 'spectral', 'agglo', 'gmm']  # Algoritmes de clústering a testejar
TARGET = 'estres'
VARIABLES_RELLEVANTS = []

ALGORITHMS = ['kmeans', 'agglo', 'gmm']  # Algoritmes de clústering a testejar
MAX_CLUSTERS = 50
GMM_PARAM_GRID = {
    'n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Diferentes valores de k (componentes)
    'covariance_type': ['full', 'tied', 'diag', 'spherical'],  # Tipos de covarianza
}

COMPONENTS = ["Component 1", "Component 2", "Component 3"]

# MAIN
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
        model = ClusteringModel(data=preprocessed_df, algorithm=algoritme)
        if algoritme == 'gmm':
            model.best_k()
        else:
            model.elbow_method(max_clusters=MAX_CLUSTERS)
        model.fit()

        # Visualitzacions:
        # clustering_kmeans.plot_clusters_PCA_2d()
        # clustering_kmeans.plot_clusters_PCA_3d()
        # clustering_kmeans.plot_clusters_TSNE_2d()
        reduced_data = model.plot_clusters_TSNE_3d() 

        model.evaluate()

        # if algoritme == 'kmeans':
        #     print('\nModel kmeans')
        #     # clustering_kmeans = ClusteringModel(preprocessed_df, algorithm=algoritme)  
        #     # clustering_kmeans.elbow_method(max_clusters=50)
        #     # clustering_kmeans.fit()

        #     # # Visualitzacions:
        #     # # clustering_kmeans.plot_clusters_PCA_2d()
        #     # # clustering_kmeans.plot_clusters_PCA_3d()
        #     # # clustering_kmeans.plot_clusters_TSNE_2d()
        #     # reduced_data = clustering_kmeans.plot_clusters_TSNE_3d() 

        #     # # Grups segons els centroides
        #     # centroides, caracteristicas_relevantes = clustering_kmeans.analisi_components_centroides(preprocessed_df)
        #     # print("Característiques segons centroides:")
        #     # for cluster, data in caracteristicas_relevantes.items():
        #     #     print(f"Cluster {cluster}:")
        #     #     print(f"  Variables más altas: {data['top']}")
        #     #     print(f"  Variables más bajas: {data['low']}")
        #     # print()

        #     # # Grups segons les correlacions de cada dimensió de TSNE:
        #     # correlations_df, dic_correlacions = clustering_kmeans.analisi_components_tsne_correlacio(reduced_data, k=5)
        #     # print("Característiques segons components del TSNE:")
        #     # for comp, correlaciones in dic_correlacions.items():
        #     #     print(f"Componente: {comp}")
        #     #     print(f"  Top positivas: {correlaciones['top_positive']}")
        #     #     print(f"  Top negativas: {correlaciones['top_negative']}")

        #     # # Evaluació final del model
        #     # clustering_kmeans.evaluate()


        # elif algoritme == 'spectral':
        #     print('\nModel spectral')
        #     clustering_spectral = ClusteringModel(preprocessed_df, algorithm='kmeans')  
        #     clustering_spectral.elbow_method(max_clusters=50)
        #     # clust
        #     clustering_spectral = ClusteringModel(preprocessed_df, n_clusters=3, algorithm='spectral')
        #     clustering_spectral.fit()
        #     reduced_data = clustering_spectral.plot_clusters_TSNE_3d()
        #     clustering_spectral.evaluate()

        # elif algoritme == 'agglo':
        #     print('\nModel algomeratiu')
        #     clustering_agglo = ClusteringModel(preprocessed_df, n_clusters=3, algorithm='agglo')
        #     clustering_agglo.fit()
        #     reduced_data = clustering_agglo.plot_clusters_TSNE_3d() 
        #     clustering_agglo.evaluate()

        # elif algoritme == 'gmm':
        #     print('\nModel gaussian')
        #     clustering_gmm = ClusteringModel(preprocessed_df, n_clusters=3, algorithm='gmm')
        #     clustering_gmm.fit()
        #     reduced_data = clustering_gmm.plot_clusters_TSNE_3d() 
        #     clustering_gmm.evaluate()