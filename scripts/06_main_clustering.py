"""
Antes:
    1. Eliminar valores null, escalar los datos y seleccionar las variables.
    2. Determinar número óptimo de clusters
    3. Aplicar algoritmo de clústering
"""
# IMPORTACIÓ
from preprocess import preprocess
from models_clustering import ClusteringModel
import matplotlib.pyplot as plt
import os

# VARIABLES CONSTANTS
PATH_DATASET = "data/cleaned_dataset.pkl"  # Dataset natejat
ALGORITHMS = ['kmeans', 'spectral', 'agglo', 'gmm']  # Algoritmes de clústering a testejar
TARGET = 'estres'


# Obtener la ruta actual
current_path = os.getcwd()

VARIABLES_RELLEVANTS = []
PATH_FILENAME = os.path.join(current_path, "visualizations", "clustering", "dataset")

# Les 4 variables més rellevants
VARIABLES_RELLEVANTS = ['ordenador', 'otrofactor','dayoftheweek', 'bienestar']
PATH_FILENAME = os.path.join(current_path, "visualizations", "clustering", "4th_important_features")

# Les 8 variables més rellevants
# VARIABLES_RELLEVANTS = ['ordenador', 'otrofactor','dayoftheweek', 'bienestar', 'µgm3', 'mean_congruent', 'Totaltime', 'energia']
# PATH_FILENAME = os.path.join(current_path, "visualizations", "clustering", "8th_important_features")

ALGORITHMS = ['kmeans', 'agglo', 'gmm']  # Algoritmes de clústering a testejar: 
REPETICIO = 1

MAX_CLUSTERS = 50
COMPONENTS = ["Component 1", "Component 2", "Component 3"]
# K = 3

# MAIN
if __name__=="__main__":
    # 1 i 2. Codificació i escalat
    whole_preprocessed_df = preprocess(PATH_DATASET, TARGET) # Carreguem les dades i les preprocessem
    preprocessed_df = whole_preprocessed_df.drop(TARGET, axis=1)  # Eliminem la variable a partir de la qual volem fer clústering
    # print(preprocessed_df.columns)

    # 3. Si hi ha variables rellevants, reduïm la dimensió del dataset
    if VARIABLES_RELLEVANTS: 
        preprocessed_df = preprocessed_df[VARIABLES_RELLEVANTS]  # Modifiquem el datases

    # 4. Elecció de l'algoritme de clústering: Inicialitzem la classe i provem
    for algoritme in ALGORITHMS:
        # plt.ion()
        for i in range(REPETICIO):
            print(f'\nModel {algoritme}')
            model = ClusteringModel(data=preprocessed_df, algorithm=algoritme)
            # model.n_clusters=5
            # model.n_clusters=4
            if algoritme == 'gmm':
                best_k = model.gmm_best_k()
            else:
                best_k = model.elbow_method(max_clusters=MAX_CLUSTERS)
            print(f'Best number of clusters: {best_k}')
            # break
            model.fit()

            # Visualitzacions:
            # model.plot_clusters_PCA_2d()
            # model.plot_clusters_PCA_3d()
            # model.plot_clusters_TSNE_2d()
            # reduced_data = model.plot_clusters_TSNE_3d() 
            reduced_data = model.plot_clusters_TSNE_3d_animated(filename=f'{PATH_FILENAME}/{algoritme}_k{best_k}_iter{i+1}_TSNE3d_animated.gif')

            # Analizar distribución de la variable target
            print(f"Distribución de la variable target ('{TARGET}') por cluster:")
            target_distribution = model.analyze_target_distribution(whole_preprocessed_df, TARGET, save_path=f'{PATH_FILENAME}/{algoritme}_k{best_k}_iter{i+1}_distribution.png')
            print(target_distribution)

        # # Grups segons les correlacions de cada dimensió de TSNE: -------------------------------------------
        # correlations_df, dic_correlacions = model.analisi_components_tsne_correlacio(reduced_data, k=K)
        # print("Característiques segons components del TSNE:")
        # for comp, correlaciones in dic_correlacions.items():
        #     print(f"Componente: {comp}")
        #     print(f"  Top positivas: {correlaciones['top_positive']}")
        #     print(f"  Top negativas: {correlaciones['top_negative']}")

        # # Grups segons els centroides ------------------------------------------------------------------------
        # centroides, caracteristicas_relevantes = model.analisi_components_centroides(preprocessed_df)
        # print("Característiques segons centroides:")
        # for cluster, data in caracteristicas_relevantes.items():
        #     print(f"Cluster {cluster}:")
        #     print(f"  Variables más altas: {data['top']}")
        #     print(f"  Variables más bajas: {data['low']}")
        # print()
