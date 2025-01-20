"""
06_main_clustering.py
Script per fer clustering amb diversos algoritmes i característiques.
Creat per: Lucía Revaliente i Aránzazu Miguélez
Data de creació: 12/12/24
Descripció: Aquest script carrega permet fer clustering amb diversos algoritmes i característiques
"""
# IMPORTACIÓ #####################################################################################################
from preprocess import preprocess, revert_preprocessing
from models_clustering import ClusteringModel
import os
import sys
import pandas as pd

# VARIABLES CONSTANTS ###########################################################################################
PATH_DATASET = "data/cleaned_dataset.pkl"  # Dataset netejat
PATH_DATASET_FILTERED = "data/filtered_dataset.pkl" # Dataset per sel·leccionar només les característiques desitjades

# ALGORITHMS = ['kmeans', 'agglo', 'gmm'] # Algoritmes de clústering a provar
ALGORITHMS = ['gmm']  # Algoritmes de clústering a provar: 'kmeans', 'agglo', 'gmm'
TARGET = 'estres'

current_path = os.getcwd() # Obtenir la ruta actual

# 1. Clústering per verificar patrons addicionals:
# VARIABLES_RELLEVANTS = []
# PATH_FILENAME = os.path.join(current_path, "visualizations", "clusters", "dataset")

# 2. Clústering per verificar la separabilitat de les dades segons el model regressor:
## a. Característiques importants generals dels models regressors: 
VARIABLES_RELLEVANTS = ['dayoftheweek', 'bienestar', 'energia', 'ordenador', 'alcohol', 'otrofactor', 'no2bcn_24h', 'no2gps_24h', 'covid_work']
num_columns = ['dayoftheweek', 'bienestar', 'energia', 'no2bcn_24h', 'no2gps_24h']
binary_columns = ['ordenador', 'alcohol', 'otrofactor']
ordinal_columns = ['covid_work']
nominal_columns = []
PATH_FILENAME = os.path.join(current_path, "visualizations", "clusters", "general_important_features")

# var_binaries = ['ordenador', 'alcohol', 'otrofactor', '']

## b.	Característiques importants del model XGBoost:
# VARIABLES_RELLEVANTS = ['ordenador', 'otrofactor', 'dayoftheweek','district_gràcia','incidence_cat_physical incidence', 'smoke', 'district_sant andreu', 'bienestar', 'Totaltime']
# PATH_FILENAME = os.path.join(current_path, "visualizations", "clusters", "XGBoost_important_features")

## c.	4 característiques més importants del model XGBoost: 
# VARIABLES_RELLEVANTS = ['ordenador', 'otrofactor','dayoftheweek', 'bienestar']
# PATH_FILENAME = os.path.join(current_path, "visualizations", "clusters", "XGBoost_4th_important_features")

AGRUPATED = False  # agrupem les característiques mal balancejades: 
# AGRUPAR = [10, 9] # [classe a canviar, classe objectiu]

ESCOLLIR_K = True # True = escollim k òptima manualment
k = 12  # definim k

VISUAL = None # escollim visualització: [PCA, TSNE]

# MAIN #################################################################################################################
if __name__ == "__main__":
    if VARIABLES_RELLEVANTS:  # Seleccionar características relevantes
        initial_dataset = pd.read_pickle(PATH_DATASET)
        VARIABLES_RELLEVANTS = VARIABLES_RELLEVANTS + [TARGET]
        filtered_dataset = initial_dataset[VARIABLES_RELLEVANTS]
        filtered_dataset.to_pickle(PATH_DATASET_FILTERED)

        PATH_DATASET = PATH_DATASET_FILTERED

    # 1. Preprocesar los datos
    whole_preprocessed_df, scaler_scale, scaler_mean, ordinal_encoder, nominal_encoder = preprocess(PATH_DATASET, TARGET)
    print(scaler_scale, scaler_mean)

    if AGRUPATED:  # Agrupar clases
        whole_preprocessed_df = whole_preprocessed_df.replace({AGRUPAR[0]: AGRUPAR[1]})
    
    preprocessed_df = whole_preprocessed_df.drop(TARGET, axis=1)  # Eliminar variable target
    
    for algoritme in ALGORITHMS:
        print(f'\nModel {algoritme}')
        model = ClusteringModel(data=preprocessed_df, algorithm=algoritme)
        model.n_clusters = k if ESCOLLIR_K else model.best_k()
        model.fit()

        # Visualitzacions:
        os.makedirs(PATH_FILENAME, exist_ok=True)  # Crear la carpeta si no existeix
        
        if VISUAL is None:
            print("No es farà la visualització del clustering perquè VISUAL és None.")
        elif VISUAL == 'PCA':
            # model.plot_clusters_PCA_2d()
            model.plot_clusters_PCA_3d()
        elif VISUAL == 'TSNE':
            # model.plot_clusters_TSNE_2d()
            # reduced_data = model.plot_clusters_TSNE_3d() 
            reduced_data = model.plot_clusters_TSNE_3d_animated(filename=f'{PATH_FILENAME}/{algoritme}_k{model.get_k()}_TSNE3d_animated.gif')
        else:
            print('Aquest mètode de visualització no està disponible')
        

        # print(f"\nDistribució de la variable target ('{TARGET}') per cluster:")
        # target_distribution = model.analyze_target_distribution(whole_preprocessed_df, TARGET, save_path=f'{PATH_FILENAME}/{algoritme}_k{model.get_k()}_distribution.png')
            
        # Calcular centroides y media del target por cluster
        centroides_df, _ = model.analisi_components_centroides(preprocessed_df)
        whole_preprocessed_df['Cluster'] = model.labels
        centroides_df['mean_target'] = whole_preprocessed_df.groupby('Cluster')[TARGET].transform('mean').unique()

        print("\nCentroides con media del target por cluster:")
        print(centroides_df)

        # Guardar resultados
        # os.makedirs(PATH_FILENAME, exist_ok=True)
        # centroides_df.to_csv(f'{PATH_FILENAME}/{algoritme}_k{model.n_clusters}_centroids_with_target.csv')

       # Revertimos las transformaciones realizadas en los centroides
        reverted_centroides = revert_preprocessing(
            centroides_df.drop("mean_target", axis=1),
            scaler_scale,
            scaler_mean,
            ordinal_columns=ordinal_columns,  # Diccionario de columnas ordinales y sus categorías
            binary_columns=binary_columns,    # Lista de columnas binarias
            nominal_columns=nominal_columns, # Lista de columnas nominales
            ordinal_encoder=ordinal_encoder,  # Encoder usado para las ordinales (si está disponible)
            nominal_encoder=nominal_encoder   # Encoder usado para las nominales (si está disponible)
        )
        reverted_centroides["mean_target"] = centroides_df["mean_target"]
        # Mostrar resultado
        print("Centroides revertidos a valores originales:")
        print(reverted_centroides)



        # # # Mostrar les característiques i els valors dels centroides per cada cluster
        # centroides_df, caracteristiques_relevantes = model.analisi_components_centroides(preprocessed_df)
        # # centroides_df, caracteristiques_relevantes = model.analisi_components_centroides(preprocessed_df)
        # print("\n\nCentorides:")
        # print(centroides_df)
        # # df['columna'] = df['columna'].map({1: 'yes', -1: 'no'})

        # binary_centroids = centroides_df[var_binaries]
        # for col in var_binaries:
        #     binary_centroids[col] = binary_centroids[col] .map({1: 'yes', -1: 'no'})

        # scaled_centroids = centroides_df[var_escalades]
        # scaled_centroids_index = [preprocessed_df.columns.get_loc(col) for col in var_escalades]

        # selected_scale = scaler_scale[scaled_centroids_index]
        # selected_mean = scaler_mean[scaled_centroids_index]

        # original_centroids = scaled_centroids * selected_scale + selected_mean

        # final = pd.concat([binary_centroids, original_centroids], axis=1)
        # print(final) 
        # sys.exit()


        # Grups segons les correlacions de cada dimensió de TSNE: -------------------------------------------
        # correlations_df, dic_correlacions = model.analisi_components_tsne_correlacio(reduced_data, k=K)
        # print("Característiques segons components del TSNE:")
        # for comp, correlaciones in dic_correlacions.items():
        #     print(f"Component: {comp}")
        #     print(f"  Top positives: {correlaciones['top_positive']}")
        #     print(f"  Top negatives: {correlaciones['top_negative']}")

        # Grups segons els centroides ------------------------------------------------------------------------
        # centroides, caracteristiques_relevantes = model.analisi_components_centroides(preprocessed_df)
        # print("Característiques segons centroides:")
        # for cluster, data in caracteristiques_relevantes.items():
        #     print(f"Cluster {cluster}:")
        #     print(f"  Variables més altes: {data['top']}")
        #     print(f"  Variables més baixes: {data['low']}")
        # print()

    # RESULTAT = {'CLUSTER 0': {'ordenador': 1, 'bienestar': 0.09},
    #             'CLUSTER 1': {'otrofactor': 1, 'dayoftheweek': 0.40},
    #             'CLUSTER 2': {'dayoftheweek': 1.47, 'bienestar': 0.47},
    #             'CLUSTER 3': {'ordenador': 1, 'otrofactor': 1},
    #             'CLUSTER 4': {'bienestar': 0.16}}
    
    # RESULTAT: 
    # Cluster Cluster 0:
    # Variables més altes:
    #     ordenador: 1.0
    #     bienestar: 0.09360505140624129
    #     dayoftheweek: -0.4085219179740924
    #     otrofactor: -1.0
    # Variables més baixes:
    #     otrofactor: -1.0
    #     dayoftheweek: -0.4085219179740924
    #     bienestar: 0.09360505140624129
    #     ordenador: 1.0

    # Cluster Cluster 1:
    # Variables més altes:
    #     otrofactor: 1.0
    #     dayoftheweek: 0.4019593168821394
    #     bienestar: -0.24698225039927516
    #     ordenador: -1.0
    # Variables més baixes:
    #     ordenador: -1.0
    #     bienestar: -0.24698225039927516
    #     dayoftheweek: 0.4019593168821394
    #     otrofactor: 1.0

    # Cluster Cluster 2:
    # Variables més altes:
    #     dayoftheweek: 1.4731547951186976
    #     bienestar: 0.472832262282025
    #     otrofactor: -1.0
    #     ordenador: -1.0
    # Variables més baixes:
    #     ordenador: -1.0
    #     otrofactor: -1.0
    #     bienestar: 0.472832262282025
    #     dayoftheweek: 1.4731547951186976

    # Cluster Cluster 3:
    # Variables més altes:
    #     ordenador: 1.0
    #     otrofactor: 1.0
    #     dayoftheweek: -0.38828883503761097
    #     bienestar: -0.5854100455473495
    # Variables més baixes:
    #     bienestar: -0.5854100455473495
    #     dayoftheweek: -0.38828883503761097
    #     otrofactor: 1.0
    #     ordenador: 1.0

    # Cluster Cluster 4:
    # Variables més altes:
    #     bienestar: 0.16137996374942615
    #     dayoftheweek: -0.014969355439394512
    #     otrofactor: -0.9999999999999992
    #     ordenador: -0.9999999999999992
    # Variables més baixes:
    #     ordenador: -0.9999999999999992
    #     otrofactor: -0.9999999999999992
    #     dayoftheweek: -0.014969355439394512
    #     bienestar: 0.16137996374942615
