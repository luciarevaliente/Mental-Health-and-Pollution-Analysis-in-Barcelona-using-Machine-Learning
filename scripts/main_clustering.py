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
current_path = os.getcwd() # Obtenir la ruta actual

# ALGORITMES
ALGORITHMS = ['kmeans', 'agglo', 'gmm',] # Algoritmes de clústering a provar
opcio = int(input("\nEscull l'algoritme de clustering desitjat (recomanem gmm):\n\t1. kmeans\n\t2. agglo\n\t3. gmm\n\t4. Tots els algoritmes (seqüencialment)"))
if opcio in [1,2,3]:
    ALGORITHMS = ALGORITHMS[opcio]
    print(f'\nModel de clustering escollit: {ALGORITHMS}')
elif opcio == 4:
    print(f'\nModel de clustering escollit: {ALGORITHMS}')
else:
    raise KeyError("\nL'algoritme escollit no existeix")

# VARIABLE OBJECTIU
TARGET = 'estres'
print(f"La variable objectiu és {TARGET}")

# VARIABLES RELLEVANTS
opcio = int(input("\nEscull les variables rellevants per realitzar clustering (recomanem l'opció 2):\n\t1. Dataset complet\n\t2. Característiques importants generals dels models regressors\n\t3. Característiques importants del model XGBoost\n\t4. 4 característiques més importants del model XGBoost"))

# 1. Clústering per verificar patrons addicionals:
if opcio == 1:
    print(f'\nModel de visualització escollit: {VISUAL}')
    VARIABLES_RELLEVANTS = []
    PATH_FILENAME = os.path.join(current_path, "visualizations", "clusters", "dataset")

# 2. Clústering per verificar la separabilitat de les dades segons el model regressor:
## a. Característiques importants generals dels models regressors: 
elif opcio == 2:
    VARIABLES_RELLEVANTS = ['dayoftheweek', 'bienestar', 'energia', 'ordenador', 'alcohol', 'otrofactor', 'no2bcn_24h', 'no2gps_24h', 'covid_work']
    num_columns = ['dayoftheweek', 'bienestar', 'energia', 'no2bcn_24h', 'no2gps_24h']
    binary_columns = ['ordenador', 'alcohol', 'otrofactor']
    ordinal_columns = ['covid_work']
    nominal_columns = []
    PATH_FILENAME = os.path.join(current_path, "visualizations", "clusters", "general_important_features")
## b.	Característiques importants del model XGBoost:
elif opcio == 3:
    VARIABLES_RELLEVANTS = ['ordenador', 'otrofactor', 'dayoftheweek','district_gràcia','incidence_cat_physical incidence', 'smoke', 'district_sant andreu', 'bienestar', 'Totaltime']
    PATH_FILENAME = os.path.join(current_path, "visualizations", "clusters", "XGBoost_important_features")
## c.	4 característiques més importants del model XGBoost: 
elif opcio == 4:
    VARIABLES_RELLEVANTS = ['ordenador', 'otrofactor','dayoftheweek', 'bienestar']
    PATH_FILENAME = os.path.join(current_path, "visualizations", "clusters", "XGBoost_4th_important_features")
## d.   Error
else:
    raise KeyError("Les variables rellevants escollides no existeixen.")

# BEST K
opcio = input("Vols sel·leccionar el nombre de clústers òptim automàticament? y/n")
if opcio in ['yes', 'y', 'si', 's']:
    ESCOLLIR_K = False 
elif opcio in ['no', 'n']:
    ESCOLLIR_K = True # True = escollim k òptima manualment
    k = 12  # definim k
else:
    raise KeyError("No has escrit amb la sintaxi correcta.")

# VISUALITZAR RESULTATS
VISUAL = [None, "PCA", "TSNE"]
opcio = int(input("\nEscull l'algoritme de clustering desitjat (recomanem TSNE):\n\t1. No es vol veure els clústers\n\t2. PCA\n\t3. TSNE"))
if opcio in [1,2,3]:
    VISUAL = VISUAL[opcio]
    print(f'\nModel de visualització escollit: {VISUAL}')
else:
    raise KeyError("\nL'algoritme escollit no existeix")


# MAIN #################################################################################################################
if __name__ == "__main__":
    # 0. REDUCCIÓ DATASET
    if VARIABLES_RELLEVANTS:  
        initial_dataset = pd.read_pickle(PATH_DATASET)
        VARIABLES_RELLEVANTS = VARIABLES_RELLEVANTS + [TARGET]
        filtered_dataset = initial_dataset[VARIABLES_RELLEVANTS]
        filtered_dataset.to_pickle(PATH_DATASET_FILTERED)

        PATH_DATASET = PATH_DATASET_FILTERED

    # 1. PREPROCESSAMENT DADES
    whole_preprocessed_df, scaler_scale, scaler_mean, ordinal_encoder, nominal_encoder = preprocess(PATH_DATASET, TARGET)
    preprocessed_df = whole_preprocessed_df.drop(TARGET, axis=1)  # Eliminar variable target
    
    # 2. CLUSTERING
    for algoritme in ALGORITHMS:
        model = ClusteringModel(data=preprocessed_df, algorithm=algoritme)
        model.n_clusters = k if ESCOLLIR_K else model.best_k()
        model.fit()

        # 3. VISUALITZACIÓ
        os.makedirs(PATH_FILENAME, exist_ok=True)  # Crear la carpeta si no existeix
        
        if VISUAL is None:
            print("No es farà la visualització del clustering perquè VISUAL és None.")
        elif VISUAL == 'PCA':
            # model.plot_clusters_PCA_2d()
            model.plot_clusters_PCA_3d()
        elif VISUAL == 'TSNE':
            # model.plot_clusters_TSNE_2d()
            reduced_data = model.plot_clusters_TSNE_3d_animated(filename=f'{PATH_FILENAME}/{algoritme}_k{model.get_k()}_TSNE3d_animated.gif')
        else:
            print('Aquest mètode de visualització no està disponible')
        
        # 4. DISTRIBUCIÓ VARIABLE ESTRÈS
        print(f"\nDistribució de la variable target ('{TARGET}') per cluster:")
        target_distribution = model.analyze_target_distribution(whole_preprocessed_df, TARGET, save_path=f'{PATH_FILENAME}/{algoritme}_k{model.get_k()}_distribution.png')
            
        # 5. CARACTERÍSTIQUES DELS CENTROIDES
        centroides_df, _ = model.analisi_components_centroides(preprocessed_df)
        whole_preprocessed_df['Cluster'] = model.labels
        centroides_df['mean_target'] = whole_preprocessed_df.groupby('Cluster')[TARGET].transform('mean').unique()

        print("\nCentroides con media del target por cluster:")
        print(centroides_df)

        # 6. REVERSIÓ FORMATS PER INTERPRETAR DADES I CREAR PERFIL
        reverted_centroides = revert_preprocessing(
            centroides_df.drop("mean_target", axis=1),
            scaler_scale,
            scaler_mean,
            ordinal_columns=ordinal_columns,  
            binary_columns=binary_columns,   
            nominal_columns=nominal_columns, 
            ordinal_encoder=ordinal_encoder,  
            nominal_encoder=nominal_encoder 
        )
        reverted_centroides["mean_target"] = centroides_df["mean_target"]
        
        print("Centroides revertidos a valores originales:")
        print(reverted_centroides)

