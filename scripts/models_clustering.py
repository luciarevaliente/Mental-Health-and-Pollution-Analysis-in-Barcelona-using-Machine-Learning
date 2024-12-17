# IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

# CLASSE
class ClusteringModel:
    def __init__(self, data, n_clusters=3, algorithm='kmeans'):
        """
        Constructor de la classe per a modelar el clustering.
        
        :param data: DataFrame o ndarray amb les dades a clustering.
        :param n_clusters: Nombre de clusters a identificar. Per defecte 3.
        :param algorithm: Algoritme de clustering a utilitzar. Pot ser 'kmeans', 'spectral', 'agglo', 'gmm'.
        """
        self.data = data
        self.n_clusters = n_clusters
        self.algorithm = algorithm
        self.model = None
        self.labels = None

    def fit(self):
        """Entrenar el model segons l'algoritme seleccionat"""
        if self.algorithm == 'kmeans':
            self.model = KMeans(n_clusters=self.n_clusters)
        elif self.algorithm == 'spectral':
            self.model = SpectralClustering(n_clusters=self.n_clusters, affinity='nearest_neighbors')
        elif self.algorithm == 'agglo':
            self.model = AgglomerativeClustering(n_clusters=self.n_clusters)
        elif self.algorithm == 'gmm':
            self.model = GaussianMixture(n_components=self.n_clusters)
        else:
            raise ValueError(f'Algoritme {self.algorithm} no reconegut')
        
        # Ajustar el model a les dades
        self.model.fit(self.data)
        
        # Assignar les etiquetes (clusters) al conjunt de dades
        if self.algorithm == 'gmm':
            # En el cas de GaussianMixture, s'ha d'utilitzar predict per obtenir les etiquetes
            self.labels = self.model.predict(self.data)
        else:
            self.labels = self.model.labels_

    def silhouette_score(self):
        """Calcular la puntuació de silueta per mesurar la qualitat del clustering"""
        return silhouette_score(self.data, self.labels)

    def get_labels(self):
        """Obtenir les etiquetes de les prediccions"""
        return self.labels

    def set_clusters(self, n_clusters):
        """Establir el nombre de clusters"""
        self.n_clusters = n_clusters
        
    def set_algorithm(self, algorithm):
        """Establir l'algoritme de clustering"""
        self.algorithm = algorithm

    def evaluate(self):
        """Evaluar el model amb la puntuació de silueta"""
        if self.n_clusters > 1:  # Solo calcular el silhouette si hay más de un cluster
            score = self.silhouette_score()
            print(f"Puntuació de silueta per {self.algorithm}: {score}")
            return score
        else:
            print(f"No es pot calcular el silhouette score. El número de clusters és {self.n_clusters}.")
            return None
   
    def elbow_method(self, max_clusters=10):
        """
        Aplica el mètode del codo per trobar el millor nombre de clusters.
        """
        # Crear el model de clustering seguint l'algoritme seleccionat
        if self.algorithm == 'kmeans':
            model = KMeans(random_state=42)
        elif self.algorithm == 'agglo':
            model = AgglomerativeClustering()
        elif self.algorithm == 'gmm':
            model = GaussianMixture(random_state=42)
        else:
            raise ValueError(f"Algoritmo '{self.algorithm}' no és vàlid.")

        # Crear el visualitzador per veure el mètode del codo
        visualizer = KElbowVisualizer(model, k=(1, max_clusters))  # Provar entre 1 i max_clusters clusters

        # Ajustar el visualitzador a les dades escalades
        visualizer.fit(self.data)
        visualizer.show()

        # Obtenir el nombre òptim de clusters (el valor del codo)
        self.n_clusters = visualizer.elbow_value_
        print(f"El número òptim de clusters és: {self.n_clusters}")

    def gmm_best_k(self, max_clusters=10):
        """
        Encuentra el mejor número de clusters utilizando el BIC (Bayesian Information Criterion) en un GMM.
        
        :param max_clusters: Número máximo de clusters a evaluar (por defecto 10).
        :return: El número óptimo de clusters según el BIC.
        """
        bic_scores = []
        
        # Evaluar modelos GMM para cada número de clusters desde 1 hasta max_clusters
        for k in range(1, max_clusters + 1):
            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm.fit(self.data)
            bic_scores.append(gmm.bic(self.data))  # Calcular el BIC para el modelo ajustado

        # Encontrar el número de clusters que minimiza el BIC
        best_k = np.argmin(bic_scores) + 1  # +1 porque el rango comienza en 1
        
        print(f"El número óptimo de clusters según el BIC es: {best_k}")
        self.n_clusters = best_k  # Establecer el número óptimo de clusters
        return best_k


    def plot_clusters_PCA_2d(self):
        """
        Visualiza los datos en un gráfico 2D coloreados según el cluster.
        Utiliza PCA para reducir la dimensionalidad si hay más de 2 características.
        """
        if self.labels is None:
            raise ValueError("El modelo debe ser ajustado antes de graficar los clusters.")
        
        # Reducir a 2 dimensiones si es necesario
        if self.data.shape[1] > 2:
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(self.data)
        else:
            reduced_data = self.data
        
        # Crear el scatter plot
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            reduced_data[:, 0], reduced_data[:, 1], 
            c=self.labels, cmap='viridis', s=50, alpha=0.6, edgecolor='k'
        )
        plt.title(f'Clusters Visualitzats ({self.algorithm}). PCA 2D.k={self.n_clusters}.')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True)
        plt.show()

    def plot_clusters_PCA_3d(self):
        """Visualitzar els clusters en un gràfic 3D"""
        # Si les dades tenen més de 3 dimensions, reduir a 3D amb PCA
        if self.data.shape[1] > 3:
            pca = PCA(n_components=3)
            reduced_data = pca.fit_transform(self.data)
        else:
            reduced_data = self.data

        # Crear el gràfic 3D amb els colors dels clusters
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(
            reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2],
            c=self.labels, cmap='viridis', s=50, alpha=0.8
        )
        ax.set_title(f'Clusters segons {self.algorithm}. PCA 3D.. k={self.n_clusters}.')
        ax.set_xlabel('Componente 1')
        ax.set_ylabel('Componente 2')
        ax.set_zlabel('Componente 3')
        fig.colorbar(scatter, label='Cluster')
        plt.show()
    
    def plot_clusters_TSNE_2d(self):
        """
        Visualiza los datos en un gráfico 2D coloreados según el cluster.
        Utiliza t-SNE para reducir la dimensionalidad si hay más de 2 características.
        """
        if self.labels is None:
            raise ValueError("El modelo debe ser ajustado antes de graficar los clusters.")
        
        # Reducir a 2 dimensiones si es necesario
        if self.data.shape[1] > 2:
            tsne = TSNE(n_components=2, random_state=42)
            reduced_data = tsne.fit_transform(self.data)
        else:
            reduced_data = self.data
        
        # Crear el scatter plot
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            reduced_data[:, 0], reduced_data[:, 1], 
            c=self.labels, cmap='viridis', s=50, alpha=0.6, edgecolor='k'
        )
        plt.title(f'Clusters Visualizados ({self.algorithm}). TSNE 2D. k={self.n_clusters}.')
        plt.xlabel('Componente 1')
        plt.ylabel('Componente 2')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True)
        plt.show()

    def plot_clusters_TSNE_3d(self):
        """
        Visualiza los datos en un gráfico 3D coloreados según el cluster.
        Utiliza t-SNE para reducir la dimensionalidad si hay más de 3 características.
        """
        if self.labels is None:
            raise ValueError("El modelo debe ser ajustado antes de graficar los clusters.")
        
        # Reducir a 3 dimensiones si es necesario
        if self.data.shape[1] > 3:
            tsne = TSNE(n_components=3, random_state=42)
            reduced_data = tsne.fit_transform(self.data)
        else:
            reduced_data = self.data
        
        # Crear el gráfico 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Graficar los datos en 3D
        scatter = ax.scatter(
            reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2],
            c=self.labels, cmap='viridis', s=50, alpha=0.6, edgecolor='k'
        )
        ax.set_title(f'Clusters Visualizados ({self.algorithm}). TSNE 3D. k={self.n_clusters}.')
        ax.set_xlabel('Componente 1')
        ax.set_ylabel('Componente 2')
        ax.set_zlabel('Componente 3')
        fig.colorbar(scatter, label='Cluster')
        plt.show()

        return reduced_data
    

    def analisi_components_centroides(self, preprocessed_df, k=5):
        """
        Analiza los centroides de los clusters para identificar las características más relevantes
        en cada cluster. Esto es útil para interpretar los resultados del clustering.

        Nota: Esta función solo es aplicable para el algoritmo KMeans.

        Retorna:
        - Un DataFrame que muestra los valores medios de cada característica para cada cluster.
        - Un diccionario que contiene, para cada cluster, las características con los valores medios
        más altos y más bajos.
        """
        if self.algorithm != 'kmeans':
            raise ValueError("Este análisis solo está disponible para el algoritmo KMeans.")

        if self.labels is None or self.model is None:
            raise ValueError("El modelo debe ser ajustado antes de realizar este análisis.")

        # Asegurarse de que el modelo tiene centroides
        if not hasattr(self.model, 'cluster_centers_'):
            raise ValueError("El modelo KMeans no tiene centroides calculados.")

        centroids = self.model.cluster_centers_  
        columns = preprocessed_df.columns  # Obtener los nombres de las columnas originales

        # Convertir los centroides a un DataFrame
        centroides_df = pd.DataFrame(centroids, columns=columns)
        centroides_df.index = [f'Cluster {i}' for i in range(len(centroides_df))]

        # Diccionario para almacenar las características más relevantes
        relevant_features = {}

        for cluster_idx, row in centroides_df.iterrows():
            # Obtener las características con los valores más altos y más bajos
            top_features = row.nlargest(k).index.tolist()  # Top k valores más altos
            low_features = row.nsmallest(k).index.tolist()  # Top k valores más bajos

            # Guardar en el diccionario
            relevant_features[cluster_idx] = {
                'top': top_features,
                'low': low_features
            }

        # Retornar el DataFrame de centroides y el diccionario
        return centroides_df, relevant_features

    def analisi_components_tsne_correlacio(self, reduced_data, k=5):
        """
        Realiza un análisis de correlación entre las características originales del conjunto de datos
        y las componentes obtenidas mediante reducción dimensional con t-SNE. Esta función calcula 
        las correlaciones entre cada componente de t-SNE y las variables originales, y devuelve una 
        lista de las k características con las correlaciones más fuertes (tanto positivas como negativas) 
        para cada componente.

        Parámetros:
        - reduced_data: ndarray o DataFrame que contiene los datos reducidos a través de t-SNE.
        Debe tener el mismo número de filas que self.data, y debe contener las componentes reducidas 
        generadas por t-SNE.
        - k: Número de correlaciones más fuertes (positivas y negativas) a identificar.

        Retorna:
        - correlations_df: DataFrame con las correlaciones completas.
        - dic_correlacions: un diccionario que contiene, para cada componente de t-SNE, las k variables 
        originales con las correlaciones más altas (positivas y negativas). La estructura es la siguiente:
        {
            'Component 1': {'top_positive': [var1, var2, ...], 'top_negative': [var3, var4, ...]},
            ...
        }
        """
        # Nombres de las componentes t-SNE
        COMPONENTS = [f"Component {i+1}" for i in range(reduced_data.shape[1])]

        # Convertir las coordenadas t-SNE a un DataFrame
        tsne_df = pd.DataFrame(reduced_data, columns=COMPONENTS)

        # Combinar datos originales con las componentes t-SNE
        combined_df = pd.concat([self.data, tsne_df], axis=1)

        # Calcular correlaciones completas entre variables originales y componentes t-SNE
        correlations = combined_df.corr().loc[self.data.columns, COMPONENTS]

        # Diccionario para guardar las correlaciones más significativas
        dic_correlacions = {}

        for comp in COMPONENTS:
            # Seleccionar las correlaciones de la componente actual
            correlacions = correlations[comp]

            # Obtener las k correlaciones más fuertes positivas y negativas
            top_k_positive = correlacions.sort_values(ascending=False).head(k)
            top_k_negative = correlacions.sort_values(ascending=True).head(k)

            # Almacenar en el diccionario
            dic_correlacions[comp] = {
                'top_positive': top_k_positive.index.tolist(),
                'top_negative': top_k_negative.index.tolist()
            }

        # Retornar el DataFrame completo y el diccionario de correlaciones
        return correlations, dic_correlacions


    # def analisi_components_tsne_correlacio(self, reduced_data, k=5):
    #     """
    #     Realiza un análisis de correlación entre las características originales del conjunto de datos
    #     y las componentes obtenidas mediante reducción dimensional con t-SNE. Esta función calcula 
    #     las correlaciones entre cada componente de t-SNE y las variables originales, y devuelve una 
    #     lista de las 5 características con las correlaciones más fuertes (tanto positivas como negativas) 
    #     para cada componente.

    #     Parámetros:
    #     - reduced_data: ndarray o DataFrame que contiene los datos reducidos a través de t-SNE.
    #     Debe tener el mismo número de filas que self.data, y debe contener las componentes reducidas 
    #     generadas por t-SNE.

    #     Retorna:
    #     - correlations_df: DataFrame con las correlaciones completas.
    #     - dic_correlacions: un diccionario que contiene, para cada componente de t-SNE, las 5 variables 
    #     originales con las correlaciones más altas (positivas y negativas). La estructura es la siguiente:
    #     {
    #         'Component 1': {'top_positive': [var1, var2, ...], 'top_negative': [var3, var4, ...]},
    #         ...
    #     }
    #     """
    #     COMPONENTS = ["Component 1", "Component 2", "Component 3"]

    #     # Convertir las coordenadas a un DataFrame
    #     tsne_df = pd.DataFrame(reduced_data, columns=COMPONENTS)

    #     # Analizar correlaciones
    #     combined_df = pd.concat([self.data, tsne_df], axis=1)
    #     correlations = combined_df.corr()[COMPONENTS]

    #     # Diccionario para guardar las correlaciones más significativas
    #     dic_correlacions = {}

    #     for comp in correlations.columns:
    #         correlacions = correlations[comp].drop(labels=[comp])

    #         # Obtener las k correlaciones más fuertes positivas y negativas
    #         top_k_positive = correlacions.sort_values(ascending=False).head(k)
    #         top_k_negative = correlacions.sort_values(ascending=True).head(k)

    #         # Almacenar en el diccionario
    #         dic_correlacions[comp] = {
    #             'top_positive': top_k_positive.index.tolist(),
    #             'top_negative': top_k_negative.index.tolist()
    #         }

    #     # Retornar el DataFrame completo y el diccionario de correlaciones
    #     return correlations, dic_correlacions
