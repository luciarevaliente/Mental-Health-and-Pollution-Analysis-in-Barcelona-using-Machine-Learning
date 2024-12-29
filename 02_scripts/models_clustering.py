"""
Script per modelar la classe de Clustering.
Creat per: Lucía Revaliente i Aránzazu Miguélez
Data de creació: 12/12/24
Descripció: Aquest script carrega permet inicialitzar diversos models de clústering, amb funcionalitats que s'expliquen al llarg del codi.
"""
# IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from matplotlib.animation import FuncAnimation
from sklearn.manifold import TSNE
import os

# CLASSE
class ClusteringModel:
    def __init__(self, data, n_clusters=3, algorithm='kmeans'):
        """
        Constructor de la classe per modelar el clustering.

        :param data: DataFrame o ndarray amb les dades per al clustering.
        :param n_clusters: Nombre de clusters a identificar. Per defecte, 3.
        :param algorithm: Algorisme de clustering a utilitzar. Pot ser 'kmeans', 'spectral', 'agglo', 'gmm'.
        """
        self.data = data
        self.n_clusters = n_clusters
        self.algorithm = algorithm
        self.model = None
        self.labels = None

    def fit(self):
        """Entrenar el model segons l'algoritme seleccionat"""
        if self.algorithm == 'kmeans':
            self.model = KMeans(n_clusters=self.n_clusters, random_state=42)
        elif self.algorithm == 'agglo':
            self.model = AgglomerativeClustering(n_clusters=self.n_clusters)
        elif self.algorithm == 'gmm':
            self.model = GaussianMixture(n_components=self.n_clusters, random_state=42)
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

    def get_labels(self):
        """Obtenir les etiquetes de les prediccions"""
        return self.labels

    def set_clusters(self, n_clusters):
        """
        Establir el nombre de clusters.
        
        :param n_clusters: int.
        """
        self.n_clusters = n_clusters
        
    def set_algorithm(self, algorithm):
        """
        Establir l'algoritme de clustering.
        
        :param algorithm: string que determina l'algorisme de clústering a emprar.
        """
        self.algorithm = algorithm

    # CLUSTER SELECTION ------------------------------------------------------------------------------------
    def elbow_method(self, max_clusters=10):
        """
        Aplica el mètode del codo per trobar el millor nombre de clusters.

        :param max_clusters: int per determinar les iteracions màximes del mètode del colze
        :return: int. Valor K òptim
        """
        if self.algorithm == 'kmeans':
            model = KMeans(random_state=42)
        elif self.algorithm == 'agglo':
            model = AgglomerativeClustering()
        elif self.algorithm == 'gmm':
            model = GaussianMixture(random_state=42)
        else:
            raise ValueError(f"Algoritmo '{self.algorithm}' no és vàlid.")

        visualizer = KElbowVisualizer(model, k=(1, max_clusters))  # Provar entre 1 i max_clusters clusters

        visualizer.fit(self.data)  # Ajustar el visualitzador a les dades escalades
        visualizer.show()

        self.n_clusters = visualizer.elbow_value_  # Obtenir el nombre òptim de clusters 
        print(f"El número òptim de clusters és: {self.n_clusters}")

        return visualizer.elbow_value_

    def gmm_best_k(self, max_clusters=10):
        """
        Troba el millor nombre de clusters utilitzant el BIC (Criteri d'Informació Bayesià) en un GMM.

        :param max_clusters: Nombre màxim de clusters a avaluar (per defecte, 10).
        :return: El nombre òptim de clusters segons el BIC.
        """
        bic_scores = []
        
        # Avaluar models GMM per a cada nombre de clusters des d'1 fins a max_clusters
        for k in range(1, max_clusters + 1):
            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm.fit(self.data)
            bic_scores.append(gmm.bic(self.data))  # Calcular el BIC per al model ajustat

        # Trobar el nombre de clusters que minimitza el BIC
        best_k = np.argmin(bic_scores) + 1  # +1 perquè el rang comença en 1
        
        print(f"El nombre òptim de clusters segons el BIC és: {best_k}")
        self.n_clusters = best_k  # Establir el nombre òptim de clusters

        return best_k


    # VISUALITZACIÓ -----------------------------------------------------------------------------
    def plot_clusters_PCA_2d(self):
        """
        Visualitza les dades en un gràfic 2D acolorides segons el clúster.
        Utilitza PCA per reduir la dimensionalitat si hi ha més de 2 característiques.
        """
        if self.labels is None:
            raise ValueError("El model ha de ser ajustat abans de graficar els clusters.")
        
        # Reduir a 2 dimensions si és necessari
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
        plt.title(f'Clústers Visualitzats ({self.algorithm}). PCA 2D.k={self.n_clusters}.')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True)
        plt.show()

    def plot_clusters_PCA_3d(self):
        """
        Visualitza les dades en un gràfic 3D acolorides segons el clúster.
        Utilitza PCA per reduir la dimensionalitat si hi ha més de 3 característiques.
        """
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
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        fig.colorbar(scatter, label='Cluster')
        plt.show()
    
    def plot_clusters_TSNE_2d(self):
        """
        Visualitza les dades en un gràfic 2D acolorides segons el clúster.
        Utilitza TSNE per reduir la dimensionalitat si hi ha més de 2 característiques.
        """
        # Reduir a 2 dimensions si és necessari
        if self.data.shape[1] > 2:
            tsne = TSNE(n_components=2, random_state=42)
            reduced_data = tsne.fit_transform(self.data)
        else:
            reduced_data = self.data
        
        # Crear plot
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            reduced_data[:, 0], reduced_data[:, 1], 
            c=self.labels, cmap='viridis', s=50, alpha=0.6, edgecolor='k'
        )
        plt.title(f'Clusters visualitzats ({self.algorithm}). TSNE 2D. k={self.n_clusters}.')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True)
        plt.show()

    def plot_clusters_TSNE_3d(self):
        """
        Visualitza les dades en un gràfic 3D acolorides segons el clúster.
        Utilitza TSNE per reduir la dimensionalitat si hi ha més de 3 característiques.
        """
        # Reduir a 3 dimensions si és necessari
        if self.data.shape[1] > 3:
            tsne = TSNE(n_components=3, random_state=42)
            reduced_data = tsne.fit_transform(self.data)
        else:
            reduced_data = self.data
        
        # Crear el gràfic
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Graficar les dades en 3D
        scatter = ax.scatter(
            reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2],
            c=self.labels, cmap='viridis', s=50, alpha=0.6, edgecolor='k'
        )
        ax.set_title(f'Clusters visualitzats ({self.algorithm}). TSNE 3D. k={self.n_clusters}.')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        fig.colorbar(scatter, label='Cluster')
        plt.show()

        return reduced_data

    def plot_clusters_TSNE_3d_animated(self, filename="clusters_3d.gif"):
        """
        Visualitza les dades en un gràfic 3D acolorides segons el clúster.
        Utilitza TSNE per reduir la dimensionalitat si hi ha més de 3 característiques i guarda un GIF.

        :params filename: string. Arxiu on es guardarà el GIF.
        :return reduced_data: components obtinguts amb TSNE.
        """
        if self.labels is None:
            raise ValueError("El model ha de ser ajustat abans de graficar els clusters.")

        # Reduir a 3 dimensions si és necessari
        if self.data.shape[1] > 3:
            tsne = TSNE(n_components=3, random_state=42)
            reduced_data = tsne.fit_transform(self.data)
        else:
            reduced_data = self.data

        # Crear el gràfic 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Graficar les dades en 3D
        scatter = ax.scatter(
            reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2],
            c=self.labels, cmap='viridis', s=50, alpha=0.6, edgecolor='k'
        )
        ax.set_title(f'Clusters Visualitzats ({self.algorithm}). TSNE 3D. k={self.n_clusters}.')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')

        # Funció per actualitzar l'animació (rotació del gràfic)
        def update(frame):
            ax.view_init(elev=30, azim=frame)
            return fig,

        # Crear animació
        ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50, blit=False)

        # Guardar com a GIF
        ani.save(filename, writer='pillow', fps=20)
        plt.close(fig)
        print(f"Animació guardada com a {filename}")

        return reduced_data

    # ANÀLISI DE LES CARACTERÍSTIQUES DELS CENTROIDES -------------------------------------------------------------------------------------
    def analisi_components_centroides(self, preprocessed_df, k=5):
        """
        Analitza les característiques més rellevants per a cada clúster en funció de l'algorisme de clustering.

        Paràmetres:
        - preprocessed_df: DataFrame amb les característiques originals (no reduïdes) utilitzades per al clustering.
        - k: Nombre de característiques més altes/més baixes a analitzar (per defecte, 5).

        Retorna:
        - centroides_df: DataFrame amb els centres o mitjanes dels clústers per característic.
        - relevant_features: Diccionari que vincula els clústers amb les seves característiques més altes/baixes.
        """
        if self.labels is None or self.model is None:
            raise ValueError("El model ha de ser ajustat abans d'executar aquesta anàlisi.")

        if self.algorithm == 'kmeans':
            # KMeans: Accés directe als centroides
            if not hasattr(self.model, 'cluster_centers_'):
                raise ValueError("El model KMeans no té els centroides calculats.")
            centroids = self.model.cluster_centers_

        elif self.algorithm == 'agglo':
            # AgglomerativeClustering: Calcular les mitjanes de les característiques per cada clúster
            clusters = pd.DataFrame(self.data, columns=preprocessed_df.columns)
            clusters['Cluster'] = self.labels
            centroids = clusters.groupby('Cluster').mean().values

        elif self.algorithm == 'gmm':
            # GaussianMixture: Utilitzar les mitjanes calculades
            if not hasattr(self.model, 'means_'):
                raise ValueError("El model GaussianMixture no té les mitjanes calculades.")
            centroids = self.model.means_

        else:
            raise ValueError("Aquesta funció només admet els algorismes 'kmeans', 'agglo' o 'gmm'.")

        # Convertir els centroides a un DataFrame per una millor interpretació
        columns = preprocessed_df.columns  # Obtenir els noms de les característiques
        centroides_df = pd.DataFrame(centroids, columns=columns)
        centroides_df.index = [f"Cluster {i}" for i in range(len(centroides_df))]

        # Analitzar les característiques més altes/baixes per cada clúster
        relevant_features = {}
        for cluster_idx, row in centroides_df.iterrows():
            # Característiques més altes i més baixes amb els valors més alts/baixos dels centroides
            top_features = row.nlargest(k).index.tolist()
            low_features = row.nsmallest(k).index.tolist()
            relevant_features[cluster_idx] = {
                'top': top_features,
                'low': low_features
            }

        return centroides_df, relevant_features

    def analisi_components_tsne_correlacio(self, reduced_data, k=5):
        """
        Realitza una anàlisi de correlació entre les característiques originals del conjunt de dades
        i les components obtingudes mitjançant la reducció dimensional amb t-SNE. Aquesta funció calcula 
        les correlacions entre cada component de t-SNE i les variables originals, i retorna una 
        llista de les k característiques amb les correlacions més fortes (tant positives com negatives) 
        per a cada component.

        Paràmetres:
        - reduced_data: ndarray o DataFrame que conté les dades reduïdes a través de t-SNE.
        Ha de tenir el mateix nombre de files que self.data, i ha de contenir les components reduïdes 
        generades per t-SNE.
        - k: Nombre de correlacions més fortes (positives i negatives) a identificar.

        Retorna:
        - correlations_df: DataFrame amb les correlacions completes.
        - dic_correlacions: un diccionari que conté, per a cada component de t-SNE, les k variables 
        originals amb les correlacions més altes (positives i negatives). L'estructura és la següent:
        {
            'Component 1': {'top_positive': [var1, var2, ...], 'top_negative': [var3, var4, ...]},
            ...
        }
        """
        # Noms de les components t-SNE
        COMPONENTS = [f"Component {i+1}" for i in range(reduced_data.shape[1])]

        # Convertir les coordenades t-SNE a un DataFrame
        tsne_df = pd.DataFrame(reduced_data, columns=COMPONENTS)

        # Combinar les dades originals amb les components t-SNE
        combined_df = pd.concat([self.data, tsne_df], axis=1)

        # Calcular les correlacions completes entre les variables originals i les components t-SNE
        correlations = combined_df.corr().loc[self.data.columns, COMPONENTS]

        # Diccionari per guardar les correlacions més significatives
        dic_correlacions = {}

        for comp in COMPONENTS:
            # Seleccionar les correlacions de la component actual
            correlacions = correlations[comp]

            # Obtenir les k correlacions més fortes positives i negatives
            top_k_positive = correlacions.sort_values(ascending=False).head(k)
            top_k_negative = correlacions.sort_values(ascending=True).head(k)

            # Emmagatzemar al diccionari
            dic_correlacions[comp] = {
                'top_positive': top_k_positive.index.tolist(),
                'top_negative': top_k_negative.index.tolist()
            }

        # Retornar el DataFrame complet i el diccionari de correlacions
        return correlations, dic_correlacions

    def analyze_target_distribution(self, original_df, target_col, save_path=None):
        """
        Analitza la distribució de la variable target dins de cada clúster i guarda el gràfic si especifica una ruta.

        :param original_df: DataFrame original que conté la variable target.
        :param target_col: Nom de la columna de la variable target en el DataFrame original.
        :param save_path: Ruta on es guardarà el gràfic (opcional).
        """
        if self.labels is None:
            raise ValueError("El model ha de ser ajustat abans d'analitzar la distribució de la variable target.")

        # Crear un DataFrame amb els clústers i la variable target
        analysis_df = pd.DataFrame({
            'Cluster': self.labels,
            target_col: original_df[target_col]
        })

        # Calcular la distribució de la variable target per clúster
        distribution = analysis_df.groupby('Cluster')[target_col].value_counts(normalize=True).unstack(fill_value=0)

        # Visualitzar la distribució amb un gràfic de barres
        distribution.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis', alpha=0.85)
        plt.title(f'Distribució de {target_col} per Clúster ({self.algorithm})')
        plt.xlabel('Clúster')
        plt.ylabel('Proporció')
        plt.legend(title=target_col, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # Guardar el gràfic si es proporciona una ruta
        if save_path:
            # Crear la carpeta si no existeix
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Gràfic guardat a: {save_path}")

        plt.show()

        return distribution
