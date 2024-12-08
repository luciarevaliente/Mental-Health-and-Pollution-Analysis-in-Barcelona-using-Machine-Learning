"""
Script per analitzar dades de salut mental i contaminació de BCN
Creat per: Lucía Revaliente i Aránzazu Miguélez
Data de creació: 02/12/24
Descripció: Aquest script carrega, processa i analitza dades de salut mental i contaminació.
"""
# IMPORTS #########################################################################################################################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# VARIABLES CONSTANTS ###############################################################################################################
PICKLE_PATH = 'data/dataset.pkl'

columnes_a_eliminar = ['ID_Zenodo', # No té sentit analitzar els IDs
                        'yearbirth', # Tenim la variable 'age_yrs' que és equivalent
                        'precip_12h_binary', 'precip_24h_binary',  # Variables binàries
                        'no2bcn_12h_x30', 'no2bcn_24h_x30', 'no2gps_12h_x30', 'no2gps_24h_x30' # Provenen d'altres variables (tenen un factor de 30)
                        ]

# FUNCIONS #########################################################################################################################
def generar_boxplots(col_numeriques_filtered, output_folder="visualizations/boxplots"):
    """
    Aquesta funció genera i desa boxplots per a les columnes numèriques d'un DataFrame en una carpeta específica.

    Paràmetres:
    - col_numeriques_filtered: DataFrame amb les columnes numèriques a visualitzar.
    - output_folder: Ruta on es desaran els gràfics. Per defecte és 'visualizations/boxplots'.
    """
    # Comprovar si la carpeta ja existeix:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Crear la carpeta si no existeix

        # Número de boxplots (columnas numéricas)
        num_plots = len(col_numeriques_filtered.columns)
        
        # Crear una figura amb subgràfics per a tots els boxplots
        fig, axes = plt.subplots(nrows=(num_plots // 3) + (num_plots % 3 > 0), ncols=3, figsize=(15, 5 * ((num_plots // 3) + (num_plots % 3 > 0))))
        axes = axes.flatten()  # Aplanar l'array de subgràfics per iterar-hi fàcilment
        
        # Crear un boxplot per a cada columna numèrica en el subgràfic corresponent
        for i, col in enumerate(col_numeriques_filtered.columns):
            sns.boxplot(x=col_numeriques_filtered[col], ax=axes[i])  # Crear el boxplot en el subgràfic
            axes[i].set_title(f"Boxplot de {col}")  # Afegir títol a cada gràfic

        # Ajustar l'espai entre subgràfics
        plt.tight_layout()

        # Ruta completa per desar la figura amb tots els boxplots
        output_path = os.path.join(output_folder, "boxplots_totals.png")
        plt.savefig(output_path)  # Desar el gràfic amb tots els boxplots

        plt.close()  # Tancar el gràfic per alliberar memòria

        print(f"Boxplots guardats a la carpeta '{output_folder}' com 'boxplots_totals.png'.\n")
    else:
        print(f"La carpeta '{output_folder}' ja existeix. No s'han generat nous boxplots.\n")


def generar_violin_plots(col_numeriques_filtered, output_folder="visualizations/violin_plots"):
    """
    Aquesta funció genera i desa violin plots per a les columnes numèriques d'un DataFrame en una carpeta específica.

    Paràmetres:
    - col_numeriques_filtered: DataFrame amb les columnes numèriques a visualitzar.
    - output_folder: Ruta on es desaran els gràfics. Per defecte és 'visualizations/violin_plots'.
    """
    # Comprovar si la carpeta ja existeix:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Crear la carpeta si no existeix

        # Número de violin plots (columnes numèriques)
        num_plots = len(col_numeriques_filtered.columns)
        
        # Crear una figura amb subgràfics per a tots els violin plots
        fig, axes = plt.subplots(nrows=(num_plots // 3) + (num_plots % 3 > 0), ncols=3, figsize=(15, 5 * ((num_plots // 3) + (num_plots % 3 > 0))))
        axes = axes.flatten()  # Aplanar l'array de subgràfics per iterar-hi fàcilment
        
        # Crear un violin plot per a cada columna numèrica en el subgràfic corresponent
        for i, col in enumerate(col_numeriques_filtered.columns):
            sns.violinplot(x=col_numeriques_filtered[col], ax=axes[i])  # Crear el violin plot en el subgràfic
            axes[i].set_title(f"Violin Plot de {col}")  # Afegir títol a cada gràfic

        # Ajustar l'espai entre subgràfics
        plt.tight_layout()

        # Ruta completa per desar la figura amb tots els violin plots
        output_path = os.path.join(output_folder, "violin_plots_totals.png")
        plt.savefig(output_path)  # Desar el gràfic amb tots els violin plots

        plt.close()  # Tancar el gràfic per alliberar memòria

        print(f"Violin plots guardats a la carpeta '{output_folder}' com 'violin_plots_totals.png'.\n")
    else:
        print(f"La carpeta '{output_folder}' ja existeix. No s'han generat nous violin plots.\n")


# ANÀLISI CONTINGUT #########################################################################################################################
if __name__=="__main__":
    df = pd.read_pickle(PICKLE_PATH)

    print(f'\nDataset: {PICKLE_PATH}')

    # DIMENSIONS ########################################################################################################
    row, col = df.shape
    print(f'\nLes dimensions del dataset són {row}x{col}.')

    # TIPUS VARIABLES ###################################################################################################
    print(df.dtypes.value_counts())

    # VALORS NULL #######################################################################################################
    registres_null = df.isnull()
    caracteristiques_null = registres_null.any()
    print(f'\nHi ha {caracteristiques_null.sum()} característiques amb al menys un valor null.')
    print(f'\nHi ha {len(registres_null)} valors null en el dataset.')

    files_amb_null = df.isnull().any(axis=1).sum()
    proporcio_null = files_amb_null / len(df)
    print(f'\nLa proporció de files amb null és de {proporcio_null}\n')
    
    ## Distribució de valors null per columna
    distribucio = {}
    proporcio = 100/row
    distribucio_per_col = df.isnull().sum()
    for i, valor in distribucio_per_col.items():
        distribucio[i] = valor*proporcio
        # if (valor*proporcio <5):
        #     print(i)
        # if (valor*proporcio > 5) and (valor*proporcio < 10):
        #     print(i)
        # if (valor*proporcio > 10):
        #     print(i)
    # print(f'El diccionari amb les distribucions per columna (%) és: {distribucio}')


    # OUTLIERS ###########################################################################################################
    col_numeriques = df.select_dtypes(include=['float64', 'int64'])

    col_numeriques_filtered = col_numeriques.drop(columns=columnes_a_eliminar)

    generar_boxplots(col_numeriques_filtered)

    # DISTRIBUCIONS #######################################################################################################
    generar_violin_plots(col_numeriques_filtered)

    # VISUALITZACIONS SOBRE LES RELACIONS ENTRE LES VARIABLES ##############################################################
    # Configuració de l'estil de visualització
    sns.set(style="whitegrid")

    # 1. Distribució de registres per districtes
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, y='district', order=df['district'].value_counts().index)
    plt.title('Distribució de registres per districtes')
    plt.xlabel('Nombre de registres')
    plt.ylabel('Districtes')
    plt.show()

    # 2. Visualització de valors nuls per variable
    plt.figure(figsize=(12, 6))
    nulls = df.isnull().mean() * 100
    nulls.sort_values(ascending=False).plot(kind='bar')
    plt.title('Percentatge de valors nuls per variable')
    plt.ylabel('Percentatge de valors nuls (%)')
    plt.xlabel('Variables')
    plt.show()

    # 3. Boxplot de la variable "estres" per analitzar valors atípics
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='estres')
    plt.title('Distribució de la variable "estres"')
    plt.xlabel('Nivell d\'estrès')
    plt.show()

    # 4. Relació entre contaminació (no2bcn_24h) i salut mental (estres)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='no2bcn_24h', y='estres', alpha=0.6)
    plt.title('Relació entre NO2 i estrès')
    plt.xlabel('NO2 (24h)')
    plt.ylabel('Nivell d\'estrès')
    plt.show()

    # 5. Correlació entre variables numèriques

    # Filtrar només les columnes numèriques
    numeric_df = df.select_dtypes(include=[float, int])
    # Calcular la matriu de correlació només amb columnes numèriques
    correlation_matrix = numeric_df.corr()

    # Configurar el tamany de la figura i estil
    plt.figure(figsize=(20, 20))  # Incrementa el tamany de la figura
    sns.set(style="whitegrid")

    # Crear el heatmap
    sns.heatmap(
        correlation_matrix,
        annot=False,  
        cmap='coolwarm',
        cbar=True,
        square=True,
        xticklabels=True,
        yticklabels=True,
        linewidths=0.5
    )

    # Ajustar etiquetes
    plt.xticks(rotation=90, fontsize=8)  # Rota i ajusta el tamany de les etiquetes eix X
    plt.yticks(fontsize=8)  # Ajusta el tamany de les etiquetes del eix Y
    plt.title('Matriu de correlació entre variables numèriques', fontsize=15)
    plt.show()

    # 6. Distribució de variables de contaminació (no2bcn_24h i pm25bcn)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['no2bcn_24h'], kde=True, color='blue', bins=30, label='NO2')
    sns.histplot(df['pm25bcn'], kde=True, color='green', bins=30, label='PM2.5')
    plt.title('Distribució de contaminació (NO2 i PM2.5)')
    plt.xlabel('Concentració')
    plt.ylabel('Freqüència')
    plt.legend()
    plt.show()

    # 7. Comparació de "sueno" (son) per districte
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='district', y='sueno')
    plt.title('Distribució de les hores de son per districte')
    plt.xlabel('Districtes')
    plt.ylabel('Hores de son')
    plt.xticks(rotation=45)
    plt.show()

    # 8. Percentatge de valors nuls en variables clau
    variables_clau = ['estres', 'energia', 'bienestar', 'no2bcn_24h', 'pm25bcn']
    nuls_clau = df[variables_clau].isnull().mean() * 100
    plt.figure(figsize=(10, 6))
    nuls_clau.sort_values().plot(kind='barh', color='orange')
    plt.title('Percentatge de valors nuls en variables clau')
    plt.xlabel('Percentatge de valors nuls (%)')
    plt.ylabel('Variables')
    plt.show()

    # Proporció de registres de salut mental
    
    mentalhealth_counts = df['mentalhealth_survey'].value_counts(normalize=True) * 100
    ocurrence_counts = df['occurrence_mental'].value_counts(normalize=True) * 100
    bienestar_counts = df['bienestar'].value_counts(normalize=True) * 100

    mentalhealth_counts.index = mentalhealth_counts.index.astype(str)
    ocurrence_counts.index = ocurrence_counts.index.astype(str)
    bienestar_counts.index = bienestar_counts.index.astype(str)

    combined_data = pd.DataFrame({
        'Mental Health Survey': mentalhealth_counts,
        'Occurrence Mental': ocurrence_counts,
        'Bienestar': bienestar_counts
    }).fillna(0)  # Omplir valors nuls amb 0

    # Crear el gràfic
    plt.figure(figsize=(10, 6))
    combined_data.plot(kind='bar', figsize=(12, 6), alpha=0.8)

    # Etiquetes i títol
    plt.title('Proporció de registres de salut mental', fontsize=14)
    plt.xlabel('Estat de salut mental', fontsize=12)
    plt.ylabel('Percentatge (%)', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title='Categories')
    plt.tight_layout()
    plt.show()
        

