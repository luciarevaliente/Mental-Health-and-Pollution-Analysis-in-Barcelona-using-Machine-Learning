# IMPORTS #######################################################################################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# VARIABLES ######################################################################################
DATASET = "data/CitieSHealth_BCN_DATA_PanelStudy_20220414.csv"

# IMPORTACIÓ DATASET #############################################################################
df = pd.read_csv(DATASET)
# print(df.head())

# ANÀLISI CONTINGUT ##############################################################################
if __name__=="__main__":
    print(f'\nDataset: {DATASET}')

    # DIMENSIONS #############################################################
    row, col = df.shape
    print(f'\nLes dimensions del dataset són {row}x{col}.')

    # TIPUS VARIABLES #######################################################
    print(df.dtypes.value_counts())

    # VALORS NULL ###########################################################
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


    # OUTLIERS ################################################################
    col_numeriques = df.select_dtypes(include=['float64', 'int64'])
    # print(col_numeriques.columns)
    columnes_a_eliminar = ['ID_Zenodo', # No té sentit analitzar els IDs
                        'yearbirth', # Tenim la variable 'age_yrs' que és equivalent
                        'precip_12h_binary', 'precip_24h_binary',  # Variables binàries
                        'no2bcn_12h_x30', 'no2bcn_24h_x30', 'no2gps_12h_x30', 'no2gps_24h_x30' # Provenen d'altres variables (tenen un factor de 30)
                        ]
    col_numeriques_filtered = col_numeriques.drop(columns=columnes_a_eliminar)
    # print(len(col_numeriques.columns), len(col_numeriques_filtered.columns))

    # Carpeta on guardar els gràfics:
    output_folder = "visualizations/boxplots"

    # Comprovar si la carpeta ja existeix:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Crear la carpeta si no existeix

        # Número de boxplots (columnas numéricas)
        num_plots = len(col_numeriques_filtered.columns)
        
        # Crear una figura con subgráficos (subplots) para todos los boxplots
        fig, axes = plt.subplots(nrows=(num_plots // 3) + (num_plots % 3 > 0), ncols=3, figsize=(15, 5 * ((num_plots // 3) + (num_plots % 3 > 0))))
        axes = axes.flatten()  # Aplanar el array de ejes para iterar fácilmente
        
        # Crear un boxplot per a cada columna numèrica en el subplot correspondiente:
        for i, col in enumerate(col_numeriques_filtered.columns):
            sns.boxplot(x=col_numeriques_filtered[col], ax=axes[i])  # Crear el boxplot en el subgráfico
            axes[i].set_title(f"Boxplot de {col}")  # Títol del gràfic

        # Ajustar el espaciado entre subgráficos
        plt.tight_layout()

        # Ruta completa per guardar la figura amb tots els boxplots
        output_path = os.path.join(output_folder, "boxplots_totals.png")
        plt.savefig(output_path)  # Guardar la figura amb tots els boxplots

        plt.close()  # Tancar el gràfic per alliberar memòria

        print(f"Boxplots guardats a la carpeta '{output_folder}' com 'boxplots_totals.png'.\n")
    else:
        print(f"La carpeta '{output_folder}' ja existeix. No s'han generat nous boxplots\n.")

    # Anàlisi boxplots
    # for col in sorted(col_numeriques_filtered.columns):
        # print(f'La característica {col} --> MIN: {df[col].min()}, MAX: {df[col].max()}, MEAN: {df[col].mean()}')
    
    # print(df['date_all'])


    # DISTRIBUCIONS ###############################################################
    # Carpeta on guardar els gràfics:
    output_folder = "visualizations/violin_plots"

    # Comprovar si la carpeta ja existeix:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Crear la carpeta si no existeix

        # Número de violin plots (columnas numéricas)
        num_plots = len(col_numeriques_filtered.columns)
        
        # Crear una figura con subgráficos (subplots) para todos los violin plots
        fig, axes = plt.subplots(nrows=(num_plots // 3) + (num_plots % 3 > 0), ncols=3, figsize=(15, 5 * ((num_plots // 3) + (num_plots % 3 > 0))))
        axes = axes.flatten()  # Aplanar el array de ejes para iterar fácilmente
        
        # Crear un violin plot per a cada columna numèrica en el subplot correspondiente:
        for i, col in enumerate(col_numeriques_filtered.columns):
            sns.violinplot(x=col_numeriques_filtered[col], ax=axes[i])  # Crear el violin plot en el subgráfico
            axes[i].set_title(f"Violin Plot de {col}")  # Títol del gràfic

        # Ajustar el espaciado entre subgráficos
        plt.tight_layout()

        # Ruta completa per guardar la figura amb tots els violin plots
        output_path = os.path.join(output_folder, "violin_plots_totals.png")
        plt.savefig(output_path)  # Guardar la figura amb tots els violin plots

        plt.close()  # Tancar el gràfic per alliberar memòria

        print(f"Violin plots guardats a la carpeta '{output_folder}' com 'violin_plots_totals.png'.\n")
    else:
        print(f"La carpeta '{output_folder}' ja existeix. No s'han generat nous violin plots\n.")

    # PROPORCIÓ DE REGISTRES ##############################################
    # count_mentalhealth = df['mentalhealth_survey'].value_counts()
    # print(count_mentalhealth, "\n")
    # count_occurence_mental = df['occurrence_mental'].value_counts()
    # print(count_occurence_mental, "\n")
    # count_bienestar = df['bienestar'].value_counts()
    # print(count_bienestar, "\n")

