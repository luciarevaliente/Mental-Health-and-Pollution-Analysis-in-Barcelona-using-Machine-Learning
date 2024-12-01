import pandas as pd

# VARIABLES
DATASET = "CitieSHealth_BCN_DATA_PanelStudy_20220414.csv"

# IMPORTACIÓ DATASET
df = pd.read_csv(DATASET)
# print(df.head())

# ANÀLISI CONTINGUT
if __name__=="__main__":
    print(f'\nDataset: {DATASET}')

    # Dimensions
    row, col = df.shape
    print(f'\nLes dimensions del dataset són {row}x{col}.')

    # Valors nulls
    registres_null = df.isnull()
    caracteristiques_null = registres_null.any()
    print(f'\nHi ha {caracteristiques_null.sum()} característiques amb al menys un valor null.')
    print(f'\nHi ha {len(registres_null)} valors null en el dataset.')

    # Distribució de valors null per columna
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
    print(f'El diccionari amb les distribucions per columna (%) és: {distribucio}')