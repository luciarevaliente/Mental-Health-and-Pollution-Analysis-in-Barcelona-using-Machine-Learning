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

    # Valors null
    valors_null = df.isnull()
    col_null = valors_null.any()
    print(f'\nHi ha {len(df.columns[col_null])} columnes amb valors null.')
    print(f'\nHi ha {len(df[valors_null])} valors null en el dataset.')
    