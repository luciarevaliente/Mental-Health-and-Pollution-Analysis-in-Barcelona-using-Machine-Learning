import pandas as pd

# VARIABLES
DATASET = "CitieSHealth_BCN_DATA_PanelStudy_20220414.csv"

# CODI
df = pd.read_csv(DATASET)
print(df.head())


caracteristiques = df.columns
# print(f'CARACTERÍSTIQUES: {caracteristiques}')

instancies = df.shape[0]
# print(f'Nº instàncies: {instancies}')