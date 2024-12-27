"""
01_load_data.py
Script per carregar les dades de salut mental i contaminació de BCN
Creat per: Lucía Revaliente i Aránzazu Miguélez
Data de creació: 02/12/24
Descripció: Aquest script carrega les dades de salut mental i contaminació en format CSV i les guarda en un arxiu Pickle.
"""
# IMPORTACIÓ
import pandas as pd

# VARIABLES CONSTANTS
CSV_PATH ='data/CitieSHealth_BCN_DATA_PanelStudy_20220414.csv'
PICKLE_PATH = 'data/dataset.pkl'

# MAIN
if __name__ == '__main__':
    # Carguem l'arxiu CSV en un DataFrame
    df = pd.read_csv(CSV_PATH)

    # Guardem el DataFrame com un arxiu pickle per fer-se servir en altres scripts
    df.to_pickle(PICKLE_PATH)

    print('Arxiu guardat en format Pickle!')