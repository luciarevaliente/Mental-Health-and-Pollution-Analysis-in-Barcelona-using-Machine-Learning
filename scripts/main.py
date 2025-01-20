"""
main.py
Script principal per processar les dades de salut mental i contaminació de BCN
Creat per: Lucía Revaliente 
Data de creació: 20/01/25
Descripció: Aquest script neteja les dades de salut mental i contaminació, les analitza, empra regressió i clústering. Finalment, crea un perfil de l'estrès poblacional. 
"""
import load_data
import exploratory_analysis
import data_cleaning

def main():
    "Benvingut/da a ..."
    print("------------------------- Importació del dataset -------------------------")
    load_data.main()

    print("------------------------- Anàlisis del dataset -------------------------")
    exploratory_analysis.main()

    print("------------------------- Nateja del dataset -------------------------")
    data_cleaning.main()

    print("------------------------- Importació del dataset -------------------------")


if __name__ == "__main__":
    main()