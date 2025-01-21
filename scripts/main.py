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
import main_regression
import main_clustering

def main():
    print("\nBenvingut/da a l'aplicació interactiva d'anàlisi de dades de salut mental i contaminació de BCN!")

    # Importació obligatoria del dataset
    print("\n------------------------- Importació del dataset -------------------------")
    load_data.main()
    print("\n----------------------- Fi importació del dataset -----------------------")

    print("\n\n------------------------- Nateja del dataset -------------------------")
    data_cleaning.main()
    print("\n----------------------- Fi nateja del dataset -----------------------")

    while True:
        # Menú interactivo
        print("\nSeleccioneu l'opció que voleu executar:")
        print("1. Anàlisis del dataset")
        print("2. Regressió")
        print("3. Clustering + Estudi Perfil")
        print("4. Sortir")

        try:
            opcio = int(input("\nIntrodueix el número de l'opció: "))
            
            if opcio == 1:
                print("\n------------------------- Anàlisis del dataset -------------------------")
                exploratory_analysis.main()
                print("\n----------------------- Fi anàlisis del dataset -----------------------")

            elif opcio == 2:
                print("\n------------------------------ Regresssió ------------------------------")
                main_regression.main()
                print("\n----------------------------- Fi regressió -----------------------------")

            elif opcio == 3:
                print("\n------------------------- Clustering + Estudi Perfil  -------------------------")
                main_clustering.main()
                print("\n------------------------ Fi clustering + estudi perfil  -------------------------")

            elif opcio == 4:
                print("\nSortint del programa. Fins aviat!\n")
                break

            else:
                print("\nOpció no vàlida. Si us plau, seleccioneu una opció del 1 al 5.")

        except ValueError:
            print("\nEntrada no vàlida. Si us plau, introdueix un número del 1 al 5.")


if __name__ == "__main__":
    main()