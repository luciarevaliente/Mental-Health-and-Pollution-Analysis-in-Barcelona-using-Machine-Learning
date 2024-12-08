# Importació de les dades CSV a Pickle

## Descripció
Aquest script, creat per **Lucía Revaliente** i **Aránzazu Miguélez**, carrega un conjunt de dades sobre la salut mental i la contaminació a Barcelona des d'un arxiu CSV i el desa en un arxiu de tipus **Pickle**.

### Objectiu
L'objectiu d'aquest script és carregar un conjunt de dades sobre la salut mental i la contaminació a Barcelona, disponible en format CSV, i convertir-lo en un arxiu **Pickle** per a una càrrega més ràpida i eficaç en scripts futurs. Aquest conjunt de dades recull informació sobre l'impacte de la contaminació en la salut mental de la població barcelonina, oferint valors sobre nivells de contaminació i diversos indicadors de salut mental.

### Funcionament
1. **Carregar les dades**: L'script utilitza `pandas` per llegir l'arxiu CSV que conté les dades.
2. **Convertir en Pickle**: Un cop carregades les dades en un DataFrame, aquestes es guarden en un fitxer **Pickle** (`dataset.pkl`) per poder-se utilitzar ràpidament en futures execucions o scripts sense la necessitat de tornar a carregar-les des del CSV.

### Per què utilitzar Pickle i no només CSV?
El format **Pickle** s'utilitza en aquest cas per diversos motius:

- **Eficàcia i velocitat**: Pickle és més ràpid que CSV quan es tracta de carregar dades en la memòria, ja que és un format binari que permet una càrrega més eficient.
- **Manteniment de tipus de dades**: A diferència de CSV, que guarda les dades en format de text, Pickle conserva tots els tipus de dades originals, com ara números, dates o valors booleans, sense necessitat de convertir-los.
- **Menor ús de recursos**: El fitxer Pickle ocupa menys espai en memòria que el CSV, fet que facilita la manipulació de conjunts de dades més grans.

### Ús del script
L'script carrega les dades des de `data/CitieSHealth_BCN_DATA_PanelStudy_20220414.csv` i les desa en un arxiu de tipus Pickle a la ubicació `data/dataset.pkl`. Un cop generat l'arxiu Pickle, es pot utilitzar en altres scripts per a analitzar o visualitzar les dades de manera més eficient.
