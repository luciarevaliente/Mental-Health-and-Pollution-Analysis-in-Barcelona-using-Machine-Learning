# Data Cleaning: Preparació del Dataset per a l'Anàlisi
En aquesta secció del projecte, es descriu el procés de neteja i preparació del conjunt de dades abans de realitzar qualsevol anàlisi o modelatge. L'objectiu és garantir que el dataset sigui consistent, lliure d'errors i preparat per al seu ús en anàlisis posteriors.

## Tractament de Valors Nuls
Un dels passos més importants del procés de neteja de dades és gestionar els valors nuls. En aquest projecte, les columnes del dataset contenien valors nuls que podien afectar la qualitat de l'anàlisi. Per resoldre això, s'ha aplicat un mètode d'imputació específic per a cada tipus de dada.

- **Columnes categòriques**: Els valors nuls s'han substituït per la **moda** (el valor més freqüent) de la columna. Això permet garantir que la columna conservi el patró més comú sense afegir variabilitat artificial.
  
- **Columnes numèriques**: Els valors nuls s'han substituït per la **mitjana** de la columna. Això assegura que les dades numèriques siguin coherents amb la distribució original i evita distorsions en les anàlisis.

Aquesta estratègia permet mantenir la màxima quantitat d'informació disponible sense eliminar files completes, cosa que podria comportar la pèrdua de dades importants. Doncs, es garanteix la consistència i utilitat del conjunt de dades.

Per garantir la compatibilitat amb futures versions de pandas, s'ha evitat l'ús del paràmetre `inplace=True`, assegurant-se que les operacions siguin realitzades sobre el DataFrame original.

Finalment, totes les columnes del dataset estan lliures de valors nuls, tal com es pot verificar amb la funció `isnull().sum()`, que confirma que el nombre de valors nuls és 0.
