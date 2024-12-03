# Data Cleaning: Preparació del Dataset per a l'Anàlisi
En aquesta secció del projecte, es descriu el procés de neteja i preparació del conjunt de dades abans de realitzar qualsevol anàlisi o modelatge. L'objectiu és garantir que el dataset sigui consistent, lliure d'errors i preparat per al seu ús en anàlisis posteriors.

## Tractament de Valors Nuls
Un dels passos més importants del procés de neteja de dades és gestionar els valors nuls. En aquest projecte, les columnes del dataset contenien valors nuls que podien afectar la qualitat de l'anàlisi. Per resoldre això, s'ha aplicat un mètode d'imputació específic per a cada tipus de dada.

- **Columnes categòriques**: Els valors nuls s'han substituït per la **moda** (el valor més freqüent) de la columna. Això permet garantir que la columna conservi el patró més comú sense afegir variabilitat artificial.
  
- **Columnes numèriques**: Els valors nuls s'han substituït per la **mitjana** de la columna. Això assegura que les dades numèriques siguin coherents amb la distribució original i evita distorsions en les anàlisis.

Aquesta estratègia permet mantenir la màxima quantitat d'informació disponible sense eliminar files completes, cosa que podria comportar la pèrdua de dades importants. Doncs, es garanteix la consistència i utilitat del conjunt de dades.

Per garantir la compatibilitat amb futures versions de pandas, s'ha evitat l'ús del paràmetre `inplace=True`, assegurant-se que les operacions siguin realitzades sobre el DataFrame original.

Finalment, totes les columnes del dataset estan lliures de valors nuls, tal com es pot verificar amb la funció `isnull().sum()`, que confirma que el nombre de valors nuls és 0.

### Gestió de valors nuls en sèries temporals
**por hacer!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!**

## Eliminació dels duplicats
Durant el procés de neteja de dades, hem revisat si hi ha files duplicades que poden afectar l'anàlisi. Per això, s'utilitza la funció `drop_duplicates()` de Pandas per eliminar qualsevol duplicat.

Després de realitzar aquesta operació, es comprova que no hi ha duplicats al conjunt de dades, ja que no es troben files repetides.

Aquesta etapa assegura que les dades són úniques i evita que la informació redundant influeixi en els resultats de l'anàlisi.

## Detecció i correcció dels errors tipogràfics
Una part important del procés de neteja de dades és assegurar que les columnes categòriques de text siguin consistents i lliures d'errors tipogràfics. Això inclou corregir problemes com:

- Diferències entre majúscules i minúscules (per exemple, "Barcelona" i "barcelona").
- Espais en blanc innecessaris al començament o al final dels valors (per exemple, " Barcelona ").
- Altres inconsistències que dificulten l'anàlisi.

Per garantir aquesta coherència, s'aplica una transformació a totes les columnes de tipus string. Això es fa convertint els valors a minúscules i eliminant els espais al voltant. Aquest procés s'automatitza mitjançant el següent enfocament:

- Identifiquem totes les columnes amb dades de text (`object` o `string`).
- Transformem els valors per assegurar la seva coherència.

Aquest pas garanteix que les dades textuals siguin estandarditzades i comparables, evitant problemes durant l'anàlisi o la creació de models.


## Conversió de tipus de dades incorrectes
**por hacer!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!**

## Normalització de dades categòriques
**por hacer!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!**

## Tractament dels outliers
**por hacer!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!**

## Estandarització o escalat de les dades numèriques
**por hacer!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!**

## Codificació de dades categòriques
**por hacer!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!**

## Revisió i manejament de valors constants
**por hacer!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!**

## Correlació i redundància entre variables
**por hacer!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!**

## Revisió del balanç de classes
**por hacer!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!**

## Validació final
**por hacer!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!**

