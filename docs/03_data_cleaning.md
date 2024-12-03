# Data Cleaning
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
A continuació, hem realitzat una sèrie de conversions per assegurar-nos que els tipus de dades del conjunt de dades eren correctes i estiguessin formats adequadament per al seu processament posterior. En els subapartats explicarem les diferents operacions que hem realitzat, així com els canvis que s'han fet a les columnes per corregir incongruències de tipus de dades que proposava la documentació del dataset.

#### **Conversió de tipus de dades incorrectes**
1. **Comprovació inicial dels tipus de dades:**
   Primer, hem fa una revisió dels tipus de dades actuals a les columnes del conjunt de dades després de la importació. Bàsicament, per identificar qualsevol discrepància entre el tipus de dades esperat i el tipus de dades real. Això es fa amb el següent codi comentat:

   ```python
   # Comprovar els tipus de dades de totes les columnes
   for i, tipo in data.dtypes.items():
       print(i, tipo)
   ```

2. **Definició de les columnes segons el seu tipus esperat:**
   A continuació, hem classificat les columnes mal definides segons el tipus de dada desitjat.
   - **Col·leccions de dates** (`col_date_real`)
   - **Col·leccions d'hores** (`col_hour_real`)
   - **Col·leccions d'enteres** (`col_int_real`)
   - **Col·leccions de cadenes de text** (`col_str_real`)
   Aquestes columnes s'han de convertir al tipus de dades corresponent si és necessari.


3. **Conversió de les columnes:**
   Seguidament, es realitzen les conversió de tipus de dades per cada grup de columnes:

   - **Columnes de dates:** Converteix les columnes de tipus cadena a tipus `datetime` per gestionar correctament les dates.
     ```python
     for col in col_date_real:
         data[col] = pd.to_datetime(data[col], errors='coerce')  # 'coerce' per gestionar errors
     ```
   - **Columnes d'hores:** Converteix les columnes que representen hores en format `HH:MM:SS` a tipus `timedelta`, que és adequat per representar durades.
     ```python
     for col in col_hour_real:
         data[col] = pd.to_timedelta(data[col], errors='coerce')
     ```
   - **Columnes d'enteres:** Converteix les columnes que s'esperen com a enters a tipus numèric.
     ```python
     for col in col_int_real:
         data[col] = pd.to_numeric(data[col], errors='coerce')  # 'coerce' per substituir errors amb NaN
     ```
   - **Columnes de cadenes de text:** Es netegen i es converteixen a cadenes de text, per garantir que totes les dades siguin uniformes. A més, es fa una conversió a minúscules i s'eliminen els espais blancs.
     ```python
     for col in col_str_real:
         data[col] = data[col].astype(str)  # Assegura't que són strings
         data[col] = data[col].str.lower().str.strip()  # Estàndard
     ```

4. **Comprovació de NaN:**
   Després de realitzar les conversions, es comprova si hi ha valors `NaN` (valors no disponibles) en el conjunt de dades. Si s'han trobat errors en la conversió, aquests es marcaran com a `NaN`. Es realitza aquesta comprovació amb el següent codi:
   
   ```python
   # Comprovar si hi ha NaN en tot el DataFrame
   nan_check = data.isna().any().any()  # Retorna True si hi ha qualsevol NaN al DataFrame

   if nan_check:
       print("Hi ha valors NaN al DataFrame.")
       # Mostrar totes les files amb almenys un NaN
       nan_per_column = data.isna().sum()
       for i, v in nan_per_column.items():
           if v != 0:
               print(i, v)
   else:
       print("No hi ha valors NaN al DataFrame.")
   ```
    En executar el codi, observem que la transformació s'ha realitzat amb èxit i no hi ha cap dades amb valors null.

### Incongruències detectades
Abans de classificar les dades segons les llistes `col_date_real`, `col_hour_real`, `col_int_real` i `col_str_real`, vam efecutar la classificació `col_date_docu`, `col_int_docu` i `col_str_docu`.

En realitzar les conversions, s'han detectat algunes incongruències que han necessitat una correcció addicional: ens sortien valors Nan per a les variables `stroop_test`, `Totaltime_estimated`, `Houron` i `Houroff`, concretament 3348. Aquest valor ens ha sorprès ja que era exactament el nombre de files del dataset, per tant, hem sospitat que el tipus destijat de la conversió no era correcte.

Efecitvament, després de fer els prints respectius, hem comprovat que en la documentació s'establien els tipus de dades següents i en el csv eren unes alres:
   - La columna `stroop_test`, que s'esperava com a enter (`int`), contenia valors de tipus cadena (`str`) com "yes" o "no". Es va corregir convertint aquesta columna a tipus `str`.
   - La columna `Totaltime_estimated`, que també era d'enter (`int`), contenia valors de tipus cadena (`str`). Així mateix, es va convertir a `str` per reflectir correctament els seus valors.
   - Les columnes `Houron` i `Houroff`, que estaven com enters (`int`), es van convertir a `datetime` per representar-les com hores en lloc de simples enters.

És doncs, quan hem reformulat la classificació de les dades en les llistes i hem obtingut un resultat correcte, sense Nan. Aquestes conversions són necessàries per garantir que les dades siguin coherents amb el seu format esperat i per evitar errors en l'anàlisi posterior.


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

