# Planificació setmanal: Objectius i Preguntes Clau
En el següent README exposem els objectius setmanals per desenvolupar el projecte, juntament amb preguntes necessàries pel correcte plantejament i execució del model.

## SETMANA 1: Preparació de Dades i Comprensió del Problema
### 1.1 Objectius:
**Objectiu principal:** Assegurar una base sòlida per al model, treballant amb les dades i definint clarament el problema.
1. **Definició del problema:**
    - Comprendre l'objectiu del model
    - Identificar mètriques d'avaluació

2. **Recopilació i exploració de dades:** 
    - Entendre les característiques de les dades (mida, tipus valors)
    - Realitzar anàlisi exploratori de les dades per identificar tendències i patrons clau
    - Script: `02_exploratory_analysis.py` (conté el codi)
    - Docs: `02_exploratory_analysis.md` (conté l'explicació del procediment i l'anàlisi dels resultats)

3. **Preparació de les dades:**
    - Neteja de dades: eliminar/transformar valors null i outliers, duplicats...
    - Transformació de les dades: normalització, escalat, codificació de variables categòriques...
    - Dividir les dades en conjunts d'entrenament, validació i prova.
    - Script: `03_data_cleaning.py` (conté el codi)
    - Docs: `03_data_cleaning.md` (conté l'explicació del procediment i de la neteja, normalització, escalat i codificació de les dades)

4. **Establir hipòtesis**
    - Formular hipòtesis inicials sobre les relacions entre les variables


### 1.2. Preguntes a respondre:
  1. **Tenim suficients dades?**
      - Considerem que sí tenim suficients dades ja que:
      - Proporció de files/columnes: el número de files és 35x vegades el número de columnes, quan el mínim és 10x. Doncs, no caldrà fer cap redimensió del dataset, regularitzacions (Ridge, Lasso) o validació creuada.
    
  2. **Les dades són consistents i adequades per al modelatge?**
     - Les dades presenten valors null de manera dispersa al llarg del dataset. En total hi ha 3348 registres null, d'un dataset amb 3348x95 valors. Per tant, les dades inconsistents representen un 1.05% de les dades. Igualment, es poden transformar perquè no suposin cap problema posterior.
     - Si analitzem les distribucions per columnes, aquestes tenen un 7% dels valors com a null. Doncs, les característiques del dataset contenen informació suficient per realitzar el modelatge i no cal eliminar cap columna. Al menys fins que analitzem amb profunditat la importància de cada variable.
     - Si observem les dades categòriques, la majoria estan ben definides. Tot i això, hem detectat errors en les variables `estres`, `sueno`, `bienestar`, `energia`, les quals contenen soroll i estan emmagatzemades amb números float.
     - A més, les columnes disponibles són rellevants i s'han identificat correlacions significatives entre salut mental i contaminació, so i estrés, meteorologia i salut, etc.
     - En conclusió, les dades necessiten petits ajustos com transformació de nulls, normalització de variables categòriques, etc. perquè siguin adequades per al modelatge.
       
  4. **Com s'interpreten les dades i el seus valors?**
      - En el document `02_exploratory_analysis.md`, en l'apartat de *Característiques* hi ha un resum amb les diferents dades. En el fitxer `03_data_cleaning.md`, s'explica en l'apartat *Incongruències detectades* com els tipus de les dades està explicat en la documentació del dataset, excepte les variables `stroop_test`, `Totaltime_estimated`, `Houron` i `Houroff`, que són d'un tipus diferent.

  5. **Hi ha valors atípics?**
      - Els valors atípics s'exposen en el fitxer `03_data_cleaning.md`, concretament en l'apartat *Outliers*.
      - Per identificar-los, hem fet servir boxplots. Es poden observar en el repositori: `visualizations/boxplots`.
      - Que hi hagi valors atípics no significa que aquests siguin incorrectes. En general, en el nostre dataset els outliers estan contextualitzats com a casos extraordinaris (ex. vents forts, altes concentracions de NO₂). Per tant, són dades inusuals però vàlides. Doncs, els boxplots amb els quals hem analitzats la info, mostren la variabilitat de les dades, etc. 
    
  6. **Com s'han recopilat les dades? Quin rang geogràfic i temporal hi ha?**
     -  Dades medioambientals i meteorològiques: s'han recollit de serveis meteorològics oficials i estacions de monitoreig ambiental de Barcelona. Algunes variables: `BCμg`, `noise_total_LDEN_55`, `humi_24h`, etc.
     -  Indicadors de salut mental i benestar: enquestes realitzades directament als participants, recopilades de manera digital. Algunes variables: `mentalhealth_survey`, `bienestar`, `estres`, etc.
     -  Dades cognitives: s'ha realitat l'Stroop test, un experiment cognitiu dissenyat per mesurar la capacidad de control inhibitorio y atención selectiva. Algunes variables: `stroop_test`, `response_duration`, `performance`, etc.
     -  Factos demogràfics i estil de vida: autoreport dels participants mitjançant els mateixos qüestionaris que per evaluar la salut mental. Algunes variables: `age_yrs`, `gender`, `education`, etc.
     -  Informació relacionada amb la COVID-19: enquestes específiques realitzades per evaluar l'impacte de la pandemia. Per tant, d'igual forma que per evaluar els indicadors de salut mental. Algunes variables: `covid_work`, `covid_mood`, `covid_sleep`, etc.
     -  En resum, les fonts són enquestes online, serveis meteorològics i estacions de monitoreig ambiental.
    
  7. **Existeixen relacions a priori que siguin evidents entre la contaminació i la salut mental?**
     Sí, s’observen diverses relacions significatives entre contaminació i salut mental:
     - Estrès i NO₂ (no2bcn_24h): Existeix una correlació moderada positiva, suggerint que majors nivells de NO₂ estan associats amb un increment de l’estrès.
     - Benestar i NO₂ (no2bcn_24h): La correlació és negativa, indicant que nivells més alts de contaminació poden reduir la sensació de benestar.
     - Energia i NO₂ (no2bcn_24h): També es detecta una correlació negativa, amb nivells més alts de NO₂ associats amb una reducció d’energia.
  
  9. **Hi ha variables que estiguin directament correlacionades?**
    Sí, s’identifiquen correlacions directes entre algunes variables:
     - Soroll i estrès: Les variables relacionades amb el soroll ambiental (sec_noise55_day i sec_noise65_day) mostren correlacions positives amb l’estrès
     - Hores de son i estrès: Hi ha una correlació negativa entre sueno (hores de son) i estres, la qual cosa indica que menys hores de son estan associades amb un augment de l’estrès.
     - Contaminació i benestar: Les correlacions negatives entre NO₂ i bienestar reflecteixen que majors nivells de contaminació impacten negativament en el benestar.
     - Variables de contaminació: Existeix una correlació gairebé perfecta entre no2bcn_24h i no2gps_24h, suggerint que aquestes són pràcticament redundants. També hi ha una correlació moderada entre no2bcn_24h i pm25bcn, ja que ambdós són contaminants atmosfèrics.

  10. **Les variables de salut mental estan equilibrades? Hi ha molts més malalts que sans? En cas que sí, hem de tenir alguna cosa en compte en crear?**
      Hi ha un desbalanç significatiu en les dades, amb només 13 malalts enfront de 3335 sans. A més, la incoherència entre categories ('Yes/No' versus valors numèrics de l'1 al 10) dificulta la comparació. Aquest desbalanç limita la representació dels malalts, pot generar biaixos en anàlisis futures i complica la generalització de conclusions.Podem veure l'anàlisi detallat a /02_exploratory_analysis.md

  11. **Les medicions de la contaminació són suficient detallades com per analitzar barris, o només la ciutat de BCN?**
    Basat en el gràfic que hem creat(/visualitzacions/registres_per_districtes), la distribució dels registres de contaminació és desigual entre els districtes de Barcelona. Alguns districtes, com Sant Martí, l'Eixample i Gràcia, tenen un nombre significativament més alt de registres en comparació amb altres districtes com Les Corts o Nou Barris.
  12. **Hi ha variables redundants per la determinació de la salut mental?**
    Sí, segons la matriu de correlació que hem generat (/visualitzacions/matriu_correlacio) hi ha variables que podrien ser redundants en la determinació de la salut mental, segons les dades i observacions prèvies. Per exemple, les variables no2bcn_24h i no2gps_24h mostren una correlació gairebé perfecta, la qual cosa indica que són equivalents. Això suggereix que només una d’aquestes variables és necessària per a l’anàlisi, ja que l’altra no aporta informació addicional.
    En relació amb el soroll ambiental, les variables sec_noise55_day i sec_noise65_day tenen una relació directa amb l’estrès. Tot i que no són completament redundants, podrien ser combinades o simplificades en una única mètrica per evitar duplicacions i facilitar l’anàlisi.
    Finalment, dins de les variables de salut mental, com estres, bienestar i energia, es podrien observar correlacions significatives. Per exemple, menors nivells de benestar solen estar associats amb majors nivells d’estrès i menor energia. Aquestes variables, tot i ser diferents, poden indicar aspectes interconnectats de l’estat mental, i caldria analitzar-les conjuntament per evitar duplicacions.

---

## SETMANA 2: ?
### 2.1. Objectius:
***por hacer!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*

### 2.2. Preguntes a respondre:
***por hacer!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*

---

## SETMANA 3: ?
### 3.1. Objectius:
***por hacer!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*

### 3.2. Preguntes a respondre:
***por hacer!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*
