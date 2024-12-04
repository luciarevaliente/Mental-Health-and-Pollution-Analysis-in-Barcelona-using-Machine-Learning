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
    
  3. **Com s'interpreten les dades i el seus valors?**
      - En el document `02_exploratory_analysis.md`, en l'apartat de *Característiques* hi ha un resum amb les diferents dades. En el fitxer `03_data_cleaning.md`, s'explica en l'apartat *Incongruències detectades* com els tipus de les dades està explicat en la documentació del dataset, excepte les variables `stroop_test`, `Totaltime_estimated`, `Houron` i `Houroff`, que són d'un tipus diferent.

  4. **Hi ha valors atípics?**
      - Els valors atípics s'exposen en el fitxer `03_data_cleaning.md`, concretament en l'apartat *Outliers*.
      - Per identificar-los, hem fet servir boxplots. Es poden observar en el repositori: `visualizations/boxplots`.
      - Que hi hagi valors atípics no significa que aquests siguin incorrectes. En general, en el nostre dataset els outliers estan contextualitzats com a casos extraordinaris (ex. vents forts, altes concentracions de NO₂). Per tant, són dades inusuals però vàlides. Doncs, els boxplots amb els quals hem analitzats la info, mostren la variabilitat de les dades, etc. 
    
  5. **Com s'han recopilat les dades? Quin rang geogràfic i temporal hi ha?**
     - Stroop Test: para evaluar su capacidad de control inhibitorio y atención selectiva. 
  7. Existeixen relacions a priori que siguin evidents entre la contaminació i la salut mental?
  8. Hi ha variables que estiguin directament correlacionades?
  9. Les variables de salut mental estan equilibrades? Hi ha molts més malalts que sans? En cas que sí, hem de tenir alguna cosa en compte en crear?
  10. Les medicions de la contaminació són suficient detallades com per analitzar barris, o només el districte de BCN?

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
