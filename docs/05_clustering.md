# Clustering
L'objectiu d'aquest apartat és identificar grups poblacionals mitjançant tècniques de clustering. 

## Preguntes
1. Les variables són numèriques, categòriques o ambdues? Necessitem transformar alguna variable?
2. Les variables tenen escales molt diferents? Necessitem estandaritzar o normalitzar les dades?
3. Totes les variables del dataset són rellevants per l'identificació de grups poblacionals? Hauria de fer servir la tècnica PCA per reduir variables? O com?
4. Quin tipus de grups esperem identificar? Les dades són adequades per K-Means o hauria de considerar un altre algoritme de cluster?
5. Com validarem si el clustering és efectiu?
6. Com podem interpretar els grups per identificar característiques comuns que definen cada grup poblacional?

---

## Pregunta 1: tipus de dades
En el nostre dataset hi ha: 
1. **Variables numèriques:** 
        - Ordinales: `bienestar`, `sueno`, `estres`, etc. --> *no cal escalar si el rang es petit i les destàncies són iguals. es pot escalar si el model utilitza distànies* --> *Si tu modelo se basa en distancias, debes escalar todas las variables numéricas (tanto ordinales como continuas) para asegurarte de que el modelo no se vea sesgado por las diferencias de rango entre las variables. Esto permite que todas las variables tengan una influencia equitativa en el cálculo de la distancia y mejora la precisión del modelo.*
        - Continues: `z_performance`, `no2bcn_24h`, `maxwindspeed_12h`, etc. --> *normalització o estandarització segons la distribució o escala*

2. **Variables categòriques:** --> *No se escalan porque el modelo interpreta que son etiquetas o son números binarios.*
        - Nominals: `smoke`,  `gender`, `district`, etc. Codifiquem amb **OneHotEncoder** ja que no tenen un ordre. Aquest crea una columna binària per cada categoria de la variable nominal.
        - Ordinals: `education`, `covid_work`, `covid_mood`, etc. Codifiquem amb **LabelEncoding** per respectar l'ordre de les categories. Aquest assigna un número a cada categoria.

## Pregunta 2: Escalat de les dades
Els algoritmes de clustering solen basar-se en les distàncies pel que si les nostres variables tenen escales molt diferents (p.e. `maxwindspeed_12h`, `z_performance`, etc.) algunes variables dominaran el procés, concretament les de rang major (major distància). És per aquest motiu és important les variables abans de fer clustering, perquè totes tinguin un imacte similar. A més, molts algoritmes com KMeans o SVM donen per fet que les dades són escalades i, si no és així, afectarà el resultat de l'algoritme.

### Mètode
Recordem que **escalar** les dades significa transformar-les perquè tinguin un rang comú. Doncs, és vital quan les variables tenen diferents magnituds. Algunes tècniques comuns són:
    1. Estandarització: transforma les dades perquè tinguin una mitja de 0 i desv. estàndard d'1. Se sol utilitzar quan les dades segueixen una distribució normal i les variables tenen diferents unitats o escales. És molt útil per algoritmes com KMeans o SVM i no afecta la distribució de les dades. 
    2. Normalització (Min-Max Scaling): Transforma les dades perquè estiguin dins d'un rang específic, generalment entre 0 i 1. S’utilitza quan les dades no segueixen una distribució normal o quan cal mantenir les relacions entre variables dins d’un rang uniforme. És útil per algoritmes com les xarxes neuronals i el KNN, i assegura que totes les variables tinguin el mateix pes.
    3. Escalatge Robust (Robust Scaling): Utilitza la mediana i el rang interquartílic (IQR) per escalar les dades, fent-lo menys sensible als valors atípics. S’utilitza quan les dades contenen outliers que podrien distorsionar l’escalat. És adequat per algoritmes que no volen que els outliers tinguin un gran impacte en el model, com K-means en presència de dades extremes.
    4. Escalatge per Quantils (Quantile Transformation): Transforma les dades perquè segueixin una distribució uniforme o normal. És útil quan les dades no segueixen una distribució normal i es volen ajustar per millorar el rendiment de certs algoritmes. També ajuda a suavitzar les dades extremadament disperses i a millorar el comportament dels models, especialment per models que assumeixen distribucions normals.

Gràcies a l'anàlisi previ del script `02_exploratory_data.py`, hem pogut observar en `visualizations/violin_plots` com la majoria de les dades no segueixen una distribució normal. Tot i això, hem hagut de realitzar un test de Shapiro per corroborar el que suposàvem. Efectivament, després dels tests, verifiquem que les dades numèriques a escalar no són normals. 

Doncs, per això i ja que volem que es preservi l'ordre lògic de les dades, farem servir **normalizació (Min-Max Scaling)**. D'aquesta manera, tots els valors tindran un rang homogeni i no hi haurà valors extrems.

Cal destacar que només escalarem les dades numèriques, ja que les dades categòriques s'han de codificar (les ordinals ja tenen un sentit per al clúster i les binàries ja estan normalitzades).

### Metodologia
Com el nostre dataset té una combinació de dades categòriques codificades i numèriques (ordinals i contínues), l'escalat ha de garantitzar:
        - Que les dades categòriques codificades no dominin les variables numèriques
        - Que les dades numèriques tinguin unes magnituds comparables, per evitar que les de major rang predominin.

En aquest cas, farem servir un **pipeline**. Aquesta és una eina de scikit-learn que ens permet encadenar diferents passos de preprocesament i modelatge de manera seqüencial i organitzada. Això és molt útil, especialment si tens diferents tipus de dades (numèriques, categòriques, etc.), perquè pots aplicar transformacions específiques a cada tipus de dada. 

Finalment, el codi que conté el pipeline que codifica i escala les dades és `scripts/preprocessament.py`.

## Pregunta 3: Sel·lecció de variables rellevants

## Pregunta 4: El·lecció de l'algoritme de clustering

## Pregunta 5: Validació del clustering

## Pregunta 6: Interpretació dels resultats



Només cal escalar les variables numèriques, no les codificades. El motiu és perquè les numèriques tenen valors continus que poden tenir diferents rangs o unitats (edat, km^2, etc.). Escalar aquest tipus de variables ajuda a que totes tinguin el mateix pes durant el procés de clustering, especialment quan s'utilitzen algoritmes que depenen de les distàncies entre punts.

Doncs, no té sentit escalar variables nominals codificades perquè aquestes són binàries. Per tant, escalar-les no té sentit perquè no tenen un ordre que s'hagi de normalitzar. D'altra banda, **hay que decidir si las variables estres, sueno, etc son categóricas o numéricas!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!**
