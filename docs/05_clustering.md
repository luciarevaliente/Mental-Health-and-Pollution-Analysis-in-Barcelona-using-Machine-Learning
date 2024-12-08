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
Actualment, el nostre conjunt de dades té un nombre relativament elevat de característiques (123), en comparació amb el nombre d'instàncies (3348). Tot i que no existeix una regla fixa, tenir un gran nombre de característiques pot comportar alguns reptes:
        - **Problema de la "maledicció de la dimensionalitat"**: A mesura que augmenta el nombre de característiques, els espais d'alta dimensionalitat poden esdevenir molt dispersos. Això pot dificultar la identificació de clústers ben definits, ja que les distàncies entre instàncies en espais de molta dimensió són generalment més grans.
        - **Redundància d'informació:** Si tens característiques molt correlacionades, podries estar incloent informació redundants que pot afectar el rendiment del clustering. 
        - **Rendiment computacional:** Clustering amb un gran nombre de variables pot ser costós computacionalment. Dependrà de l'algorisme que fem servir, però algorismes com K-means poden veure's afectats per la complexitat computacional quan hi ha moltes característiques.

Doncs, abans de realitzar un procés de clustering (agrupament), és fonamental assegurar-se que les variables que s’utilitzen siguin rellevants per identificar grups o segments dins de la població. Alguns motius per fer-ho són:
        - **Evitar overfitting:** quan tenim massa variables, el model pot començar a "aprofitar-se" de petites variacions o sorolls en les dades que no són rellevants, però que influeixen de manera significativa en el resulta.
        - **Millor interpretació:** amb un conjunt de dades amb moltes variables, pot ser molt difícil interpretar els resultats del model. Reduir les variables a les més rellevants ajuda a simplificar els models, facilitant l'anàlisi i la comprensió del que està passant. Això és especialment útil quan es volen obtenir conclusions pràctiques sobre els grups que es detecten en el clustering, que és just el que volem fer.
        - **Millorar l'eficència computacional:** com hem explicat abans.
        - **Reducció de la redundància:** com hem dit, si dues variables expliquen el mateix fenomen, no aportaran més informació útil al model i, per tant, podrien ser eliminades sense perdre valor. La reducció de dimensions ajuda a eliminar aquesta redundància, millorant la qualitat del model.

Cal destacar que hem d'escalar i codificar abans de reduir la dimensionalitat del dataset ja que la majoria dels mètodes depenen de la variabilitat de les dades. Per tant, per no fer una mala representació de les dades, codifiquem i escalem. 

Hem escollit fer-ho amb PCA perquè **por hacer!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!**


### Correlació entre variables
Com en el dataset tenim moltes característiques, exactament 123, i no tenim una intuició clara sobre quines són les més importants, aplicarem el PCA directament i observarem els resultats.
**por hacer!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!**

### Reducció de dimensions
**por hacer!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!**

## Pregunta 4: El·lecció de l'algoritme de clustering

## Pregunta 5: Validació del clustering

## Pregunta 6: Interpretació dels resultats

