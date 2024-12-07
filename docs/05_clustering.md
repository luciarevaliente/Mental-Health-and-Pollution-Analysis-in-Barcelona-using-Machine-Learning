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
        - Ordinales: `bienestar`, `sueno`, `estres`, etc.
        - Continues: `z_performance`, `no2bcn_24h`, `maxwindspeed_12h`, etc.

2. **Variables categòriques:** 
        - Nominals: `smoke`,  `gender`, `district`, etc. --> OneHotEncoder
        - Ordinals: `education`, `covid_work`, `covid_mood`, etc. --> LabelEncoding

Les variables **categòriques ordinals** tenen un ordre, pel que les podem deixar tal qual. Les variables **categòriques nominals** les hem de codificar amb OneHotEncoder ja que no tenen un ordre. Aquest crea una columna binària per cada categoria de la variable nominal. 

Només cal escalar les variables numèriques, no les codificades. El motiu és perquè les numèriques tenen valots continus que poden tenir diferents rangs o unitats (edat, km^2, etc.). Escalar aquest tipus de variables ajuda a que totes tinguin el mateix pes durant el procés de clustering, especialment quan s'utilitzen algoritmes que depenen de les distàncies entre punts.

Doncs, no té sentit escalar variables nominals codificades perquè aquestes són binàries. Per tant, escalar-les no té sentit perquè no tenen un ordre que s'hagi de normalitzar. D'altra banda, **hay que decidir si las variables estres, sueno, etc son categóricas o numéricas!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!**
