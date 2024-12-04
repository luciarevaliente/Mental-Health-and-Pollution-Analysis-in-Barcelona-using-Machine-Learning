# Comprensió de les dades
En aquest arxiu, farem un anàlisi general del dataset i respondrem les preguntes de la **setmana 1**. Comprovarem que inclou tant dades sobre la contaminació com dades sobre salut mental. A més, realitzarem una exploració inicial per entendre el conjunt de dades (tipus de variables, missing values, outliers, etc.).
Les comandes executades per obtenir la informació exposada es troba en l'arxiu `csv_to_dataset.py`.

## Característiques:
El dataset té 3348 files (instàncies) i 95 columnes (característiques). Les dades són numèriques, categòriques i temporals. 

1. **Dades temporals:**
    year, month, day, hour, dayoftheweek: Informació útil per analitzar variacions temporals en la contaminació o en la salut mental.
2. **Indicadors de salut mental:**
    occurrence_mental, bienestar, energia, estres, sueno: Dades reportades pels participants que poden servir com a variables dependents per predir l’impacte de la contaminació.
3. **Contaminació ambiental:**
    no2bcn_24h, no2bcn_12h, pm25bcn: Concentracions de contaminants atmosfèrics.
    BCμg: Black Carbon, rellevant per a la salut respiratòria i mental.
    sec_noise55_day, sec_noise65_day: Exposició al soroll.
    hours_greenblue_day: Temps d'exposició a espais verds/blaus.
4. **Dades demogràfiques i personals:**
    age_yrs, gender, education, district: Context de l’individu per estudiar efectes diferenciats segons la població.
5. **Factors relacionats amb la COVID-19:**
    covid_work, covid_mood, covid_sleep: Canvis de comportament durant la pandèmia.
6. **Condicions meteorològiques:**
    tmean_24h, humi_24h, pressure_24h, precip_24h: Factors climàtics que poden influir en la salut mental.
7. **Exposició espacial i temporal:**
    min_gps, hour_gps, access_greenbluespaces_300mbuff: Dades de localització i accés a espais verds/blaus.

## Valors null:
Hi ha 3348 valors null en tot el dataset. Observem que es distribueixen de manera dispersa i no afecta a una columna en concret. És a dir, no hi ha cap columna amb tot de valors null, per tant, d'entrada no és pot eliminar cap columna sencera.

Tot i això, hem analitzat la distribució de valors null per columna. Observem que les distribucions amb <5% són columnes amb pocs nulls on la pèrdua d'informació és petita. Les columnes amb <5% i >10%, la proporció no és massa alta i encara és pot gestionar de la mateixa manera que els de <5%: substituint per els valors de les columnes null per la mitjana (var. numèriques) o per la moda (var. categòriques). En canvi, quan la proporció és >10%, els valors null són alts i pot haver-hi una mancança d'informació. Després de plantejar-nos l'eliminació d'aquestes característiques "poc valioses" a priori, decidim mantenir-les fins analitzar l'impacte de cada una en el model.
1. **Característiques amb <5% valors null:** ['ID_Zenodo', 'date_all', 'year', 'month', 'day', 'dayoftheweek', 'hour', 'mentalhealth_survey', 'occurrence_mental', 'bienestar', 'energia', 'sueno', 'horasfuera', 'actividadfisica', 'ordenador', 'dieta', 'alcohol', 'drogas', 'bebida', 'enfermo', 'otrofactor', 'stroop_test', 'no2bcn_24h', 'no2bcn_12h', 'no2bcn_12h_x30', 'no2bcn_24h_x30', 'min_gps', 'hour_gps', 'pm25bcn', 'tmean_24h', 'tmean_12h', 'humi_24h', 'humi_12h', 'pressure_24h', 'pressure_12h', 'precip_24h', 'precip_12h', 'precip_12h_binary', 'precip_24h_binary', 'maxwindspeed_24h', 'maxwindspeed_12h', 'gender', 'district', 'covid_work'].
2. **Característiques amb >5% i <10% valors null:** ['estres', 'occurrence_stroop', 'mean_incongruent', 'correct', 'response_duration_ms', 'performance', 'mean_congruent', 'inhib_control', 'z_performance', 'z_mean_incongruent', 'z_inhib_control', 'no2gps_24h', 'no2gps_12h', 'no2gps_12h_x30', 'no2gps_24h_x30', 'BCμg', 'noise_total_LDEN_55', 'access_greenbluespaces_300mbuff', 'µgm3', 'incidence_cat', 'start_day', 'start_month', 'start_year', 'start_hour', 'end_day', 'end_month', 'end_year', 'end_hour', 'Totaltime', 'Totaltime_estimated', 'Houron', 'Houroff', 'age_yrs', 'yearbirth', 'education', 'covid_mood', 'covid_sleep', 'covid_espacios', 'covid_aire', 'covid_motor', 'covid_electric', 'covid_bikewalk', 'covid_public_trans'].
3. **Característiques amb >10% valors null:**  ['sec_noise55_day', 'sec_noise65_day', 'sec_greenblue_day', 'hours_noise_55_day', 'hours_noise_65_day', 'hours_greenblue_day', 'smoke', 'psycho'].

## Outliers: 
Els outliers (valors atípics) són dades que es troben molt lluny de la resta de valors en un conjunt de dades. Aquests poden ser resultats d’errors de mesura, errors de registre, anomalies reals, o simplement punts inusuals en el conjunt de dades. Hi ha diverses tècniques per analitzar els outliers d'un dataset. Es poden identificar amb tècniques visuals (boxplots, scatterplots) o estadístiques (IQR, desviació estàndard). Per últim, el tractament dels outliers depèn del context: es poden eliminar, transformar o analitzar com a casos especials.

En el nostre projecte, farem servir la **tècnica visual boxplot**. L’ús de **boxplots** és ideal perquè permet una visualització clara i ràpida dels **outliers** i de la distribució de les dades. Aquesta tècnica destaca els valors atípics d’una manera intuïtiva gràcies als **bigotis** i als punts fora del rang esperat, sense necessitat de càlculs complexos. A més, facilita la comparació entre múltiples variables numèriques de manera simultània. Tot i que hi ha tècniques alternatives (IQR, desviació estàndard), els boxplots són més comprensibles visualment i eficients per a una anàlisi inicial.

Si cerquem l'apartat d'outliers en el codi `csv_to_dataset.py`, observem que hem generat els diferents gràfics per a les variables numèriques que poden tenir valors anòmals. A continuació, s'explica les observacions realitzades a partir d'aquests:
1. `age_yrs`: les enquestes s'han realitzat a persones d'entre 18 i 76 anys. La mitjana d'edat és 37.82 anys.
2. `BCµg`: el valor mitjà és de 0.9478µg. Es consideren 3 outliers, els quals >2µg. Doncs, la contaminació sol prendre valors baixos.
3. `bienestar`: la mesura de benestar comprèn el rang [0, 10]. El valor mitjà és de 7.22. Es consideren 3 outliers, els quals <3. Doncs, el benestar és bastant alt en general.
4. `µgm3`: el valor mitjà és 28.15 µg/m³. Existeixen diversos valors atípics (<10 y >50), indicant variabilitat en els nivells de contaminació.
5. `correct`: Els valors correctes estan entre 9 i 11 amb una mitjana de 10.47. Es consideren outliers els valors inferiors a 8 i superiors a 12.
6. `date_all` : Les dates registrades varien dins del rang [22,200, 22,250], amb una mitjana de 2 i 230.48. Es detecten outliers en dates posteriors a 22 i 300, que poden correspondre a registres anòmals.
7. `day`: Els dies registrats comprenen el rang [1, 31]. La mitjana és de 15.45 i la mediana de 15.00. No es detecten outliers en aquesta variable.
8. `dayoftheweek`: Els valors comprenen el rang [0, 6], representant els dies de la setmana. La mitjana és de 3.21. No hi ha outliers.
9. `end_day`: Els dies finals comprenen el rang [1, 31], amb una mitjana de 15.58. No es detecten valors atípics en aquesta variable.
10. `end_hour`: Les hores finals varien entre les 10 i les 20 hores, amb una mitjana de 14.82 hores. No es detecten outliers.
11. `end_month`: Els valors mes estan concentrats en els mesos finals de l'any (9-12). La mediana és 10.00 i la mitjana és 9.8. S'han detectat alguns outliers en mesos anteriors al 5, possiblement errors o dades menys representatives.
12. `end_year`: La majoria dels registres corresponen a l'any 2020 amb una mediana i mitjana de 2020.0. Hi ha un outlier a l'any 2021, que podria indicar un registre anòmal.
13. `inhib_control`: Els valors principals estan propers a 0, amb una mediana de 0.05 i una mitjana de -0.12. S'han detectat molts outliers a bandes negatives (< -2000) i positives (> 2000), indicant variabilitat alta en aquest indicador.
14. `maxwindspeed_12h`: Els valors típics són baixos (0-5 m/s), amb una mediana de 1.2 m/s i una mitjana de 2.0 m/s. Els outliers es troben per sobre de 10 m/s, representant vents atípics o extrems.
15. `occurrence_mental`: Les puntuacions es distribueixen entre 2 i 12 amb una mediana de 7.0 i una mitjana de 7.3. No s'han detectat outliers significatius.
16. `occurrence_stroop`: La distribució és similar a la variable anterior, amb una mitjana de 7.1. No hi ha valors extrems destacats.
17. `start_day`: Els dies estan distribuïts uniformement entre 1 i 30, amb una mitjana de 15.3. No s'han detectat outliers.
18. `start_hour`: Els valors centrals són entre 8 i 12, amb una mediana de 10.0 i una mitjana de 9.8. Els outliers es troben abans de les 5 i després de les 20 hores, probablement per activitats irregulars.
19. `z_mean_incongruent`: Els valors centrals estan propers a 0, amb una mitjana de 0.02. Els outliers van des de -2.0 fins a més de 10.0, indicant dispersió important.
20. `z_performance`: La mediana és 0.00 i la mitjana és 0.05. S'han detectat outliers a ambdues bandes (< -3 i > 3), però la major part de les dades es troben en un rang ajustat.
21. `energia`: Els valors estan principalment entre 6 i 8, amb una mediana de 7.5 i una mitjana de 7.3. Hi ha 3 outliers amb valors inferiors a 3, representant una baixa energia.
22. `estres`: Els nivells d'estrès es distribueixen entre 2 i 8, amb una mediana de 5.0 i una mitjana de 5.1. No es detecten outliers destacats.
23. `horasfuera`: Les hores fora varien entre 0 i 10, amb una mitjana de 4.8. Hi ha diversos outliers amb valors superiors a 15 hores, sent un valor màxim de 35 hores.
24. `hour`: Els registres d'hores es concentren entre 15 i 21, amb una mediana de 19.0 i una mitjana de 18.7. S'han detectat outliers abans de les 10 hores, indicant activitats poc freqüents en aquest horari.
25. `hour_gps`: Els valors es distribueixen uniformement entre 0 i 24 hores, amb una mediana de 12.0 i una mitjana de 12.1. No hi ha valors extrems destacats.
26. `hours_greenblue_day`: Les hores en espais verds i blaus són generalment inferiors a 5, amb una mediana de 1.0 i una mitjana de 2.3. S'han detectat diversos outliers superiors a 20 hores.
27. `hours_noise_55_day`: La majoria dels valors es troben entre 0 i 5 hores, amb una mediana de 2.0 i una mitjana de 2.7. S'han identificat outliers amb més de 15 hores d'exposició al soroll de 55 dB.
28. `hours_noise_65_day`: La distribució és similar a `hours_noise_55_day`, amb una mitjana de 1.9. Els outliers superen les 15 hores d'exposició al soroll de 65 dB.
29. `humi_12h`: Els valors d'humitat relativa oscil·len entre 50% i 80%, amb una mitjana de 66.3%. No es detecten outliers significatius.
30. `humi_24h`: Les dades d'humitat de 24 hores tenen un comportament similar a `humi_12h`, amb una mitjana de 66.8%. Els valors estan dins del rang esperat.
31 . `maxwindspeed_24h`: la velocitat màxima del vent en 24 hores comprèn valors entre 0 i 25 m/s. El valor mitjà és de 2.34 m/s. Es consideren outliers els valors superiors a 10 m/s. Doncs, la velocitat del vent sol ser baixa la major part del temps.
32. `mean_congruent`: el temps mitjà de resposta congruent varia entre 0 i 8000 ms. El valor mitjà és de 1312.45 ms. Es consideren outliers els valors >3000 ms. Doncs, la majoria de respostes congruents es donen ràpidament.
33. `mean_incongruent`: el temps mitjà de resposta incongruent comprèn el rang de 0 a 6000 ms. El valor mitjà és de 1452.89 ms. Es consideren outliers els valors superiors a 4000 ms. Doncs, les respostes incongruents solen requerir més temps que les congruents.
34. `min_gps`: el valor mínim de GPS (distància o temps segons la variable) oscil·la entre 0 i 1400 unitats. El valor mitjà és de 623.11 unitats. No s’identifiquen outliers evidents en aquesta variable.
35. `month`: la distribució mensual indica que les observacions es concentren sobretot a la tardor (setembre, octubre, novembre). El valor mitjà és de 9.12 (corresponent a setembre). Es consideren outliers els mesos de gener, febrer i març.
36. `no2bcn_12h`: la concentració de NO2 en 12 hores varia entre 10 i 80 µg/m³. El valor mitjà és de 34.67 µg/m³. Es consideren outliers els valors >60 µg/m³, que corresponen a episodis d’alta contaminació.
37. `no2bcn_24h` la concentració de NO2 en 24 hores té un rang similar, amb un valor mitjà de 33.45 µg/m³. Els outliers també es consideren per sobre dels 60 µg/m³.
38. `no2gps_12h`: el valor mitjà de NO2 mesurat per GPS en 12 hores és de 38.12 µg/m³. Es detecten diversos outliers >70 µg/m³, que podrien indicar zones amb alta densitat de trànsit o fonts de contaminació puntuals.
39. `no2gps_24h`: la concentració mitjana en 24 hores és similar a la de 12 hores, amb un valor mitjà de 37.78 µg/m³. Els outliers també superen els 70 µg/m³.
40. `noise_total_LDEN_55`: la mesura de soroll total (LDEN >55 dB) varia entre 0 i 1 (indicador binari). El valor mitjà és de 0.78, el que implica que en la majoria de casos es superen els 55 dB. Només s'identifiquen pocs casos amb valors propers a 0.


## Gestió dels valors null:
**esto va en preprocessament de les dades!!!!!!!!!!!!!!!!!!!!**
1. **Eliminació de les files / columnes amb valors null:** és una manera ràpida i senzilla, només ens asseguraria treballar amb dades completes, ja que no afegim informació artificial. Però, com els valors null estan dispersos entre les files i columnes, això podria reduir significativament els registres disponibles per entrenar el model, afectant a la capacitat de generalitzar. D'altra banda, si els registres amb valors null tenen característiques diferents als complets, eliminar-los significaria introduir un biaix, ja que el model no representaria correctament les dades. Doncs, no farem servir aquesta tècnica per no disminuir la mostra del model. A més, perquè no seria correcte, ja que es recomanable eliminar registres quan la proporció de files amb null és <5% o quan es perd un 10% de la mostra original. En el nostre cas, la meitat dels registres (52.38%) contenen almenys un valor null, per tant, no seria adhient. Pel que respecta a les columnes, no sabem si les característiques són significatives, per tant, preferim mantenir-les per averiguar la seva importància en el model.

2. Imputació simple (mitjana/moda): **por hacer!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!**

3. Models predictius per imputar els nulls: **por hacer!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!**

4. Crear variables indicadores: **por hacer!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!**

## Gestió dels outliers:
**por hacer????????????????????**

## Proporció dels registres de mentalhealth 
per veure si Les variables de salut mental estan equilibrades? Hi ha molts més malalts que sans? En cas que sí, hem de tenir alguna cosa en compte en crear? --> **por hacer!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!**

# Preguntes:
***por hacer!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*
