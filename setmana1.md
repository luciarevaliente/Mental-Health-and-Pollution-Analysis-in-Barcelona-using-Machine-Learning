En aquest arxiu, farem un anàlisi genearl del dataset i respondrem les preguntes de la setmana 1.
Les comandes executades per obtenir la informació exposada es troba en l'arxiu "csv_to_dataset.py".

# Anàlisi general
El dataset té 3348 files (instàncies) i 95 columnes (característiques). Les dades són numèriques, categòriques i temporals. 

**Característiques:**
1. Dades temporals:
    year, month, day, hour, dayoftheweek: Informació útil per analitzar variacions temporals en la contaminació o en la salut mental.
2. Indicadors de salut mental:
    occurrence_mental, bienestar, energia, estres, sueno: Dades reportades pels participants que poden servir com a variables dependents per predir l’impacte de la contaminació.
3. Contaminació ambiental:
    no2bcn_24h, no2bcn_12h, pm25bcn: Concentracions de contaminants atmosfèrics.
    BCμg: Black Carbon, rellevant per a la salut respiratòria i mental.
    sec_noise55_day, sec_noise65_day: Exposició al soroll.
    hours_greenblue_day: Temps d'exposició a espais verds/blaus.
4. Dades demogràfiques i personals:
    age_yrs, gender, education, district: Context de l’individu per estudiar efectes diferenciats segons la població.
5. Factors relacionats amb la COVID-19:
    covid_work, covid_mood, covid_sleep: Canvis de comportament durant la pandèmia.
6. Condicions meteorològiques:
    tmean_24h, humi_24h, pressure_24h, precip_24h: Factors climàtics que poden influir en la salut mental.
7. Exposició espacial i temporal:
    min_gps, hour_gps, access_greenbluespaces_300mbuff: Dades de localització i accés a espais verds/blaus.

**Valors null:**
Hi ha 3348 valors null en tot el dataset. Observem que es distribueixen de manera dispersa i no afecta a una columna en concret.
Tot i això, hem analitzat la distribució de valors null per columna. 