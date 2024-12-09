# Preprocessament de les dades

## Propòsit del Codi
Aquest script té com a objectiu processar un **dataset net** i preparar-lo per al seu ús en **l'entrenament de models d'aprenentatge automàtic**. S'ocupa de **codificar les variables categòriques**, **escalar les variables numèriques**, i realitzar certes modificacions en les variables binàries per tal de fer que el conjunt de dades sigui adequat per als models que requereixen dades numèriques estandarditzades i codificades.

En aquest procés, s'apliquen diferents tècniques de codificació i escalat que milloren la qualitat i el rendiment dels models d'aprenentatge automàtic. El flux de treball va des de la **codificació** de les variables categòriques fins a l'**escalat** de les variables numèriques, creant un dataset que estarà llest per alimentar a un model predictiu.

## Flux del Procés
1. **Carregar el Dataset Neta**: 
    - Es carrega l'arxiu `data/cleaned_dataset.pkl` al script, que ja ha passat per un procés de neteja inicial.
   
2. **Codificació de Columnes Categòriques**:
    - **Columnes Ordinals**: Les variables amb categories ordenades (com `education`) es codifiquen utilitzant el **`OrdinalEncoder`**. Aquest codificador assigna un valor numèric a cada categoria seguint un ordre definit prèviament (per exemple, de menor a major en "educació").
    - **Columnes Nominals**: Per a les variables categòriques sense un ordre intrínsec (com `districte`), s'utilitza **`OneHotEncoder`**. Aquest codificador crea una columna per a cada categoria i assigna un valor binari (0 o 1) segons la presència o absència de la categoria en una fila.
    - **Columnes Binàries**: Per a les columnes amb valors binaris (`yes/no` o `home/dona`), es mapegen a valors `1` i `-1`, respectivament. 

3. **Canvi de Codificació de [0, 1] a [-1, 1]**:
    - **Motiu**: Es realitza un canvi d'escala en les variables binàries codificades. Tot i que les variables binàries normalment es codifiquen en un rang de [0,1], en aquest script es decideix codificar-les en un rang de [-1,1]. A més, alguns algoritmes, solen funcionar millor quan les entrades estan centrades al voltant de 0, el que permet una convergència més ràpida i estable en els models.

4. **Escalat de les Columnes Numèriques**:
    - **Motiu**: El canvi de [0,1] a [-1,1] es fa per evitar la necessitat d'escalat posterior. Quan utilitzem StandardScaler més endavant, aquest mètode escala les dades perquè tinguin una mitjana de 0 i una desviació estàndard de 1, cosa que ja es compleix si les dades estan dins del rang [-1, 1]. Així, les columnes binàries ja estaran centrades al voltant de 0, eliminant la necessitat de transformacions addicionals després de l'escalat.

    - El **StandardScaler** assegura que els valors de les característiques estiguin distribuïts de manera que el 68% de les dades estiguin dins del rang de [-1, 1]. Aquesta transformació és important per garantir que cap atribut tingui més pes només per la seva escala original.

    - A més, alguns models d'aprenentatge automàtic, funcionen millor amb dades centrades al voltant de 0 i amb distribucions simètriques. Per tant, aquest canvi de rang facilita l'optimització i la convergència del model.

5. **Variable Binària Codificada per Separat**:
    - Algunes variables binàries, com `precip_12h_binary` i `precip_24h_binary`, han estat processades per separat a causa de la seva naturalesa específica. Aquestes columnes, que contenen valors `0` o `1`, es gestionen de manera diferent per mantenir la integritat de la codificació binària.

6. **Visualització i Emmagatzematge**:
    - Després de realitzar totes les codificacions i escalats, el dataset processat es guarda en dos formats:
      - **`codif_dataset.pkl`**: Es guarda com un arxiu `.pkl` per poder ser carregat directament pel model.
      - **`codif_dataset.xlsx`**: Es guarda com un arxiu d'Excel per permetre la visualització i l'anàlisi més accessible de les dades processades.
    - Finalment, el dataset amb les columnes escalades es guarda en l'arxiu `scaled_dataset.pkl` i `scaled_dataset.xlsx` per al seu posterior ús.

## Detalls Tècnics

### Codificació de Columnes Categòriques

- **Columnes Ordinals**: Es codifiquen mitjançant el `OrdinalEncoder`, que assigna nombres enters d'acord amb un ordre especificat prèviament al diccionari `ordinal_columns`. Això és essencial per a les variables que tenen un ordre natural entre les seves categories.
  
- **Columnes Nominals**: Per a les variables nominales (sense un ordre intrínsec), s'utilitza el `OneHotEncoder`, que converteix cada valor únic d'una columna en una columna separada, assignant un valor binari (0 o 1) segons si l'observació pertany o no a aquella categoria.

### Escalat de Variables Numèriques

- **`StandardScaler`**: Aquest escalat assegura que totes les variables numèriques estiguin centrades en 0 i tinguin una desviació estàndard de 1, el que és fonamental perquè el model no es vegi influenciat per les diferències de magnitud entre les variables.

### Modificació de Columnes Binàries

- **Per què canviem de [0,1] a [-1,1]**: Aquest canvi d'escala es fa per evitar problemes als algoritmes d'aprenentatge automàtic que poden sorgir a causa de les diferències en l'escala de les característiques. Sovint, els algoritmes funcionen millor quan les variables numèriques estan centrades en zero i en un rang de valors simètric.

### Emmagatzematge en Excel

- **Motiu**: Guardar el dataset en format Excel permet una fàcil inspecció visual de les dades, la qual cosa pot ser útil per revisar i validar la codificació i l'escalat. També facilita l'anàlisi del dataset sense haver de carregar-lo en un entorn de programació.

## Conclusió

Aquest script proporciona un procés complet per preparar un dataset abans d'entrenar models d'aprenentatge automàtic. La codificació de les columnes categòriques i l'escalat de les variables numèriques són passos fonamentals per assegurar-se que els models puguin aprendre de les dades de manera eficaç. Al separar les diferents formes de codificació, aquest script assegura que cada tipus de dada es gestioni de la manera més adequada, millorant així la capacitat predictiva del model. A més, l'ús de `StandardScaler` per a les variables numèriques i la codificació en el rang [-1,1] contribueixen a una millor convergència i rendiment als algoritmes d'Aprenentatge Automàtic.