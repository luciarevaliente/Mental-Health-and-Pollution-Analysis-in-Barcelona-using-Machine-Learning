# **Salut Mental i Contaminació a Barcelona: Un Estudi Basat en Machine Learning**
## **Descripció del Projecte** 📄
Aquest projecte té com a objectiu analitzar la relació entre la salut mental i la contaminació ambiental a la ciutat de Barcelona l'any 2020/2021 mitjançant tècniques d'aprenentatge automàtic. Utilitzant un conjunt de dades que inclou informació sobre la qualitat de l'aire, el benestar general, i altres factors contextuals com l'activitat física, l'alimentació i l'ús d'espais verds, pretenem identificar patrons i establir connexions significatives sobre els nivells d'estrès en la població.

L'anàlisi es basa en models de regressió per predir els nivells d'estrès i tècniques de clustering per identificar grups de persones amb característiques similars, amb l'objectiu de millorar la comprensió de com els factors ambientals i socials poden influir en la salut mental a la ciutat.

Amb aquest enfocament, busquem proporcionar una visió més clara dels factors que poden contribuir a l'estrès de la població, facilitant la creació d'estratègies d'intervenció i millora de la qualitat de vida en entorns urbans.

---

## Índex
1. Descripció del Projecte 📄  
2. Objectius del Projecte 🎯  
3. Contingut del Repositori 📁  
4. Dataset 📊  
5. Metodologia 🧠  
6. Resultats i Conclusions 📈  
7. Properes Etapes 🚀  
8. Dependències i Instal·lació 📦  
9. Contribucions 🤝  
10. Crèdits 📝  
11. Llicència 📜  
12. Contacte 📬 

---

## **Objectius del Projecte** 🎯
1. Predir indicadors de salut mental utilitzant models de regressió, concretament en l'estrès.
2. Observar si les característiques més importants sobre salut mental formen clústers. És a dir, si hi ha una clara segmentació en funció de les variables influents en l'estrès.
3. Desenvolupar perfils predictius per caracteritzar l'estrès a partir de l'anàlisi de les variables influents i la segmentació en clústers.
4. Realitzar propostes de millora per tal de reduir l'estrès poblacional a BCN.

## **Contingut del Repositori** 📁
```plaintext
📦 Mental-Health-and-Pollution-Analysis-in-Barcelona-using-Machine-Learning
 ┣ 📂 data
 ┃ ┣ 📜 CitieSHealth_BCN_DATA_PanelStudy_20220414.csv (dataset principal)
 ┃ ┣ 📜 dataset.pkl (dataset emprat en els scripts)
 ┃ ┣ 📜 cleaned_dataset.pkl (dataset natejat emprat en els scripts)
 ┃ ┣ 📜 processed_dataset.pkl (dataset processat)
 ┣ 📂 scripts
 ┃ ┣ 📜 main.py (menú principal del programa, únic script que s'ha d'executar)
 ┃ ┣ 📜 load_data.py (carrega les dades i les guarda en format pickle)
 ┃ ┣ 📜 exploratory_analysis.py (processa i analitza les dades)
 ┃ ┣ 📜 data_cleaning.py (neteja i preprocesament de dades)
 ┃ ┣ 📜 preprocess.py (conté la funció que preprocessa les dades, exceptuant la variable target)
 ┃ ┣ 📜 evaluation_regression.py (conté les funcions per evaluar els models regressors)
 ┃ ┣ 📜 feature_importance_regression.py (conté les funcions per trobar la importància de les variables)
 ┃ ┣ 📜 main_regression.py (realiza regressió mitjançant diversos models)
 ┃ ┣ 📜 models_clustering.py (conté la classe que inicialitza els mètodes de clústering i defineix totes les funcionalitats) 
 ┃ ┣ 📜 main_clustering.py (realitza clústering amb diversos algoritmes i característiques)
 ┣ 📂 visualizations
 ┃ ┣ 📂 boxplots/ (gràfiques per analitzar outliers)
 ┃ ┣ 📂 violinplots/ (gràfiques per analitzar distribucions)
 ┃ ┣ 📂 analisi_correlacio/ (gràfiques per analitzar la correlació entre variables)
 ┃ ┣ 📂 preprocess_visualization/ (gràfiques per visualitzar les dades processades)
 ┃ ┣ 📂 data_regression/ (excels amb les característiques més importants)
 ┃ ┣ 📂 regressio/ (visualitzacions del procés)     
 ┃ ┣ 📂 clusters/ (gifs i gràfiques per analitzar clústers formats)
 ┃ ┃ ┣ 📂 dataset/ (gràfiques per visualitzar clústers amb dataset complet)
 ┃ ┃ ┣ 📂 general_important_features/ (gràfiques per visualitzar clústers amb característiques generals dels models de regressió)
 ┃ ┃ ┣ 📂 XGBoost_important_features/ (gràfiques per visualitzar clústers amb característiques del model XGBoost)
 ┃ ┃ ┣ 📂 XGBoost_4thimportant_features/ (gràfiques per visualitzar clústers amb les 4 característiques més importants del model XGBoost)
 ┣ 📂 results
 ┃ ┣ 📜 metriques_models_regressors.png (gràfica que compara els errors dels models regressors)
 ┃ ┣ 📜 grafic_dispersio_valors_reals_vs_prediccions.png (gràfic sobre l'error en les prediccions de XGBoost, que és el millor model)
 ┃ ┣ 📜 clustering_visualization.gif (visualització de la segmentació de les dades amb el millor model de clústering GMM k=12)
 ┃ ┣ 📜 clustering_distribution.png (distribució de l'estrès en els clústers de GMM k=12)
 ┃ ┣ 📜 caraceteristiques_centroides.png (matriu amb les característiques mitjanes per clúster)
 ┃ ┣ 📜 perfil_estres_poblacional.png (taula amb l'evaluació de l'estrès de la població barcelonina 2020/2021)
 ┣ 📂 docs
 ┃ ┣ 📜 informe.docx (informe amb resultats del projecte)
 ┣ 📜 .gitignore 
 ┣ 📜 README.md (aquest fitxer)
 ┣ 📜 requirements.txt (llibreries necessàries)
 ┗ 📜 LICENSE (llicència del projecte)
```

---

## **Dataset** 📊
### Descripció del Dataset
- **Nom**: CitieSHealth_BCN_DATA_PanelStudy_20220414.csv
- **Nombre de registres**: 3348
- **Nombre de columnes**: 95
- **Principals característiques**:
  - Factors de salut mental: *benestar, estrès, energia, son...*
  - Contaminació: *NO₂, PM₂.₅, BCμg...*
  - Altres factors: *activitat física, dieta, soroll ambiental, accés a espais verds/blaus...*
  
---

## **Metodologia** 🧠 
La metodologia aplicada ens ha permès desenvolupar un enfocament complet i estructurat per assolir els nostres objectius. Començant per la importació i preprocessament de dades, hem assegurat la qualitat i consistència dels inputs per al nostre modelatge. L’anàlisi exploratòri ens ha proporcionat una comprensió profunda de les dades, identificant correlacions rellevants i patrons inicials.  

Mitjançant l’ús de tècniques avançades de regressió i clustering, hem pogut no només predir els nivells d’estrès, sinó també agrupar les dades en clústers significatius. Això ens ha permès extreure coneixements clau sobre les característiques que més influeixen en l’estrès i desenvolupar perfils que expliquen aquests resultats.  

A continuació, expliquem en detall el procés seguit en cadascuna de les etapes per il·lustrar com s’han obtingut els resultats i validar la robustesa del nostre anàlisi. Aquest treball no només garanteix una base sòlida per a la interpretació dels resultats, sinó que també proporciona una eina pràctica per identificar patrons de comportament i establir relacions entre variables. 

1. **Importació de dades**:  
   - Formats treballats: de CSV a [Pickle].  

2. **Neteja i preprocessament de dades**:  
   - **Gestió de valors nuls**: Imputació utilitzant [KNN Imputer].  
   - **Escalament de variables**: [StandardScaler] per a variables ordinals i contínues.  
   - **Codificació de variables categòriques**:  
     - [OneHotEncoder] per a variables nominals.  
     - [OrdinalEncoder] per a variables ordinals.  

3. **Anàlisi exploratòria de dades**:  
   - Estadístiques descriptives per comprendre les distribucions de les dades.  
   - Visualització de correlacions: [matriu de correlació → heatmap].  

4. **Models utilitzats**:  
   - **Regressió**:  
     - **Objectiu**: Predir el nivell d’estrès a partir de les característiques disponibles.  
     - **Models aplicats**: [RandomForest, XGBoost, GradientBoosting, SVR, Polynomial].  
     - **Optimització de paràmetres**: Mitjançant [RandomizedSearchCV, GridSearchCV].  

   - **Clustering**:  
     - **Objectiu**: Identificar clústers basats en les característiques més influents per predir l’estrès.  
     - **Models aplicats**: [K-Means, Agglomerative Clustering, Gaussian Mixture Models].  
     - **Selecció de millors paràmetres**: Utilitzant [Elbow Method, BIC].  

5. **Avaluació dels models**:  
   - **Regressió**:  
     - Mètriques aplicades: [MAE, MSE].  
   - **Clustering**:  
     - Visualització i validació: [t-SNE] per a la reducció de dimensionalitat i anàlisi de separació de grups.  

6. **Creació de perfils d’estrès**:  
   - A partir dels resultats de regressió i clustering, es defineixen perfils segons la distribució de variables i nivells d’estrès associats als clústers.  

---

## **Resultats i Conclusions** 📈
A continuació, resumim les conclusions. Per un anàlisi detallat, consultar /docs/informe.docx.
- Regressió: XGBoost (MAE 1.49).
- Clustering: GMM (k=12) en establir les característiques més importants dels regressors com a variables d’entrada.
- Perfil d'estrès poblacional: ![Perfil d'estrès poblacional](/results/perfil_estres_poblacional.png)


Les ciutats poden adoptar una varietat de mesures que aborden els factors ambientals, socials i emocionals que influeixen en l'estrès de la població. La **reducció de la contaminació**, la **millora de l'espai públic**, el **foment de la mobilitat sostenible**, i la **promoció de la salut mental** són claus per crear entorns urbans més saludables. A més, l'educació i la **cohesió social** juguen un paper fonamental en la creació de comunitats resilients a l'estrès.

Amb l'impacte de la **pandèmia de COVID-19**, la salut mental de la població urbana ha adquirit una importància renovada, ja que els confinaments, l'aïllament social i els canvis en la rutina han intensificat els nivells d'estrès. En aquest context, aquestes accions esdevindran encara més essencials per al benestar general de les ciutats en el futur. **El reforç de les polítiques públiques orientades a la salut mental**, l'augment d'espais verds i la creació d'espais comuns per a la socialització seran claus per ajudar les ciutats a recuperar-se i a millorar la qualitat de vida post-pandèmia. 

Amb aquestes accions, les ciutats poden no només millorar el benestar de la seva població, sinó també reduir els nivells d'estrès a llarg termini, ajudant a crear comunitats més fortes i saludables.


---

## **Properes Etapes** 🚀
- Millorar els models actuals amb tècniques més avançades (ex. DBSCAN , PCA).
- Evaluar i comparar les distribucions amb mètriques.
- Treballar amb dades a temps real per veure l'evolució de l'estrès i comprovar si les mesures són efectives.

---

## **Dependències i Instal·lació** 📦
### Llibreries necessàries
Aquest projecte requereix Python 3 i les següents llibreries:
- **pandas**: Per manipular dades tabulars.
- **numpy**: Per a operacions numèriques i manipulació de matrius.
- **scikit-learn**: Conté eines per a l'aprenentatge automàtic (models, preprocesament, validació, etc.).
- **matplotlib**: Per crear gràfics estàtics i visualitzacions.
- **seaborn**: Basat en `matplotlib`, però orientat a la visualització estadística.
- **imbalanced-learn**: Conté eines com `RandomOverSampler`, `SMOTE` i `RandomUnderSampler` per equilibrar classes en conjunts de dades desequilibrats.
- **xgboost**: Una biblioteca per a l'algorisme de Gradient Boosting optimitzat.
- **yellowbrick**: Proporciona visualitzacions per a l'aprenentatge automàtic, com el `KElbowVisualizer`.


### Instal·lació
```bash
# Clonar aquest repositori
git clone https://github.com/luciarevaliente/Mental-Health-and-Pollution-Analysis-in-Barcelona-using-Machine-Learning.git

# Navegar al directori del projecte
cd Mental-Health-and-Pollution-Analysis-in-Barcelona-using-Machine-Learning

# Instal·lar les dependències
pip install -r requirements.txt

# Executar programa 
python scripts/main.py
```

---

## **Contribucions** 🤝  
Les contribucions són benvingudes! En cas de voler aportar un gra de sorra, si us plau segueix els passos:
1. Fes un fork del repositori.
2. Crea una branca per a la teva nova funcionalitat (git checkout -b feature/nova-funcionalitat).
3. Fes els canvis i fes commit (git commit -am 'Afegeix nova funcionalitat').
4. Fes push a la teva branca (git push origin feature/nova-funcionalitat).
5. Obre un Pull Request.


---

## **Crèdits** 📝  
**Autors del projecte**:
- **Lucía Revaliente Torres**:  
   - Importació de dades.  
   - Anàlisis de dades.
   - Neteja de dades
   - Preprocessament.
   - Implementació dels models de clustering.
   - Creació de perfils d'estrès.  
   - Propostes de millora per reduir l'estrès poblacional.
  
- **Aránzazu Miguélez Montenegro**:  
   - Anàlisis de dades.
   - Preprocessament.
   - Generació de les visualitzacions i gràfics.
   - Desenvolupament dels models de regressió.

**Autors del dataset:** Gignac, F., Righi, V., Toran, R., Paz Errandonea, L., Ortiz, R., Mijling, B., Naranjo, A., Nieuwenhuijsen, M., Creus, J., & Basagaña, X. (2022). CitieS-Health Barcelona Panel Study Results [Data set]. Zenodo. https://doi.org/10.5281/zenodo.6503022

---

## **Llicència** 📜
Aquest projecte està llicenciat sota la **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**.

Es permet utilitzar, modificar i distribuir el codi per a usos no comercials amb la condició que es reconegui l'autor original. L'ús comercial requereix el permís express de l'autor.

Consulta el fitxer [LICENSE](LICENSE) per obtenir més informació.

---

## **Contacte** 📬
Per a més informació o col·laboracions, pots contactar amb:
- **Nom**: [Lucía Revaliente Torres]
- **LinkedIn**: [[enllaç](https://www.linkedin.com/in/lucia-revaliente-torres)]
- **Nom**: [Aránzazu Miguélez Montenegro]
- **LinkedIn**: [[enllaç](https://www.linkedin.com/in/aranzazu-miguelez)]

---

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/USx538Ll)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=17348921&assignment_repo_type=AssignmentRepo)
