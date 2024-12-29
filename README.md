# **Salut Mental i Contaminació a Barcelona: Un Estudi Basat en Machine Learning**
## **Descripció del Projecte** 📄
Aquest projecte analitza la relació entre la salut mental i la contaminació ambiental a la ciutat de Barcelona utilitzant tècniques de Machine Learning. Amb un dataset complet que inclou factors relacionats amb el benestar, la qualitat de l'aire i altres variables contextuals, explorem patrons, prediccions i agrupacions que poden contribuir a comprendre millor aquests factors.

---

## **Objectius del Projecte** 🎯
1. Predir indicadors de salut mental utilitzant models de regressió, concretament en l'estrès.
2. Observar si les característiques més importants sobre salut mental formen clústers. És a dir, si hi ha una clara segmentació en funció de les variables influents en l'estrès.
3. Desenvolupar perfils predictius per caracteritzar l'estrès a partir de l'anàlisi de les variables influents i la segmentació en clústers.

---

## **Contingut del Repositori** 📁

```plaintext
📦 Mental-Health-and-Pollution-Analysis-in-Barcelona-using-Machine-Learning
 ┣ 📂 data
 ┃ ┣ 📜 CitieSHealth_BCN_DATA_PanelStudy_20220414.csv (dataset principal)
 ┃ ┣ 📜 dataset.pkl (dataset emprat en els scripts)
 ┃ ┣ 📜 cleaned_dataset.pkl (dataset natejat emprat en els scripts)
 ┣ 📂 docs
 ┃ ┣ 📜 00_planificacio_setmanal_objectius_i_preguntes_clau.md (document on es planteja la cronologia del projecte, objectius i preguntes clau)
 ┃ ┣ 📜 02_exploratory_analysis.md (conté l'explicació del procediment i l'anàlisi dels resultats)
 ┃ ┣ 📜 03_data_cleaning.md (conté l'explicació del procediment i de la neteja, normalització, escalat i codificació de les dades)
 ┣ 📂 scripts
 ┃ ┣ 📜 00_codi_inspiracio.py (codi des del que partim) 
 ┃ ┣ 📜 01_load_data.py (carrega les dades i les guarda en format pickle)
 ┃ ┣ 📜 02_exploratory_analysis.py (processa i analitza les dades)
 ┃ ┣ 📜 03_data_cleaning.py (neteja i preprocesament de dades)
 ┣ 📂 visualizations
 ┃ ┣ 📂 boxplots/ (gràfiques per analitzar outliers)
 ┃ ┣ 📂 analisi_correlacio/ (gràfiques per analitzar la correlació entre variables)
 ┃ ┣ 📂 normalització_dades_categòriques/ (variables categòriques a normalitzar)
 ┣ 📂 results
 ┃ ┣ 📂 visuals/ (gràfiques i visualitzacions generades)
 ┃ ┣ 📂 metrics/ (resultats dels models)
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
   - Formats treballats: [CSV → Pickle].  

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
     - Mètriques aplicades: [MAE, RMSE, R²].  
   - **Clustering**:  
     - Visualització i validació: [t-SNE] per a la reducció de dimensionalitat i anàlisi de separació de grups.  

6. **Creació de perfils d’estrès**:  
   - A partir dels resultats de regressió i clustering, es defineixen perfils segons la distribució de variables i nivells d’estrès associats als clústers.  

 
---

## **Estructura del Codi** 🛠️
- **`scripts/`**: Conté el codi principal per al processament de dades, l'anàlisi exploratòria i l'entrenament de models.
- **`visualizations/`**: Arxius generats durant l'execució del projecte, incloent visualitzacions i mètriques.
- **`results/`**: Resultats del model: visualitzacions i mètriques.
- **`requirements.txt`**: Llista de dependències necessàries per a executar el projecte. 

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
git clone https://github.com/DCC-UAB/ACproject-19-grup.git

# Navegar al directori del projecte
cd ACproject-19-grup

# Instal·lar les dependències
pip install -r requirements.txt
```

---

## **Resultats i Conclusions** 📈
### **Resultats preliminars**
por hacer!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

### **Conclusió**
por hacer!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

---

## **Properes Etapes** 🚀
- Millorar els models actuals amb tècniques més avançades (ex. XGBoost, PCA).
- Incorporar dades de més anys per analitzar tendències temporals.
- Automatitzar el procés per utilitzar-lo en temps real.
---

## **Crèdits**
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
