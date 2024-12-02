# **Salut Mental i Contaminació a Barcelona: Un Estudi Basat en Machine Learning**

## **Descripció del Projecte** 📄
Aquest projecte analitza la relació entre la salut mental i la contaminació ambiental a la ciutat de Barcelona utilitzant tècniques de Machine Learning. Amb un dataset complet que inclou factors relacionats amb el benestar, la qualitat de l'aire i altres variables contextuals, explorem patrons, prediccions i agrupacions que poden contribuir a comprendre millor aquests factors.

---

## **Objectius del Projecte** 🎯
1. Analitzar la correlació entre factors de contaminació i salut mental.
2. Predir indicadors de salut mental (estrès, energia, son) utilitzant models de regressió.
3. Identificar grups poblacionals mitjançant tècniques de clustering basades en dades ambientals i demogràfiques.
4. Proporcionar eines i conclusions que puguin ser útils per a polítiques públiques i intervencions urbanes.

---

## **Contingut del Repositori** 📁

```plaintext
📦 Nom del projecte
 ┣ 📂 data
 ┃ ┣ 📜 CitieSHealth_BCN_DATA_PanelStudy_20220414.csv (dataset principal)
 ┣ 📂 notebooks
 ┃ ┣ 📜 01_data_cleaning.ipynb (neteja i preprocesament de dades)
 ┃ ┣ 📜 02_exploratory_analysis.ipynb (anàlisi exploratòria)
 ┃ ┣ 📜 03_regression_models.ipynb (models de regressió)
 ┃ ┣ 📜 04_clustering_analysis.ipynb (clustering i insights)
 ┣ 📂 models
 ┃ ┣ 📜 final_regression_model.pkl (model final de regressió)
 ┃ ┣ 📜 clustering_results.pkl (resultats del clustering)
 ┣ 📂 results
 ┃ ┣ 📜 visuals/ (gràfiques i visualitzacions generades)
 ┃ ┣ 📜 metrics/ (resultats dels models)
 ┣ 📜 README.md (aquest fitxer)
 ┣ 📜 requirements.txt (llibreries necessàries)
 ┗ 📜 LICENSE.md (llicència del projecte)
```

---

## **Dataset** 📊

### Descripció del Dataset
- **Nom**: CitieSHealth_BCN_DATA_PanelStudy_20220414.csv
- **Nombre de registres**: 3348
- **Nombre de columnes**: 95
- **Principals característiques**:
  - Factors de salut mental: *bienestar, estrès, energia, sueno.*
  - Contaminació: *NO₂, PM₂.₅, BCμg.*
  - Altres factors: *activitat física, dieta, soroll ambiental, accés a espais verds/blaus.*
  
### Preprocessament
- Gestió de valors nuls: [estratègia utilitzada].
- Escalament de variables: [sí/no i tècnica utilitzada].
- Codificació de variables categòriques: [sí/no i tècnica utilitzada].

---

## **Metodologia** 🧠

1. **Neteja i preprocessament de dades**:
   - Tractament de valors nuls: [estratègia específica].
   - Normalització i transformació de dades: [especificació].
   
2. **Anàlisi exploratòria**:
   - Estadístiques descriptives.
   - Visualitzacions de correlacions (ex. mapes de calor, scatterplots).

3. **Models utilitzats**:
   - **Regressió**:
     - Objectiu: Predir variables de salut mental (ex. estrès, energia).
     - Models utilitzats: [lineal, regressió logística, etc.].
   - **Clustering**:
     - Objectiu: Agrupar persones segons similituds en els factors de contaminació i salut mental.
     - Models utilitzats: K-Means, DBSCAN, etc.

4. **Avaluació dels models**:
   - Mètriques per a regressió: [MAE, RMSE, R², etc.].
   - Mètriques per a clustering: [Silhouette Score, Inertia, etc.].

---

## **Estructura del Codi** 🛠️

- **`notebooks/`**: Conté el codi principal per al processament de dades, l'anàlisi exploratòria i l'entrenament de models.
- **`models/`**: Models entrenats i exportats.
- **`results/`**: Arxius generats durant l'execució del projecte, incloent visualitzacions i mètriques.
- **`requirements.txt`**: Llista de dependències necessàries per a executar el projecte.

---

## **Dependències i Instal·lació** 📦

### Llibreries necessàries
Aquest projecte requereix Python 3.x i les següents llibreries:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter

### Instal·lació
```bash
# Clonar aquest repositori
git clone https://github.com/usuari/nom-del-repositori.git

# Navegar al directori del projecte
cd nom-del-repositori

# Instal·lar les dependències
pip install -r requirements.txt
```

---

## **Resultats i Conclusions** 📈

### **Resultats preliminars**
- Les variables més correlacionades amb la salut mental són: [variables destacades].
- Els nivells de NO₂ i PM₂.₅ presenten un impacte significatiu en [indicadors concrets].

### **Conclusió**
Aquest projecte demostra com els factors ambientals influeixen en la salut mental. Els models desenvolupats poden servir com a base per a futures investigacions i polítiques públiques.

---

## **Properes Etapes** 🚀

- Millorar els models actuals amb tècniques més avançades (ex. XGBoost, PCA).
- Incorporar dades de més anys per analitzar tendències temporals.
- Automatitzar el procés per utilitzar-lo en temps real.

---

## **Llicència** 📜
Aquest projecte està llicenciat sota la llicència [MIT](LICENSE.md).

---

## **Contacte** 📬
Per a més informació o col·laboracions, pots contactar amb:
- **Nom**: [El teu nom]
- **Email**: [el.teu.email@example.com]
- **LinkedIn**: [enllaç]
