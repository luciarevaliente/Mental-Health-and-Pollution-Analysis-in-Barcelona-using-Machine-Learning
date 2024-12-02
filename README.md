# **Salut Mental i Contaminació a Barcelona: Un Estudi Basat en Machine Learning**

## **Descripció del Projecte** 📄
por hacer

---

## **Objectius del Projecte** 🎯
por hacer

---

## **Contingut del Repositori** 📁

```plaintext
📦 Mental-Health-and-Pollution-Analysis-in-Barcelona-using-Machine-Learning
 ┣ 📂 data
 ┃ ┣ 📜 CitieSHealth_BCN_DATA_PanelStudy_20220414.csv (dataset principal)
 ┃ ┣ 📜 dataset.pkl (dataset format pkl)
 ┣ 📂 scripts
 ┃ ┣ 📜 01_load_data.py (carrega les dades i les guarda en format pickle)
 ┃ ┣ 📜 02_exploratory_analysis.py (processa i analitza les dades)
 ┃ ┣ 📜 03_data_cleaning.py (neteja i preprocesament de dades)
 ┣ 📂 visualizations
 ┃ ┣ 📂 boxplots/ (gràfiques per analitzar outliers)
 ┣ 📂 results
 ┃ ┣ 📂 visuals/ (gràfiques i visualitzacions generades)
 ┃ ┣ 📂 metrics/ (resultats dels models)
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
  - Factors de salut mental: *benestar, estrès, energia, son.*
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

- **`scripts/`**: Conté el codi principal per al processament de dades, l'anàlisi exploratòria i l'entrenament de models.
- **`results/`**: Arxius generats durant l'execució del projecte, incloent visualitzacions i mètriques.
- **`requirements.txt`**: Llista de dependències necessàries per a executar el projecte. 

---

## **Visualitzacions** 📸

Les visualitzacions generades pel projecte es guarden a la carpeta **`results/visuals/`** com arxius d'imatge (ex. PNG, JPEG, SVG). A continuació, es mostren algunes de les gràfiques generades:

### **Exemple de Gràfiques Generades** 📊

1. **Gràfica de Correlació de Variables Ambientals i Salut Mental**

   La següent gràfica mostra la correlació entre la qualitat de l'aire i els índexs de salut mental:

   ![Correlació de variables](results/visuals/grafica_1.png)

2. **Distribució dels Nivells de Contaminació per Districte de Barcelona**

   ![Distribució per districte](results/visuals/grafica_2.png)

### **Visualització en el Codi**

Les visualitzacions es generen dins dels scripts de la carpeta **`scripts/`**. Per exemple:

- **Anàlisi exploratòria** (`02_exploratory_analysis.py`): Conté les visualitzacions de la correlació entre diferents variables de salut i contaminació.
- **Clustering** (`04_clustering_analysis.py`): Genera gràfiques de dispersió per veure els resultats del clustering.

Les imatges es desaran a **`results/visuals/`** automàticament quan s'executin els scripts.

---

## **Resultats i Mètriques** 🧮

Les mètriques dels models s'emmagatzemen a la carpeta **`results/metrics/`**. Aquí trobaràs informació detallada sobre el rendiment dels models utilitzats en aquest projecte.

### **Exemple de Mètriques del Model de Regressió**:
- **RMSE**: 0.85
- **R²**: 0.92

Els resultats es poden consultar al fitxer **`results/metrics/regression_metrics.txt`**.

---

## **Dependències i Instal·lació** 📦

### Llibreries necessàries
Aquest projecte requereix Python 3.x i les següents llibreries:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

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

## **Autors del dataset**
Gignac, F., Righi, V., Toran, R., Paz Errandonea, L., Ortiz, R., Mijling, B., Naranjo, A., Nieuwenhuijsen, M., Creus, J., & Basagaña, X. (2022). CitieS-Health Barcelona Panel Study Results [Data set]. Zenodo. https://doi.org/10.5281/zenodo.6503022

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

Amb aquesta nova versió del README, inclou informació detallada sobre **les visualitzacions generades pels teus scripts**, com s'emmagatzemen a la carpeta **`results/visuals/`**, i com el codi pot generar els fitxers d'imatge de manera automàtica. També es fa referència a les mètriques dels models que s'emmagatzemen a **`results/metrics/`**.
