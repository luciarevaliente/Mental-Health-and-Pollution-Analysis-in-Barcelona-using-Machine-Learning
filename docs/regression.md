### Descripció detallada dels fitxers i funcionalitats

#### **Fitxer: `RegressionModels`**
Aquest fitxer conté la definició d’una classe que encapsula diferents models de regressió disponibles.

---

##### **Classe `RegressionModels`**
1. **Objectiu:**
   - Proporcionar una manera flexible per inicialitzar, entrenar i fer prediccions utilitzant diferents models de regressió.
   - Centralitza la gestió dels models per facilitar el seu ús en una pipeline de *machine learning*.

2. **Components:**

   - **`__init__(self, model_type="linear", **kwargs`)**
     - Inicialitza el model segons el tipus (`model_type`) especificat.
     - Permet seleccionar models com:
       - `LinearRegression`: Model bàsic de regressió lineal.
       - `Ridge`: Regressió amb regularització L2.
       - `Lasso`: Regressió amb regularització L1.
       - `RandomForestRegressor`: Ensamblatge basat en arbres de decisió.
       - `GradientBoostingRegressor`: Model de *boosting* per a regressió.
       - `XGBRegressor`: Implementació de *boosting* amb XGBoost.
     - Els hiperparàmetres específics de cada model es passen com arguments (`kwargs`).

   - **`train(self, X_train, y_train)`**
     - Entrena el model amb el conjunt d'entrenament.
     - Utilitza el mètode `fit` del model subjacient.
     - Assegura que `y_train` tingui una dimensió compatible amb el model.

   - **`predict(self, X_test)`**
     - Genera prediccions utilitzant el conjunt de prova.
     - Retorna les prediccions del model subjacient.

   - **`get_model(self)`**
     - Retorna el model subjacient ja instanciat (útil per a ajustos addicionals, com la cerca d'hiperparàmetres amb `GridSearchCV`).

---

#### **Fitxer: `preparation`**
Aquest fitxer inclou funcions per gestionar el preprocesament de dades i dividir-les en conjunts d’entrenament i prova.

---

##### **Funcions:**

1. **`load_data(file_path)`**
   - Carrega un fitxer en format `pickle` com un `DataFrame` de pandas.
   - **Entrada:**
     - `file_path`: Ruta del fitxer `pickle`.
   - **Sortida:**
     - Un `DataFrame` amb el conjunt de dades preprocesat llest per al seu ús.

2. **`separacio_train_test(data, target_columns)`**
   - Divideix el conjunt de dades en conjunts d’entrenament i prova.
   - **Entrada:**
     - `data`: `DataFrame` amb dades preprocesades.
     - `target_columns`: Llista amb els noms de les columnes objectiu.
   - **Sortida:**
     - `X_train`, `X_test`: Subconjunts de les característiques.
     - `y_train`, `y_test`: Subconjunts de la/les columna/es objectiu.
   - **Detalls:**
     - Utilitza `train_test_split` de `scikit-learn` amb un 80%-20% per entrenament i prova.
     - Converteix `y_train` i `y_test` a arrays unidimensionals (`.ravel()`).

---

#### **Fitxer principal**
Aquest fitxer executa tota la pipeline per entrenar, avaluar i visualitzar models de regressió.

---

##### **Flux del codi:**

1. **Carrega i preparació de dades:**
   - Carrega un conjunt de dades preprocessat des d’un fitxer `pickle` utilitzant `load_data`.
   - Divideix les dades en entrenament i prova amb `separacio_train_test`.

2. **Definició de models i hiperparàmetres:**
   - Defineix els models a avaluar:
     - `ridge`, `lasso`, `random_forest`, `gradient_boosting`, `xgboost`.
   - Específica els hiperparàmetres a optimitzar mitjançant `GridSearchCV` a `GRID_PARAMS`.

3. **Entrenament i optimització:**
   - Utilitza la funció `get_best_model` per:
     - Cercar els millors hiperparàmetres mitjançant validació creuada.
     - Ajustar els models amb el conjunt d'entrenament (`X_train`, `y_train`).

4. **Avaluació del rendiment:**
   - Genera prediccions amb el conjunt de prova (`X_test`).
   - Calcula mètriques com:
     - **RMSE (Root Mean Squared Error):** Penalitza errors grans.
     - **MAE (Mean Absolute Error):** Promig d’errors absoluts.
     - **R² (Coeficient de determinació):** Mesura com de bé el model explica la variància.
   - Mostra les mètriques per a cada model.

5. **Importància de característiques:**
   - Calcula la importància de les característiques:
     - Per `ridge` i `lasso`: Utilitza els coeficients dels models (`coef_`).
     - Per altres models: Utilitza `feature_importances_`.
   - Desa els resultats en fitxers Excel per a anàlisi posterior.

6. **Visualització de resultats:**
   - **Comparació de models:** Gràfic de mètriques (RMSE, MAE, R²) per cada model.
   - **Valors reals vs prediccions:** Gràfic de dispersió per avaluar l'alineació entre prediccions i valors reals.
   - **Residus:** Mostra els errors entre les prediccions i els valors reals, útil per detectar patrons d'error.

---

#### **Fitxer: `evaluation`**
1. **Funció `get_best_model(model_name, base_model, param_grid)`**
   - Utilitza `GridSearchCV` per trobar els millors hiperparàmetres d’un model.
   - **Entrada:**
     - `model_name`: Nom del model.
     - `base_model`: Model base a optimitzar.
     - `param_grid`: Quadrícula d’hiperparàmetres.
   - **Sortida:**
     - Model ajustat amb els millors hiperparàmetres.

---

#### **Objectiu general del sistema**
- **Entrenar i avaluar models:**
  - Seleccionar i comparar diferents algorismes de regressió.
  - Optimitzar els seus hiperparàmetres.
  - Avaluar el rendiment amb mètriques estàndard.

- **Interpretar la importància de les característiques:**
  - Identificar les variables més rellevants per a les prediccions.

- **Visualitzar resultats:**
  - Proporcionar gràfics intuïtius per entendre l'ajust i els errors dels models.

---

#### **Extensions possibles:**
- Afegir més models (p. ex. regressió amb vectors de suport o xarxes neuronals).
- Automatitzar més etapes del preprocessament.
- Incorporar tècniques avançades de selecció de característiques.