from sklearn.linear_model import LinearRegression, Ridge, Lasso  # Models lineals clàssics.
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor  # Models basats en arbres.
from xgboost import XGBRegressor  # Model avançat basat en arbres.
XGBRegressor._estimator_type = "regressor" # definim el tipus d'estimador manualment
from sklearn.svm import SVR  # Suport vectorial per a regressió.
from sklearn.preprocessing import PolynomialFeatures  # Transformació de dades per a regressió polinòmica.
from sklearn.pipeline import Pipeline  # Pipeline per encadenar passos del model.
from sklearn.model_selection import RandomizedSearchCV  # Optimització d'hiperparàmetres amb cerca aleatòria.

# Classe per gestionar diferents models de regressió
class RegressionModels:
    def __init__(self, model_type="linear", **kwargs):
        """
        Inicialitza un model basat en el tipus especificat.
        model_type pot ser: 'linear', 'ridge', 'lasso', 'random_forest',
        'xgboost', 'gradient_boosting', 'svr', 'polynomial_regression'.
        """
        self.model = self._initialize_model(model_type, **kwargs)

    def _initialize_model(self, model_type, **kwargs):
        """
        Retorna el model corresponent basat en el tipus especificat.
        """
        models = {
            "linear": LinearRegression(**kwargs),  # Regressió lineal bàsica.
            "ridge": Ridge(**kwargs),  # Regressió Ridge per evitar sobreajustament.
            "lasso": Lasso(**kwargs),  # Regressió Lasso per seleccionar característiques.
            "random_forest": RandomForestRegressor(
                max_depth=10,  # Limita la profunditat per evitar sobreajustament.
                min_samples_split=5,  # Nombre mínim de mostres per dividir un node.
                min_samples_leaf=2,  # Nombre mínim de mostres a les fulles.
                n_estimators=100,  # Nombre d'arbres.
                random_state=42,
                **kwargs
            ),
            "xgboost": XGBRegressor(
                eval_metric="rmse",  # Mètrica d'avaluació: arrel de l'error quadràtic mitjà.
                objective="reg:squarederror",  # Objectiu de la regressió.
                random_state=42,
                min_child_weight=5,
                **kwargs
            ),
            "gradient_boosting": GradientBoostingRegressor(
                random_state=42,
                **kwargs
            ),
            "svr": SVR(**kwargs),  # Regressió basada en màquines de vectors de suport.
            "polynomial_regression": Pipeline([
                ("polynomial_features", PolynomialFeatures()),  # Transformació de característiques polinòmiques.
                ("linear", LinearRegression(**kwargs))  # Model de regressió lineal amb les característiques polinòmiques.
            ])
        }
        if model_type not in models:
            raise ValueError(f"Model no reconegut: {model_type}")
        return models[model_type]

    def train(self, X_train, y_train):
        """Entrena el model seleccionat."""
        self.model.fit(X_train, y_train.ravel()) 

    def predict(self, X_test):
        """Realitza prediccions amb el model entrenat."""
        return self.model.predict(X_test)

    def get_model(self):
        """Retorna el model subjacents."""
        return self.model


# Paràmetres de cerca per a cada model
GRID_PARAMS = {
    "random_forest": {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2, 4]
    },
    "gradient_boosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 6, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2, 4]
    },
    "xgboost": {
        "n_estimators": [100, 300, 500],
        "max_depth": [3, 5, 10, 15],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.4, 0.6, 0.8],
        "colsample_bytree": [0.4, 0.6, 0.8, 1.0],
        "reg_alpha": [0, 0.1, 1],  # Regularització L1.
        "reg_lambda": [1, 5, 10]  # Regularització L2.
    },
    "svr": {
        "C": [0.1, 1, 10],  # Control de regularització.
        "epsilon": [0.01, 0.1, 0.2],  # Marges tolerats en prediccions.
        "kernel": ["linear", "rbf", "poly"],  # Tipus de funcions de nucli.
        "gamma": ["scale", "auto"]  # Gamma per a nuclis no lineals.
    },
    "polynomial_regression": {
        "polynomial_features__degree": [2, 3, 4],  # Grau del polinomi.
        "linear__fit_intercept": [True, False]  # Si incloure o no l'ordenada en el model.
    }
}


# Cerca del millor model
def get_best_model(model_name, base_model, param_grid, X_train, y_train, X_test, y_test):
    """
    Cerca els millors hiperparàmetres per a un model utilitzant RandomizedSearchCV.
    GridSearchCV va ser provat, però RandomizedSearchCV va donar millors resultats.
    """
    print(f"Optimitzant {model_name}...")
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        scoring="neg_mean_squared_error",
        n_iter=50,  # Nombre de combinacions a provar.
        cv=5,  # Validació creuada.
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    if model_name == 'xgboost':
        random_search.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)], 
            verbose=False
        )
    else:
        random_search.fit(X_train, y_train)
    print(f"Millors paràmetres per a {model_name}: {random_search.best_params_}")
    return random_search.best_estimator_
