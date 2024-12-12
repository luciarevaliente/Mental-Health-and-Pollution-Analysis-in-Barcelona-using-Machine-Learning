from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

# Clase de modelos de regresión
class RegressionModels:
    def __init__(self, model_type="linear", **kwargs):
        """
        Inicializa un modelo basado en el tipo especificado.
        model_type puede ser: 'linear', 'ridge', 'lasso', 'random_forest',
        'xgboost', 'gradient_boosting', 'svr', 'polynomial_regression'.
        """
        self.model = self._initialize_model(model_type, **kwargs)

    def _initialize_model(self, model_type, **kwargs):
        """Devuelve el modelo correspondiente basado en el tipo."""
        models = {
            "linear": LinearRegression(**kwargs),
            "ridge": Ridge(**kwargs),
            "lasso": Lasso(**kwargs),
            "random_forest": RandomForestRegressor(**kwargs),
            "xgboost": XGBRegressor(**kwargs),
            "gradient_boosting": GradientBoostingRegressor(**kwargs),
            "polynomial_regression": Pipeline([
                ("polynomialfeatures", PolynomialFeatures(degree=2)),
                ("linearregression", Lasso(**kwargs))
            ])
        }
        if model_type not in models:
            raise ValueError(f"Modelo no reconocido: {model_type}")
        return models[model_type]

    def train(self, X_train, y_train):
        """Entrena el modelo."""
        self.model.fit(X_train, y_train.ravel())

    def predict(self, X_test):
        """Realiza predicciones."""
        return self.model.predict(X_test)

    def get_model(self):
        """Devuelve el modelo subyacente."""
        return self.model

# Parámetros de búsqueda
GRID_PARAMS = {
    "ridge": {"alpha": [0.1, 1.0, 10.0]},
    "lasso": {"alpha": [0.01, 0.1, 1.0]},
    "random_forest": {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    },
    "gradient_boosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 6, 10]
    },
    "xgboost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 6, 10],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.8, 1.0]
    },
    "svr": {
        "C": [0.1, 1.0, 10.0],
        "kernel": ["linear", "rbf"],
        "epsilon": [0.01, 0.1, 1.0]
    },
    "polynomial_regression": {
    "polynomialfeatures__degree": [2],
    "linearregression__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]  # Valores de regularización
}
}

# Buscar el mejor modelo
def get_best_model(model_name, base_model, param_grid,X_train, y_train):
    print(f"Optimizando {model_name}...")
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        scoring="neg_mean_squared_error",
        n_iter=10,  # Número de combinaciones
        cv=5,  # Validación cruzada
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    random_search.fit(X_train, y_train)
    print(f"Mejores parámetros para {model_name}: {random_search.best_params_}")
    return random_search.best_estimator_
