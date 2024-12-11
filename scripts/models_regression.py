from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


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
            "svr": SVR(**kwargs),
            "polynomial_regression": Pipeline([
                ("polynomialfeatures", PolynomialFeatures()),
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
