from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

class RegressionModels:
    def __init__(self, model_type="linear", **kwargs):
        """
        Inicializa un modelo basado en el tipo.
        model_type: 'linear', 'ridge', 'lasso', 'random_forest', 'xgboost'.
        kwargs: hiperparámetros del modelo.
        """
        if model_type == "linear":
            self.model = LinearRegression(**kwargs)
        elif model_type == "ridge":
            self.model = Ridge(**kwargs)
        elif model_type == "lasso":
            self.model = Lasso(**kwargs)
        elif model_type == "random_forest":
            self.model = RandomForestRegressor(**kwargs)
        elif model_type == "xgboost":
            self.model = XGBRegressor(**kwargs)
        else:
            raise ValueError("Modelo no reconocido.")

    def train(self, X_train, y_train):
        """Entrena el modelo."""
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Realiza predicciones."""
        return self.model.predict(X_test)

    def get_model(self):
        """Devuelve el modelo subyacente."""
        return self.model
