from sklearn.model_selection import RandomizedSearchCV

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
