# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# import plotly.graph_objects as go
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor

# cleaned_dataset_path = 'data/cleaned_dataset.pkl'
# codif_dataset_path = 'data/codif_dataset.pkl'

# cleaned_dataset = pd.read_pickle(cleaned_dataset_path)
# data = pd.read_pickle(codif_dataset_path)

# # Verificar si la columna 'estres' está en el dataset limpio
# if 'estres' in cleaned_dataset.columns:
#     # Añadir la columna 'estres' al dataset codificado
#     if cleaned_dataset.shape[0] == data.shape[0]:
#         data['estres'] = cleaned_dataset['estres']
#         print("Columna 'estres' añadida correctamente.")
# def load_and_prepare_data(data, features, target_column):
#     """Carga y prepara los datos, seleccionando las características y el target."""
    
#     X = data[features]
#     y = data[target_column]
#     return train_test_split(X, y, test_size=0.2, random_state=42)

# X_train, y_train, X_test, y_test = load_and_prepare_data(data, ['ordenador', 'otrofactor', 'dayoftheweek', 'bienestar','covid_motor','drogas',], 'estres')

# rf = RandomForestRegressor()
# rf.fit(X_train, y_train)
# fig = go.Figure(go.Bar(
#             x=rf.feature_importances_,
#             y=X_train.columns,
#             orientation='h', marker_color='steelblue'))
# fig.update_layout(template='plotly_dark', title='<b>Estimating feature importance through the Random Forest model', title_x=0.5, 
#                  xaxis_title="Feature importance", yaxis_title='Feature')

# fig.show()

import matplotlib.pyplot as plt
import numpy as np

# Simulación de una curva de aprendizaje ideal
iterations = np.arange(1, 21)
train_error = 1 / iterations + 0.2  # Error de entrenamiento disminuye rápido
test_error = 1 / (iterations + 3) + 0.5  # Error de prueba disminuye más lento y se estabiliza

# Gráfico de la curva de aprendizaje
plt.figure(figsize=(8, 6))
plt.plot(iterations, train_error, label='Error de Entrenamiento', marker='o')
plt.plot(iterations, test_error, label='Error de Prueba', marker='o')
plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.7)
plt.title('Curva de Aprendizaje Ideal', fontsize=14)
plt.xlabel('Número de Iteraciones (o Complejidad del Modelo)', fontsize=12)
plt.ylabel('Error', fontsize=12)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()
