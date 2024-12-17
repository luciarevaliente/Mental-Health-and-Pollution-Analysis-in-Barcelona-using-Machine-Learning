import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

cleaned_dataset_path = 'data/cleaned_dataset.pkl'
codif_dataset_path = 'data/codif_dataset.pkl'

cleaned_dataset = pd.read_pickle(cleaned_dataset_path)
data = pd.read_pickle(codif_dataset_path)

# Verificar si la columna 'estres' está en el dataset limpio
if 'estres' in cleaned_dataset.columns:
    # Mostrar el valor máximo y mínimo de la columna 'estres'
    max_estres = cleaned_dataset['estres'].max()
    min_estres = cleaned_dataset['estres'].min()
    print(f"Valor máximo de 'estres': {max_estres}")
    print(f"Valor mínimo de 'estres': {min_estres}")

