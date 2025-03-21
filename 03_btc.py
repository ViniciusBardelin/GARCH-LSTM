##### Esse script calcula o QLIKE e o MSE médio por pastas (a ideia é usar esse script dentro das pastas onde foram salvos os 25 arquivos CSVs)

import pandas as pd
import numpy as np
import glob
import os
from sklearn.metrics import mean_squared_error

# Função para calcular o QLIKE
def calculate_qlike(y_true, y_pred):
    return np.mean(np.log(y_pred ** 2) + (y_true ** 2) / (y_pred ** 2))

# Obtém todos os arquivos CSV que seguem o padrão "run_*_predictions.csv" no diretório atual
csv_files = glob.glob("run_*_predictions.csv")

if not csv_files:
    print("Nenhum arquivo CSV encontrado no diretório atual.")
else:
    qlike_list = []
    mse_list = []

    # Processa cada arquivo CSV encontrado
    for file in csv_files:
        df = pd.read_csv(file)
       
        y_true = df["Proxy Values"].values
        y_pred = df["Predicted Values"].values

        qlike_list.append(calculate_qlike(y_true, y_pred))
        mse_list.append(mean_squared_error(y_true, y_pred))

    # Calcula as métricas médias e desvios padrão
    mean_qlike = np.mean(qlike_list)
    std_qlike = np.std(qlike_list)
    mean_mse = np.mean(mse_list)
    std_mse = np.std(mse_list)

    # Exibe os resultados
    print(f"QLIKE médio: {mean_qlike:.6f} (±{std_qlike:.6f})")
    print(f"MSE médio: {mean_mse:.6f} (±{std_mse:.6f})")
    print("-" * 50)

# Exibe o diretório atual para verificar onde o código está sendo executado
print("Diretório atual:", os.getcwd()) 