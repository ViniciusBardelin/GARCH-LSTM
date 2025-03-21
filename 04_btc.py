##### Esse script encontra o arquivo CSV que, dentre os 25 arquivos CSV gerados, possui o menor QLIKE.

import pandas as pd
import numpy as np
import glob
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Função para calcular o QLIKE
def calculate_qlike(y_true, y_pred):
    return np.mean(np.log(y_pred ** 2) + (y_true ** 2) / (y_pred ** 2))

# FunÃ§Ã£o para calcular HMSE e HMAE
def calculate_hmse_hmae(y_true, y_pred):
    hmse = np.mean(((y_true - y_pred) ** 2) / np.maximum(1e-10, y_true ** 2))
    hmae = np.mean(np.abs(y_true - y_pred) / np.maximum(1e-10, y_true))
    return hmse, hmae

# Lista todos os arquivos CSV gerados
csv_files = glob.glob("run_*_predictions.csv")

# Inicializa variÃ¡veis para armazenar o melhor arquivo e o menor QLIKE
best_file = None
best_qlike = float("inf")

# Percorre todos os arquivos e calcula o QLIKE
for file in csv_files:
    df = pd.read_csv(file)
   
    y_true = df["Proxy Values"].values
    y_pred = df["Predicted Values"].values
   
    qlike = calculate_qlike(y_true, y_pred)

    # Atualiza se o novo QLIKE for menor
    if qlike < best_qlike:
        best_qlike = qlike
        best_file = file

# ApÃ³s encontrar o melhor arquivo, calcular as demais mÃ©tricas
if best_file:
    df_best = pd.read_csv(best_file)
    y_true_best = df_best["Proxy Values"].values
    y_pred_best = df_best["Predicted Values"].values

    mse = mean_squared_error(y_true_best, y_pred_best)
    mae = mean_absolute_error(y_true_best, y_pred_best)
    hmse, hmae = calculate_hmse_hmae(y_true_best, y_pred_best)

    print(f"Melhor arquivo: {best_file}")
    print(f"QLIKE: {best_qlike}")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"HMSE: {hmse}")
    print(f"HMAE: {hmae}")
else:
    print("Nenhum arquivo encontrado.")
    
    
'''

# sqrt:
Melhor arquivo: run_23_predictions.csv
QLIKE: -2.03689749434655

# normal:
Melhor arquivo: run_23_predictions.csv
QLIKE: -2.033587035923413



'''
