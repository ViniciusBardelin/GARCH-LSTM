## Nesse script, eu calculo as métricas de erro para os modelos econométricos
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

###########
## GARCH ##
###########

# Carregar previsões do modelo GARCH
volatility_data = pd.read_csv('garch_forecasts.csv')
volatility_data.rename(columns={'V1': 'Volatility'}, inplace=True)

# Padronizar as previsões de volatilidade do GARCH
x_min = volatility_data['Volatility'].min()
x_max = volatility_data['Volatility'].max()
volatility_data['Normalized_Volatility'] = (volatility_data['Volatility'] - x_min) / (x_max - x_min)
volatility_data = volatility_data.tail(441).reset_index(drop=True)  # Pegando as últimas 441 previsões e resetando o índice

# Carregar a proxy (volatilidade de Parkinson)
proxy_data = pd.read_csv('petr4_vol_forecasts_parkinson.csv')

# Padronizar a proxy
proxy_min = proxy_data['V1'].min()
proxy_max = proxy_data['V1'].max()
proxy_data['Normalized_Proxy'] = (proxy_data['V1'] - proxy_min) / (proxy_max - proxy_min)
proxy_data = proxy_data.tail(441).reset_index(drop=True)  # Pegando os últimos 441 valores e resetando o índice

# Criar o DataFrame final
predictions_df = pd.DataFrame({
    "Proxy Values": proxy_data['Normalized_Proxy'],  # Valores da proxy
    "Predicted Values": volatility_data['Normalized_Volatility']
})


# Calcular métricas de erro
mse = mean_squared_error(predictions_df["Proxy Values"], predictions_df["Predicted Values"])
mae = mean_absolute_error(predictions_df["Proxy Values"], predictions_df["Predicted Values"])
hmae = np.mean(np.abs(predictions_df["Proxy Values"] - predictions_df["Predicted Values"]) / 
               np.maximum(1e-10, predictions_df["Proxy Values"]))
hmse = np.mean(((predictions_df["Proxy Values"] - predictions_df["Predicted Values"]) ** 2) / 
               np.maximum(1e-10, predictions_df["Proxy Values"]))
qlike = np.mean(np.log(predictions_df["Predicted Values"] ** 2) + (predictions_df["Proxy Values"] ** 2) / (predictions_df["Predicted Values"] ** 2))

# DataFrame para armazenar métricas
metrics_df = pd.DataFrame({
    "Metric": ["MSE", "MAE", "HMAE", "HMSE", "QLIKE"],
    "Value": [mse, mae, hmae, hmse, qlike]
})


#############
## MSGARCH ##
#############

# Carregar previsões do modelo MSGARCH
volatility_data = pd.read_csv('msgarch_forecasts.csv')
volatility_data.rename(columns={'V1': 'Volatility'}, inplace=True)

# Padronizar as previsões de volatilidade do MSGARCH
x_min = volatility_data['Volatility'].min()
x_max = volatility_data['Volatility'].max()
volatility_data['Normalized_Volatility'] = (volatility_data['Volatility'] - x_min) / (x_max - x_min)
volatility_data = volatility_data.tail(441).reset_index(drop=True) #441 # Pegando as últimas 441 previsões e resetando o índice

# Carregar a proxy
proxy_data = pd.read_csv('petr4_vol_forecasts_parkinson.csv')

# Padronizar a proxy
proxy_min = proxy_data['V1'].min()
proxy_max = proxy_data['V1'].max()
proxy_data['Normalized_Proxy'] = (proxy_data['V1'] - proxy_min) / (proxy_max - proxy_min)
proxy_data = proxy_data.tail(441).reset_index(drop=True)  # Pegando os últimos 441 valores e resetando o índice

# Criar o DataFrame final
predictions_df = pd.DataFrame({
    "Proxy Values": proxy_data['Normalized_Proxy'],  # Valores da proxy
    "Predicted Values": volatility_data['Normalized_Volatility']
})


# Calcular métricas de erro
mse = mean_squared_error(predictions_df["Proxy Values"], predictions_df["Predicted Values"])
mae = mean_absolute_error(predictions_df["Proxy Values"], predictions_df["Predicted Values"])
hmae = np.mean(np.abs(predictions_df["Proxy Values"] - predictions_df["Predicted Values"]) / 
               np.maximum(1e-10, predictions_df["Proxy Values"]))
hmse = np.mean(((predictions_df["Proxy Values"] - predictions_df["Predicted Values"]) ** 2) / 
               np.maximum(1e-10, predictions_df["Proxy Values"]))

epsilon = 1e-10
qlike = np.mean(np.log(np.maximum(predictions_df["Predicted Values"] ** 2, epsilon)) + (predictions_df["Proxy Values"] ** 2) / np.maximum(predictions_df["Predicted Values"] ** 2, epsilon))

# DataFrame para armazenar métricas
metrics_df = pd.DataFrame({
    "Metric": ["MSE", "MAE", "HMAE", "HMSE", "QLIKE"],
    "Value": [mse, mae, hmae, hmse, qlike]
})

print(metrics_df)

#########
## GAS ##
#########

# Carregar previsões do modelo MSGARCH
volatility_data = pd.read_csv('gas_forecasts.csv')
volatility_data.rename(columns={'V1': 'Volatility'}, inplace=True)

# Padronizar as previsões de volatilidade do MSGARCH
x_min = volatility_data['Volatility'].min()
x_max = volatility_data['Volatility'].max()
volatility_data['Normalized_Volatility'] = (volatility_data['Volatility'] - x_min) / (x_max - x_min)
volatility_data = volatility_data.tail(441).reset_index(drop=True)  # Pegando as últimas 441 previsões e resetando o índice

# Carregar a proxy (volatilidade de Parkinson)
proxy_data = pd.read_csv('petr4_vol_forecasts_parkinson.csv')

# Padronizar a proxy
proxy_min = proxy_data['V1'].min()
proxy_max = proxy_data['V1'].max()
proxy_data['Normalized_Proxy'] = (proxy_data['V1'] - proxy_min) / (proxy_max - proxy_min)
proxy_data = proxy_data.tail(441).reset_index(drop=True)  # Pegando os últimos 441 valores e resetando o índice

# Criar o DataFrame final
predictions_df = pd.DataFrame({
    "Proxy Values": proxy_data['Normalized_Proxy'],  # Valores da proxy
    "Predicted Values": volatility_data['Normalized_Volatility']
})


# Calcular métricas de erro
mse = mean_squared_error(predictions_df["Proxy Values"], predictions_df["Predicted Values"])
mae = mean_absolute_error(predictions_df["Proxy Values"], predictions_df["Predicted Values"])
hmae = np.mean(np.abs(predictions_df["Proxy Values"] - predictions_df["Predicted Values"]) / 
               np.maximum(1e-10, predictions_df["Proxy Values"]))
hmse = np.mean(((predictions_df["Proxy Values"] - predictions_df["Predicted Values"]) ** 2) / 
               np.maximum(1e-10, predictions_df["Proxy Values"]))
qlike = np.mean(np.log(predictions_df["Predicted Values"] ** 2) + (predictions_df["Proxy Values"] ** 2) / (predictions_df["Predicted Values"] ** 2))

# DataFrame para armazenar métricas
metrics_df = pd.DataFrame({
    "Metric": ["MSE", "MAE", "HMAE", "HMSE", "QLIKE"],
    "Value": [mse, mae, hmae, hmse, qlike]
})
