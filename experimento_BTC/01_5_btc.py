##### Esse script calcula as métricas de erro para os modelos econométricos. O MSGARCH é um pouco diferente por algumas complicações no ajuste.

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

###########
## GARCH ##
###########

# Carregar previsões do modelo GARCH
volatility_data = pd.read_csv('act_garch_forecasts.csv')
volatility_data.rename(columns={'Forecasts': 'Volatility'}, inplace=True)

# Carregar a proxy (volatilidade de Parkinson)
proxy_data = pd.read_csv('act_btc_proxy.csv')

# Supondo que ambas as tabelas tenham uma coluna de data para alinhamento
volatility_data['Index'] = pd.to_datetime(volatility_data['Index'])
proxy_data['Index'] = pd.to_datetime(proxy_data['Index'])

# Fazer um merge para garantir o alinhamento correto dos dados
merged_data = pd.merge(volatility_data, proxy_data, on='Index', how='inner')

# Padronizar as previsões de volatilidade do GARCH
x_min = merged_data['Volatility'].min()
x_max = merged_data['Volatility'].max()
merged_data['Normalized_Volatility'] = (merged_data['Volatility'] - x_min) / (x_max - x_min)

# Padronizar a proxy
proxy_min = merged_data['Forecasts'].min()
proxy_max = merged_data['Forecasts'].max()
merged_data['Normalized_Proxy'] = (merged_data['Forecasts'] - proxy_min) / (proxy_max - proxy_min)

# Selecionar os últimos 1112 valores alinhados (out-of-sample)
final_data = merged_data.tail(322).reset_index(drop=True)

# Criar o DataFrame final
predictions_df = final_data[['Index', 'Normalized_Proxy', 'Normalized_Volatility']]
predictions_df.rename(columns={'Normalized_Proxy': 'Proxy Values', 'Normalized_Volatility': 'Predicted Values'}, inplace=True)

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
    "Metric": ["QLIKE", "MSE", "MAE", "HMAE", "HMSE"],
    "Value": [qlike, mse, mae, hmae, hmse]
})

metrics_df

#############
## MSGARCH ##
#############

# Carregar previsões do modelo MSGARCH
volatility_data = pd.read_csv('act_msgarch_forecasts.csv')
volatility_data.rename(columns={'Forecasts': 'Volatility'}, inplace=True)

# Carregar a proxy (volatilidade de Parkinson)
proxy_data = pd.read_csv('act_btc_proxy.csv')

# Supondo que ambas as tabelas tenham uma coluna de data para alinhamento
volatility_data['Index'] = pd.to_datetime(volatility_data['Index'])
proxy_data['Index'] = pd.to_datetime(proxy_data['Index'])

# Fazer um merge para garantir o alinhamento correto dos dados
merged_data = pd.merge(volatility_data, proxy_data, on='Index', how='inner')

# Padronizar as previsões de volatilidade do MSGARCH
x_min = merged_data['Volatility'].min()
x_max = merged_data['Volatility'].max()
merged_data['Normalized_Volatility'] = (merged_data['Volatility'] - x_min) / (x_max - x_min)

# Padronizar a proxy
proxy_min = merged_data['Forecasts'].min()
proxy_max = merged_data['Forecasts'].max()
merged_data['Normalized_Proxy'] = (merged_data['Forecasts'] - proxy_min) / (proxy_max - proxy_min)

# Selecionar os últimos 1112 valores alinhados (out-of-sample)
final_data = merged_data.tail(322).reset_index(drop=True)

# Criar o DataFrame final
predictions_df = final_data[['Index', 'Normalized_Proxy', 'Normalized_Volatility']]
predictions_df.rename(columns={'Normalized_Proxy': 'Proxy Values', 'Normalized_Volatility': 'Predicted Values'}, inplace=True)

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
    "Metric": ["QLIKE", "MSE", "MAE", "HMAE", "HMSE"],
    "Value": [qlike, mse, mae, hmae, hmse]
})

metrics_df

#########
## GAS ##
#########

# Carregar previsões do modelo GAS
volatility_data = pd.read_csv('act_gas_forecasts.csv')
volatility_data.rename(columns={'Forecasts': 'Volatility'}, inplace=True)

# Carregar a proxy (volatilidade de Parkinson)
proxy_data = pd.read_csv('act_btc_proxy.csv')

# Supondo que ambas as tabelas tenham uma coluna de data para alinhamento
volatility_data['Index'] = pd.to_datetime(volatility_data['Index'])
proxy_data['Index'] = pd.to_datetime(proxy_data['Index'])

# Fazer um merge para garantir o alinhamento correto dos dados
merged_data = pd.merge(volatility_data, proxy_data, on='Index', how='inner')

# Padronizar as previsões de volatilidade do GAS
x_min = merged_data['Volatility'].min()
x_max = merged_data['Volatility'].max()
merged_data['Normalized_Volatility'] = (merged_data['Volatility'] - x_min) / (x_max - x_min)

# Padronizar a proxy
proxy_min = merged_data['Forecasts'].min()
proxy_max = merged_data['Forecasts'].max()
merged_data['Normalized_Proxy'] = (merged_data['Forecasts'] - proxy_min) / (proxy_max - proxy_min)

# Selecionar os últimos 1112 valores alinhados (out-of-sample)
final_data = merged_data.tail(322).reset_index(drop=True)

# Criar o DataFrame final
predictions_df = final_data[['Index', 'Normalized_Proxy', 'Normalized_Volatility']]
predictions_df.rename(columns={'Normalized_Proxy': 'Proxy Values', 'Normalized_Volatility': 'Predicted Values'}, inplace=True)

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
    "Metric": ["QLIKE", "MSE", "MAE", "HMAE", "HMSE"],
    "Value": [qlike, mse, mae, hmae, hmse]
})

metrics_df


########## TESTES COM SQRT

## msgarch com sqrt

# Carregar previsões do modelo
volatility_data = pd.read_csv('act_msgarch_forecasts.csv')
volatility_data.rename(columns={'Forecasts': 'Volatility'}, inplace=True)

# Carregar a proxy (volatilidade de Parkinson)
proxy_data = pd.read_csv('act_btc_proxy.csv')

# Supondo que ambas as tabelas tenham uma coluna de data para alinhamento
volatility_data['Index'] = pd.to_datetime(volatility_data['Index'])
proxy_data['Index'] = pd.to_datetime(proxy_data['Index'])

# Fazer um merge para garantir o alinhamento correto dos dados
merged_data = pd.merge(volatility_data, proxy_data, on='Index', how='inner')


merged_data['Sqrt_Volatility'] = np.sqrt(merged_data['Volatility'])

# Padronizar as previsões de volatilidade do MSGARCH
x_min = merged_data['Sqrt_Volatility'].min()
x_max = merged_data['Sqrt_Volatility'].max()
merged_data['Normalized_Volatility'] = (merged_data['Sqrt_Volatility'] - x_min) / (x_max - x_min)

# Padronizar a proxy
proxy_min = merged_data['Forecasts'].min()
proxy_max = merged_data['Forecasts'].max()
merged_data['Normalized_Proxy'] = (merged_data['Forecasts'] - proxy_min) / (proxy_max - proxy_min)

# Selecionar os últimos 1112 valores alinhados (out-of-sample)
final_data = merged_data.tail(322).reset_index(drop=True)

# Criar o DataFrame final
predictions_df = final_data[['Index', 'Normalized_Proxy', 'Normalized_Volatility']]
predictions_df.rename(columns={'Normalized_Proxy': 'Proxy Values', 'Normalized_Volatility': 'Predicted Values'}, inplace=True)

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
    "Metric": ["QLIKE", "MSE", "MAE", "HMAE", "HMSE"],
    "Value": [qlike, mse, mae, hmae, hmse]
})

metrics_df