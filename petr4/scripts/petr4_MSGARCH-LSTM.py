# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 17:07:56 2025

@author: vinic
"""

## GARCH-LSTM - Petrobras

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Função para criar janelas deslizantes
def create_sliding_windows(data, proxy, window_size):
    x, y = [], []
    for i in range(len(data) - window_size):
        x.append(data[i:i + window_size])
        y.append(proxy[i + window_size])
    return np.array(x), np.array(y)

# Função para treinar e avaliar a rede LSTM
def train_lstm(x_train, y_train, x_val, y_val, window_size):
    model = Sequential([
        LSTM(256, input_shape=(window_size, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(128, return_sequences=False),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    start_time = time.time()
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=[early_stop],
        verbose=1
    )
    training_time = time.time() - start_time
    return model, training_time, history

# Função para treinar e salvar previsões do modelo híbrido
def train_and_save_hybrid_predictions(data, proxy, window_size, output_csv):
    print(f"Training GARCH-LSTM hybrid model with window size {window_size}...")
    
    # Criar janelas deslizantes
    x, y = create_sliding_windows(data, proxy, window_size)
    
    # Dividir os dados em treino e validação
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # Redimensionar entradas para o formato esperado pela LSTM
    x_train = np.expand_dims(x_train, axis=-1)
    x_val = np.expand_dims(x_val, axis=-1)
    
    # Treinar o modelo
    model, training_time, history = train_lstm(x_train, y_train, x_val, y_val, window_size)
    
    # Fazer previsões
    y_pred = model.predict(x_val)
    
    # Criar DataFrame com previsões e valores proxy
    predictions_df = pd.DataFrame({
        "Proxy Values": y_val,  # Valores da proxy
        "Predicted Values": y_pred.flatten()
    })
    
    # Calcular métricas de erro
    mse = mean_squared_error(predictions_df["Proxy Values"], predictions_df["Predicted Values"])
    mae = mean_absolute_error(predictions_df["Proxy Values"], predictions_df["Predicted Values"])
    hmae = np.mean(np.abs(predictions_df["Proxy Values"] - predictions_df["Predicted Values"]) / 
                   np.maximum(1e-10, predictions_df["Proxy Values"]))
    hmse = np.mean(((predictions_df["Proxy Values"] - predictions_df["Predicted Values"]) ** 2) / 
                   np.maximum(1e-10, predictions_df["Proxy Values"]))
    
    # DataFrame para armazenar métricas
    metrics_df = pd.DataFrame({
        "Metric": ["MSE", "MAE", "HMAE", "HMSE"],
        "Value": [mse, mae, hmae, hmse]
    })
    
    # Exibir métricas calculadas
    print(metrics_df)
    
    # Salvar previsões em CSV
    predictions_df.to_csv(output_csv, index=False)


###################
### Treinamento ###
###################

# Carregar previsões do modelo GARCH
volatility_data = pd.read_csv('msgarch_forecasts.csv')
volatility_data.rename(columns={'V1': 'Volatility'}, inplace=True)

# Padronizar as previsões de volatilidade do GARCH
x_min = volatility_data['Volatility'].min()
x_max = volatility_data['Volatility'].max()
volatility_data['Normalized_Volatility'] = (volatility_data['Volatility'] - x_min) / (x_max - x_min)

# Carregar proxy (retornos ao quadrado ou janela exponencial)
proxy_data = pd.read_csv('petrobras_squared_returns.csv')  # retornos ao quadrado

# Padronizar a proxy
proxy_min = proxy_data['Squared_Returns'].min()
proxy_max = proxy_data['Squared_Returns'].max()
proxy_data['Normalized_Proxy'] = (proxy_data['Squared_Returns'] - proxy_min) / (proxy_max - proxy_min)

# "Alinhando" as previsões e proxy (datas conformes entre as séries)
aligned_length = min(len(proxy_data['Normalized_Proxy']), len(volatility_data['Normalized_Volatility']))
proxy = proxy_data['Normalized_Proxy'].values[-aligned_length:]
data = volatility_data['Normalized_Volatility'].values[-aligned_length:]

# Treinar o modelo GARCH-LSTM
train_and_save_hybrid_predictions(data, proxy, window_size=22, output_csv="predictions_msgarch_lstm.csv")

# Gráfico

# Carregar as previsões do modelo GARCH-LSTM
predictions_df = pd.read_csv("predictions_msgarch_lstm.csv")
predictions_df.columns = ["Proxy Values", "Predicted Values"]

# Adicionar a coluna de datas manualmente (29/01/2016 como início, onde começam as previsões)
start_date = "2016-01-29"
date_range = pd.date_range(start=start_date, periods=len(predictions_df))
predictions_df["Date"] = date_range

# Carregar a proxy (Squared Returns neste caso)
proxy_data = pd.read_csv("petrobras_squared_returns.csv")

# Ajustar a proxy para começar em 29/01/2016 
proxy_data["Date"] = pd.to_datetime(proxy_data["Date"])
proxy_data = proxy_data[proxy_data["Date"] >= date_range[0]]

# Garantindo que as séries estão alinhadas
aligned_length = min(len(proxy_data), len(predictions_df))
proxy_data = proxy_data.iloc[-aligned_length:]
predictions_df = predictions_df.iloc[-aligned_length:]

# Gráfico comparativo entre a proxy e as previsões
plt.figure(figsize=(12, 6))
plt.plot(predictions_df["Date"], predictions_df["Proxy Values"], label="Proxy Values", color="blue", alpha=0.7)
plt.plot(predictions_df["Date"], predictions_df["Predicted Values"], label="Predicted Values (MSGARCH-LSTM)", color="red", linestyle="--", alpha=0.7)
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.title("Comparison: Proxy vs Predicted Values (MSGARCH-LSTM)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("comparison_proxy_msgarch_lstm_petrobras.png")
plt.savefig("comparison_proxy_msgarch_lstm_petr4.pdf")
plt.show()


# Salvar o arquivo ajustado com datas
predictions_df.to_csv("predictions_msgarch_lstm_with_dates.csv", index=False)

## Identificando eventos extremos

# Carregar as previsões e as proxies (com datas)
predictions_df = pd.read_csv("predictions_msgarch_lstm_with_dates.csv")
proxy_data = pd.read_csv("petrobras_squared_returns.csv")

# Filtrar apenas as colunas necessárias e garantir alinhamento
proxy_data = proxy_data[["Date", "Squared_Returns"]]
proxy_data["Date"] = pd.to_datetime(proxy_data["Date"])
predictions_df["Date"] = pd.to_datetime(predictions_df["Date"])

# Unir os dados de previsões e proxy
merged_df = pd.merge(predictions_df, proxy_data, on="Date", how="inner")

# Identificar limites extremos (95º percentil como exemplo)
proxy_threshold = merged_df["Proxy Values"].quantile(0.95)
predicted_threshold = merged_df["Predicted Values"].quantile(0.95)

# Filtrar os dias extremos para a proxy e previsões
extreme_proxy = merged_df[merged_df["Proxy Values"] > proxy_threshold]
extreme_predictions = merged_df[merged_df["Predicted Values"] > predicted_threshold]

# Dias de eventos extremos em ambos os casos
extreme_events = pd.concat([extreme_proxy, extreme_predictions]).drop_duplicates()

# Salvar os dias extremos em um arquivo CSV
extreme_events.to_csv("msgarch_lstm_extreme_events.csv", index=False)