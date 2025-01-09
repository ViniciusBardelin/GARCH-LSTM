# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 18:20:19 2025

@author: vinic
"""

import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Função para criar janelas deslizantes
def create_sliding_windows(data, window_size):
    x, y = [], []
    for i in range(len(data) - window_size):
        x.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(x), np.array(y)

# Função para criar e treinar a rede LSTM
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
    
    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Treinamento
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

# Função para treinar e salvar previsões
def train_and_save_predictions(data, window_size, output_csv):
    print(f"Treinando LSTM para dados com janela de {window_size}...")
    
    # Criar janelas deslizantes
    x, y = create_sliding_windows(data, window_size)
    
    # Dividir em treino e validação
    split = int(0.8 * len(y))
    x_train, y_train = x[:split], y[:split]
    x_val, y_val = x[split:], y[split:]
    
    # Redimensionar entradas para o formato esperado pela LSTM
    x_train = np.expand_dims(x_train, axis=-1)
    x_val = np.expand_dims(x_val, axis=-1)
    
    # Treinar o modelo
    model, training_time, history = train_lstm(x_train, y_train, x_val, y_val, window_size)
    
    # Fazer previsões
    y_pred = model.predict(x_val)
    
    # Salvar previsões em CSV
    predictions_df = pd.DataFrame({
        "True Values": y_val,
        "Predicted Values": y_pred.flatten()
    })
    predictions_df.to_csv(output_csv, index=False)
    print(f"Previsões salvas em: {output_csv}")
    return model, predictions_df

###################
### Treinamento ###
###################

# Para processar o modelo GARCH
volatility_data = pd.read_csv('vol_forecasts_garch.csv')
volatility_data.rename(columns={'V1': 'Volatility'}, inplace=True)

# Padronização
x_min = volatility_data['Volatility'].min()
x_max = volatility_data['Volatility'].max()
volatility_data['Normalized_Volatility'] = (volatility_data['Volatility'] - x_min) / (x_max - x_min)

# Treinar e salvar previsões
data = volatility_data['Normalized_Volatility'].values
train_and_save_predictions(data, window_size=22, output_csv="predictions_garch_lstm.csv")

# Para processar o modelo MSGARCH
volatility_data = pd.read_csv('vol_forecasts_msgarch.csv')
volatility_data.rename(columns={'V1': 'Volatility'}, inplace=True)

# Padronização
x_min = volatility_data['Volatility'].min()
x_max = volatility_data['Volatility'].max()
volatility_data['Normalized_Volatility'] = (volatility_data['Volatility'] - x_min) / (x_max - x_min)

# Treinar e salvar previsões
data = volatility_data['Normalized_Volatility'].values
train_and_save_predictions(data, window_size=22, output_csv="predictions_msgarch_lstm.csv")

# Para processar o modelo GAS
volatility_data = pd.read_csv('vol_forecasts_gas.csv')
volatility_data.rename(columns={'GAS_Volatility': 'Volatility'}, inplace=True)

# Padronização
x_min = volatility_data['Volatility'].min()
x_max = volatility_data['Volatility'].max()
volatility_data['Normalized_Volatility'] = (volatility_data['Volatility'] - x_min) / (x_max - x_min)

# Treinar e salvar previsões
data = volatility_data['Normalized_Volatility'].values
train_and_save_predictions(data, window_size=22, output_csv="predictions_gas_lstm.csv")


####################
## PREVISÃO FINAL ##
####################

# Carregar previsões dos três modelos híbridos
garch_predictions = pd.read_csv("predictions_garch_lstm.csv")["Predicted Values"]
msgarch_predictions = pd.read_csv("predictions_msgarch_lstm.csv")["Predicted Values"]
gas_predictions = pd.read_csv("predictions_gas_lstm.csv")["Predicted Values"]

# Concatenar previsões em uma matriz para entrada da rede neural
X = np.stack([garch_predictions, msgarch_predictions, gas_predictions], axis=1)

# Converter a matriz X em um DataFrame
X_df = pd.DataFrame(X, columns=["GARCH_Predictions", "MSGARCH_Predictions", "GAS_Predictions"])

# Salvar em um arquivo CSV
X_df.to_csv("combined_predictions.csv", index=False)
