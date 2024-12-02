# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 17:20:58 2024

@author: vinic
"""

## Bibliotecas
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

## GARCH-LSTM 

# Dados (previsões de volatilidade calculadas no R)
lstm_data = pd.read_excel("vol_forecasts.xlsx")
print(lstm_data.head())

# Definir o número de timesteps (tamanho da janela)
n_timesteps = 100

# Preparar os dados para a LSTM
X = []
y = []


# O objetivo é treinar a LSTM para prever as volatilidades futuras com base nas previsões passadas
for i in range(n_timesteps, len(lstm_data)):
    X.append(lstm_data.iloc[i - n_timesteps:i, 1:].values)  # Usando a coluna "Volatility" como entrada
    y.append(lstm_data.iloc[i, 1])  # Previsão da volatilidade para o próximo dia

# Converter as listas em arrays numpy
X = np.array(X)
y = np.array(y)

# Reshaping X para o formato (amostras, timesteps, características)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construir a rede neural LSTM
model = Sequential()

# Adicionar a camada LSTM
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))

# Adicionar mais uma camada LSTM (se necessário)
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

# Adicionar a camada de saída
model.add(Dense(units=1))

# Compilar o modelo
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Treinar o modelo
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Prever a volatilidade no conjunto de teste
y_pred = model.predict(X_test)

# Avaliar a performance do modelo
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')

# Plotar a comparação entre as previsões e os valores reais
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Valores Reais')
plt.plot(y_pred, label='Previsões')
plt.title('Previsões vs Valores Reais de Volatilidade')
plt.legend()
plt.show()
