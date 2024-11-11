# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 18:30:57 2024

@author: vinic
"""

# Esse script contem apenas a implementação da rede LSTM; os dados que usarei como input para a rede
# foram obtidos no R, de nome `cod_IC_v1.R`.

## Bibliotecas
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


# Carregar os dados (previsões de volatilidade, parâmetros GARCH e volume de transação) obtidos no R
lstm_data = pd.read_csv("lstm_data.csv")
realized_volatility = pd.read_csv("realized_volatility.csv")

# Definir o número de timesteps (tamanho da janela)
n_timesteps = 30 # usando como base Kim e Won (2018) que usam 1 mês de negociação

# Separar as variáveis de entrada X e a variável target (y)
X = []
y = []

for i in range(n_timesteps, len(lstm_data)):
    X.append(lstm_data.iloc[i - n_timesteps:i, :-1].values)
    y.append(lstm_data.iloc[i, 0]) # a coluna target é a previsão da volatilidade

# Converter para numpy arrays
X, y = np.array(X), np.array(y)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir o modelo (muito simples, só para testes) LSTM
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(n_timesteps, X.shape[2])))
model.add(Dense(1))

# Compilar o modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Treinar o modelo
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Fazer previsões out-of-sample
predictions = model.predict(X_test)

# Ajuste para mesmo comprimento entre `realized_volatility` e `predictions`
min_length = min(len(realized_volatility), len(predictions))
realized_volatility = realized_volatility[:min_length]
predictions = predictions[:min_length]

# Calcular MAE e MSE usando volatilidade realizada
mae = mean_absolute_error(realized_volatility, predictions)
mse = mean_squared_error(realized_volatility, predictions)

print(f"MAE: {mae}")
print(f"MSE: {mse}")

# Calcular HMAE e HMSE (ajustados para heteroscedasticidade)
weights = 1 / realized_volatility.squeeze()  # Inverter a volatilidade realizada para dar pesos às observações
hmae = np.mean(weights * np.abs(realized_volatility.squeeze() - predictions.squeeze()))
hmse = np.mean(weights * (realized_volatility.squeeze() - predictions.squeeze())**2)

print(f"HMAE: {hmae}")
print(f"HMSE: {hmse}")

# Visualizar as previsões vs. volatilidade realizada
plt.plot(realized_volatility, color='blue', label='Volatilidade Realizada')
plt.plot(predictions, color='red', label='Previsão GARCH-LSTM')
plt.title('Previsão de Volatilidade com GARCH-LSTM')
plt.xlabel('Amostra')
plt.ylabel('Volatilidade')
plt.legend()
plt.show()
