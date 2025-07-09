###############
# BIBLIOTECAS
###############

from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import matplotlib.dates as mdates

###############
# DADOS
###############

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data = pd.read_csv("volatilidades_previstas_completo_corrigido_GARCH_1_1.csv")
print(data.head())

data['Date'] = pd.to_datetime(data['Date'], format="%Y-%m-%d")
data = data.sort_values('Date')
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data['Date'], data['Parkinson'],
        label="Proxy", linewidth=1, alpha=0.8)
ax.plot(data['Date'], data['Sigma_GARCH'],
        label="GARCH", linewidth=1, alpha=0.8)
ax.set_title("Volatilidade Proxy vs GARCH ao Longo do Tempo")
ax.set_xlabel("Data")
ax.set_ylabel("Volatilidade")
ax.grid(True, linestyle="--", alpha=0.5)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=45, ha="right")
ax.legend()
plt.tight_layout()
plt.show()

###############
# LSTM
###############

garch = data.filter(['Sigma_GARCH']) # dados de treino
dataset = garch.values # converte para array numpy
training_data_len = int(np.ceil(len(dataset)*0.95))

# Pre-processamento
scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset)

training_data = scaled_data[:training_data_len] # 95% dos dados

X_train, y_train = [], []

# sliding window de 22 dias
for i in range(22, len(training_data)):
    X_train.append(training_data[i-22:i, 0])
    y_train.append(training_data[i,0])
    
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# modelo
model = keras.models.Sequential()

model.add(keras.layers.LSTM(256, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(keras.layers.LSTM(128, return_sequences = False))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(1, activation = 'relu'))

model.summary()
model.compile(optimizer='adam',
              loss='mse',          # função de perda
              metrics=['mse']      # métrica de monitoramento: exibe MSE a cada época
)

training = model.fit(X_train, y_train, epochs = 100, batch_size = 64)

# Dados de teste
test_data = scaled_data[training_data_len - 22:]
X_test, y_test = [], dataset[training_data_len:]

for i in range(22, len(test_data)):
    X_test.append(test_data[i-22:i, 0])
    
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Previsão
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Gráfico
train = data[:training_data_len]
test = data[training_data_len:]

test = test.copy()

test['Predictions'] = predictions

plt.figure(figsize = (12,8))
plt.plot(train['Date'], train['Sigma_GARCH'], label = 'Treino', color = 'blue')
plt.plot(test['Date'], test['Sigma_GARCH'], label = 'Teste', color = 'orange')
plt.plot(test['Date'], test['Predictions'], label = 'Previsões', color = 'red')
plt.title("Previsões volatilidade - PETR4")
plt.xlabel('Data')
plt.legend()
plt.show()
























