import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Função para criar janelas deslizantes
def create_sliding_windows(data, proxy, window_size):
    x, y = [], []
    for i in range(len(data) - window_size):
        x.append(data[i:i + window_size])
        y.append(proxy[i + window_size])
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
def train_and_save_hybrid_predictions(data, proxy, window_size, output_csv):
    print(f"Treinando LSTM para dados com janela de {window_size}...")
    
    # Criar janelas deslizantes
    x, y = create_sliding_windows(data, proxy, window_size)
    
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

# Carregar dados de retornos com proxy (retornos absolutos e ao quadrado) já calculado
returns_data = pd.read_csv("returns_data.csv")  

# Escolha do tipo de proxy: "volatility" ou "variance"
model_type = "volatility"

# Definir proxy diretamente a partir das colunas dos dados de retorno
if model_type == "volatility":
    proxy = returns_data["Absolute_Returns"].values
elif model_type == "variance":
    proxy = returns_data["Squared_Returns"].values
else:
    raise ValueError("model_type deve ser 'volatility' ou 'variance'")

# Para processar o modelo GARCH
volatility_data = pd.read_csv('vol_forecasts_garch_atual.csv')
volatility_data.rename(columns={'V1': 'Volatility'}, inplace=True)

# Padronização
x_min = volatility_data['Volatility'].min()
x_max = volatility_data['Volatility'].max()
volatility_data['Normalized_Volatility'] = (volatility_data['Volatility'] - x_min) / (x_max - x_min)

# Treinar e salvar previsões do modelo híbrido GARCH-LSTM
data = volatility_data['Normalized_Volatility'].values
aligned_length = min(len(proxy), len(data))  # Alinhar proxy e previsões
proxy = proxy[-aligned_length:]
data = data[-aligned_length:]
train_and_save_hybrid_predictions(data, proxy, window_size=22, output_csv="predictions_garch_lstm.csv")

# Calcular o MSE
predictions_df = pd.read_csv("predictions_garch_lstm.csv")
mse = mean_squared_error(predictions_df["True Values"], predictions_df["Predicted Values"])
print(f"MSE do modelo GARCH-LSTM: {mse:.6f}") # 1.464322

# Para processar o modelo MSGARCH
volatility_data = pd.read_csv('vol_forecasts_msgarch_atual.csv')
volatility_data.rename(columns={'V1': 'Volatility'}, inplace=True)

# Padronização
x_min = volatility_data['Volatility'].min()
x_max = volatility_data['Volatility'].max()
volatility_data['Normalized_Volatility'] = (volatility_data['Volatility'] - x_min) / (x_max - x_min)

data = volatility_data['Normalized_Volatility'].values
aligned_length = min(len(proxy), len(data))
proxy = proxy[-aligned_length:]
data = data[-aligned_length:]
train_and_save_hybrid_predictions(data, proxy, window_size=22, output_csv="predictions_msgarch_lstm.csv")

# Calcular o MSE MSGARCH
predictions_df = pd.read_csv("predictions_msgarch_lstm.csv")
mse = mean_squared_error(predictions_df["True Values"], predictions_df["Predicted Values"])
print(f"MSE do modelo MSGARCH-LSTM: {mse:.6f}") # 1.445450

# Para processar o modelo GAS
volatility_data = pd.read_csv('vol_forecasts_gas_certo.csv')
volatility_data.rename(columns={'V1': 'Volatility'}, inplace=True)

# Padronização
x_min = volatility_data['Volatility'].min()
x_max = volatility_data['Volatility'].max()
volatility_data['Normalized_Volatility'] = (volatility_data['Volatility'] - x_min) / (x_max - x_min)

data = volatility_data['Normalized_Volatility'].values
aligned_length = min(len(proxy), len(data))
proxy = proxy[-aligned_length:]
data = data[-aligned_length:]
train_and_save_hybrid_predictions(data, proxy, window_size=22, output_csv="predictions_gas_lstm.csv")

# Calcular o MSE GAS
predictions_df = pd.read_csv("predictions_garch_lstm.csv")
mse = mean_squared_error(predictions_df["True Values"], predictions_df["Predicted Values"])
print(f"MSE do modelo GAS-LSTM: {mse:.6f}") # 1.464322
