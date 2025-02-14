######################
## PETR4 GARCH-LSTM ## ## código original: GARCH_LSTM.py
######################

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
        "Proxy Values": y_val,  
        "Predicted Values": y_pred.flatten()
    })
    
    # Calcular métricas de erro
    mse = mean_squared_error(predictions_df["Proxy Values"], predictions_df["Predicted Values"])
    mae = mean_absolute_error(predictions_df["Proxy Values"], predictions_df["Predicted Values"])
    hmae = np.mean(np.abs(predictions_df["Proxy Values"] - predictions_df["Predicted Values"]) / 
                   np.maximum(1e-10, predictions_df["Proxy Values"]))
    hmse = np.mean(((predictions_df["Proxy Values"] - predictions_df["Predicted Values"]) ** 2) / 
                   np.maximum(1e-10, predictions_df["Proxy Values"]))
    qlike = np.mean(np.log(y_pred ** 2) + (y_val ** 2) / (y_pred ** 2))
    
    # DataFrame para armazenar métricas
    metrics_df = pd.DataFrame({
        "Metric": ["MSE", "MAE", "HMAE", "HMSE", "QLIKE"],
        "Value": [mse, mae, hmae, hmse, qlike]
    })
    
    # Exibir métricas calculadas
    print(metrics_df)
    
    # Salvar previsões em CSV
    predictions_df.to_csv(output_csv, index=False)


##################################
### Treinamento com  Parkinson ###
##################################

################
## GARCH-LSTM ##
################

# Carregar previsões do modelo GARCH
volatility_data = pd.read_csv('garch_forecasts.csv') # 3727 - 1500 = 2227 previsões do modelo GARCH
volatility_data.rename(columns={'V1': 'Volatility'}, inplace=True)

# Padronizar as previsões de volatilidade do GARCH
x_min = volatility_data['Volatility'].min()
x_max = volatility_data['Volatility'].max()
volatility_data['Normalized_Volatility'] = (volatility_data['Volatility'] - x_min) / (x_max - x_min)

# Carregar proxy
proxy_data = pd.read_csv('petr4_vol_forecasts_parkinson_proxy.csv')

# Padronizar a proxy
proxy_min = proxy_data['V1'].min()
proxy_max = proxy_data['V1'].max()
proxy_data['Normalized_Proxy'] = (proxy_data['V1'] - proxy_min) / (proxy_max - proxy_min)

# "Alinhando" as previsões e proxy (datas conformes entre as séries)
aligned_length = min(len(proxy_data['Normalized_Proxy']), len(volatility_data['Normalized_Volatility']))
proxy = proxy_data['Normalized_Proxy'].values[-aligned_length:]
data = volatility_data['Normalized_Volatility'].values[-aligned_length:]

# Treinar o modelo GARCH-LSTM
train_and_save_hybrid_predictions(data, proxy, window_size=22, output_csv="teste1_predictions_garch_lstm_parkinson.csv")

##################
## MSGARCH-LSTM ##
##################

# Carregar previsões do modelo GARCH
volatility_data = pd.read_csv('msgarch_forecasts.csv')
volatility_data.rename(columns={'V1': 'Volatility'}, inplace=True)

# Padronizar as previsões de volatilidade do GARCH
x_min = volatility_data['Volatility'].min()
x_max = volatility_data['Volatility'].max()
volatility_data['Normalized_Volatility'] = (volatility_data['Volatility'] - x_min) / (x_max - x_min)

# Carregar proxy
proxy_data = pd.read_csv('petr4_vol_forecasts_parkinson_proxy.csv')  # PARKINSON

# Padronizar a proxy
proxy_min = proxy_data['V1'].min()
proxy_max = proxy_data['V1'].max()
proxy_data['Normalized_Proxy'] = (proxy_data['V1'] - proxy_min) / (proxy_max - proxy_min)

# "Alinhando" as previsões e proxy (datas conformes entre as séries)
aligned_length = min(len(proxy_data['Normalized_Proxy']), len(volatility_data['Normalized_Volatility']))
proxy = proxy_data['Normalized_Proxy'].values[-aligned_length:]
data = volatility_data['Normalized_Volatility'].values[-aligned_length:]

# Treinar o modelo GARCH-LSTM
train_and_save_hybrid_predictions(data, proxy, window_size=22, output_csv="teste_predictions_msgarch_lstm_parkinson.csv")

##############
## GAS-LSTM ##
##############

# Carregar previsões do modelo GARCH
volatility_data = pd.read_csv('gas_forecasts.csv')
volatility_data.rename(columns={'V1': 'Volatility'}, inplace=True)

# Padronizar as previsões de volatilidade do GARCH
x_min = volatility_data['Volatility'].min()
x_max = volatility_data['Volatility'].max()
volatility_data['Normalized_Volatility'] = (volatility_data['Volatility'] - x_min) / (x_max - x_min)

# Carregar proxy
proxy_data = pd.read_csv('petr4_vol_forecasts_parkinson_proxy.csv')  # PARKINSON

# Padronizar a proxy
proxy_min = proxy_data['V1'].min()
proxy_max = proxy_data['V1'].max()
proxy_data['Normalized_Proxy'] = (proxy_data['V1'] - proxy_min) / (proxy_max - proxy_min)

# "Alinhando" as previsões e proxy (datas conformes entre as séries)
aligned_length = min(len(proxy_data['Normalized_Proxy']), len(volatility_data['Normalized_Volatility']))
proxy = proxy_data['Normalized_Proxy'].values[-aligned_length:]
data = volatility_data['Normalized_Volatility'].values[-aligned_length:]

# Treinar o modelo GARCH-LSTM
train_and_save_hybrid_predictions(data, proxy, window_size=22, output_csv="teste_predictions_gas_lstm_parkinson.csv") 