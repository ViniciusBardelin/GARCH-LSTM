#####  Esse script treina os modelos 25x e salva os 25 arquivos de cada configuração

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

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
        LSTM(512, input_shape=(window_size, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(256, return_sequences=False),
        Dropout(0.2),
        Dense(64, activation ='relu'),
        Dense(32, activation = 'relu'),
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
    print(f"Training hybrid model with window size {window_size}...")

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

    return qlike

# Função para treinar e selecionar o melhor modelo com base no QLIKE
def train_and_select_best_model(data, proxy, window_size, num_runs=25):
    best_qlike = np.inf
    best_run = None

    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")
        output_csv = f"run_{run + 1}_predictions.csv"
        qlike = train_and_save_hybrid_predictions(data, proxy, window_size, output_csv)

        if qlike < best_qlike:
            best_qlike = qlike
            best_run = run + 1

    print(f"Best run: {best_run} with QLIKE: {best_qlike}")


################
## GARCH-LSTM ##
################

# Carregar os dados e treinar o modelo
volatility_data = pd.read_csv('act_garch_forecasts.csv')
volatility_data.rename(columns={'Forecasts': 'Volatility'}, inplace=True)

# Carregar proxy
proxy_data = pd.read_csv('act_btc_proxy.csv')

# Convertendo as colunas de data para datetime
volatility_data['Index'] = pd.to_datetime(volatility_data['Index'])
proxy_data['Index'] = pd.to_datetime(proxy_data['Index'])

# Merge para alinhar os dados com base nas datas
merged_data = pd.merge(volatility_data, proxy_data, on='Index', how='inner')

# Padronizar as previsões de volatilidade do GARCH
x_min = merged_data['Volatility'].min()
x_max = merged_data['Volatility'].max()
merged_data['Normalized_Volatility'] = (merged_data['Volatility'] - x_min) / (x_max - x_min)

# Padronizar a proxy
proxy_min = merged_data['Forecasts'].min()
proxy_max = merged_data['Forecasts'].max()
merged_data['Normalized_Proxy'] = (merged_data['Forecasts'] - proxy_min) / (proxy_max - proxy_min)

# Remover NAs
merged_data.dropna(subset=['Normalized_Volatility', 'Normalized_Proxy'], inplace=True)

# Definir variáveis de entrada
proxy = merged_data['Normalized_Proxy'].values
#proxy = merged_data['Forecasts'].values
data = merged_data['Normalized_Volatility'].values

# Treinar e selecionar o melhor modelo com base no QLIKE
train_and_select_best_model(data, proxy, window_size=22, num_runs=25)

##################
## MSGARCH-LSTM ##
##################

# Carregar os dados e treinar o modelo
volatility_data = pd.read_csv('act_msgarch_forecasts.csv')
volatility_data.rename(columns={'Forecasts': 'Volatility'}, inplace=True)

# Carregar proxy
proxy_data = pd.read_csv('act_btc_proxy.csv')

# Convertendo as colunas de data para datetime
volatility_data['Index'] = pd.to_datetime(volatility_data['Index'])
proxy_data['Index'] = pd.to_datetime(proxy_data['Index'])

# Merge para alinhar os dados com base nas datas
merged_data = pd.merge(volatility_data, proxy_data, on='Index', how='inner')

# Padronizar as previsões de volatilidade do MSGARCH
x_min = merged_data['Volatility'].min()
x_max = merged_data['Volatility'].max()
merged_data['Normalized_Volatility'] = (merged_data['Volatility'] - x_min) / (x_max - x_min)

# Padronizar a proxy
proxy_min = merged_data['Forecasts'].min()
proxy_max = merged_data['Forecasts'].max()
merged_data['Normalized_Proxy'] = (merged_data['Forecasts'] - proxy_min) / (proxy_max - proxy_min)

# Remover NAs
merged_data.dropna(subset=['Normalized_Volatility', 'Normalized_Proxy'], inplace=True)

# Definir variáveis de entrada
proxy = merged_data['Normalized_Proxy'].values
#proxy = merged_data['Forecasts'].values
data = merged_data['Normalized_Volatility'].values

# Treinar e selecionar o melhor modelo com base no QLIKE
train_and_select_best_model(data, proxy, window_size=22, num_runs=25)

##############
## GAS-LSTM ##
##############

# Carregar os dados e treinar o modelo
volatility_data = pd.read_csv('act_gas_forecasts.csv')
volatility_data.rename(columns={'Forecasts': 'Volatility'}, inplace=True)

# Carregar proxy
proxy_data = pd.read_csv('act_btc_proxy.csv')

# Convertendo as colunas de data para datetime
volatility_data['Index'] = pd.to_datetime(volatility_data['Index'])
proxy_data['Index'] = pd.to_datetime(proxy_data['Index'])

# Merge para alinhar os dados com base nas datas
merged_data = pd.merge(volatility_data, proxy_data, on='Index', how='inner')

# Padronizar as previsões de volatilidade do MSGARCH
x_min = merged_data['Volatility'].min()
x_max = merged_data['Volatility'].max()
merged_data['Normalized_Volatility'] = (merged_data['Volatility'] - x_min) / (x_max - x_min)

# Padronizar a proxy
proxy_min = merged_data['Forecasts'].min()
proxy_max = merged_data['Forecasts'].max()
merged_data['Normalized_Proxy'] = (merged_data['Forecasts'] - proxy_min) / (proxy_max - proxy_min)

# Remover NAs
merged_data.dropna(subset=['Normalized_Volatility', 'Normalized_Proxy'], inplace=True)

# Definir variáveis de entrada
proxy = merged_data['Normalized_Proxy'].values
#proxy = merged_data['Forecasts'].values
data = merged_data['Normalized_Volatility'].values

# Treinar e selecionar o melhor modelo com base no QLIKE
train_and_select_best_model(data, proxy, window_size=22, num_runs=25)




##################
## MSGARCH-LSTM ## ## usando raiz quadrada das previsões
##################

# Carregar os dados e treinar o modelo
volatility_data = pd.read_csv('act_msgarch_forecasts.csv')
volatility_data.rename(columns={'Forecasts': 'Volatility'}, inplace=True)

# Carregar proxy
proxy_data = pd.read_csv('act_btc_proxy.csv')

# Convertendo as colunas de data para datetime
volatility_data['Index'] = pd.to_datetime(volatility_data['Index'])
proxy_data['Index'] = pd.to_datetime(proxy_data['Index'])

# Merge para alinhar os dados com base nas datas
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

# Remover NAs
merged_data.dropna(subset=['Normalized_Volatility', 'Normalized_Proxy'], inplace=True)

# Definir variáveis de entrada
proxy = merged_data['Normalized_Proxy'].values
#proxy = merged_data['Forecasts'].values
data = merged_data['Normalized_Volatility'].values

# Treinar e selecionar o melhor modelo com base no QLIKE
train_and_select_best_model(data, proxy, window_size=22, num_runs=25)










'''

## MSGARCH-LSTM
# Carregar os dados e treinar o modelo
msgarch = pd.read_csv('hbd_msgarch.csv')

# Padronizar as previsões de volatilidade do GARCH
x_min = msgarch['Predicted_Volatility'].min()
x_max = msgarch['Predicted_Volatility'].max()
msgarch['Normalized_Volatility'] = (msgarch['Predicted_Volatility'] - x_min) / (x_max - x_min)

# Padronizar a proxy
proxy_min = msgarch['Proxy_Values'].min()
proxy_max = msgarch['Proxy_Values'].max()
msgarch['Normalized_Proxy'] = (msgarch['Proxy_Values'] - proxy_min) / (proxy_max - proxy_min)

# Alinhar as previsões e proxy
aligned_length = min(len(msgarch['Normalized_Proxy']), len(msgarch['Normalized_Volatility']))
proxy = msgarch['Normalized_Proxy'].values[-aligned_length:]
data = msgarch['Normalized_Volatility'].values[-aligned_length:]

# Remover NAs
msgarch.dropna(inplace=True)
aligned_length = min(len(proxy_data), len(volatility_data))
proxy = msgarch['Normalized_Proxy'].values[-aligned_length:]
data = msgarch['Normalized_Volatility'].values[-aligned_length:]

# Treinar e selecionar o melhor modelo com base no QLIKE
train_and_select_best_model(data, proxy, window_size=22, num_runs=25)

'''

