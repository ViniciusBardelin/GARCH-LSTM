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
        LSTM(256, input_shape=(window_size, 1), return_sequences=True),
        #Dropout(0.2),
        LSTM(128, return_sequences=False),
        #Dropout(0.2),
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


##########
## LSTM ##
##########

# Carregar os dados e treinar o modelo
volatility_data = pd.read_csv('returns_data.csv')
volatility_data.rename(columns={'Returns': 'Volatility'}, inplace=True)

# Carregar proxy
proxy_data = pd.read_csv('adi_btc_proxy.csv')

# Convertendo as colunas de data para datetime
volatility_data['Date'] = pd.to_datetime(volatility_data['Date'])
volatility_data.rename(columns={'Date': 'Index'}, inplace=True)
proxy_data['Index'] = pd.to_datetime(proxy_data['Index'])

# Merge para alinhar os dados com base nas datas
merged_data = pd.merge(volatility_data, proxy_data, on='Index', how='inner')

# Padronizar as previsões de volatilidade do GARCH
x_min = merged_data['Volatility'].min()
x_max = merged_data['Volatility'].max()
merged_data['Normalized_Volatility'] = (merged_data['Volatility'] - x_min) / (x_max - x_min)

# Padronizar a proxy
proxy_min = merged_data['Parkinson'].min()
proxy_max = merged_data['Parkinson'].max()
merged_data['Normalized_Proxy'] = (merged_data['Parkinson'] - proxy_min) / (proxy_max - proxy_min)

# Remover NAs
merged_data.dropna(subset=['Normalized_Volatility', 'Normalized_Proxy'], inplace=True)

# Definir variáveis de entrada
proxy = merged_data['Normalized_Proxy'].values
#proxy = merged_data['Forecasts'].values
data = merged_data['Normalized_Volatility'].values

# Treinar e selecionar o melhor modelo com base no QLIKE
train_and_select_best_model(data, proxy, window_size=30, num_runs=25)

### Métricas

## Esse script calcula a previsão média dentre todas as 25 iterações

import pandas as pd
import glob
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Obter a lista de arquivos CSV
csv_files = sorted(glob.glob("run_*_predictions.csv"))

# Carregar todos os arquivos e armazená-los em uma lista
dfs = [pd.read_csv(f) for f in csv_files]

# Verificar se todas as colunas Proxy Values são iguais (para segurança)
proxy_values = dfs[0]['Proxy Values']  # Usar a primeira como referência
assert all((df['Proxy Values'] == proxy_values).all() for df in dfs), "Os valores da proxy não são idênticos!"

# Concatenar as previsões em um único dataframe
predicted_values = pd.concat([df['Predicted Values'] for df in dfs], axis=1)

# Calcular a média das previsões
predicted_mean = predicted_values.mean(axis=1)

# Criar o dataframe final com Proxy Values e a média das previsões
final_df = pd.DataFrame({
    "Proxy Values": proxy_values,
    "Predicted Mean": predicted_mean
})

# Salvar em um novo CSV
final_df.to_csv("average_predictions.csv", index=False)

print("Média das previsões calculada e salva como 'average_predictions.csv'.")

## métricas

# Carregar o dataframe final
df = pd.read_csv("average_predictions.csv")

df = df.tail(322).reset_index(drop=True) # selecionando as últimas observações (comparação justa entre os períodos out-of-sample)


# Definir as variáveis
y_true = df["Proxy Values"].values
y_pred = df["Predicted Mean"].values


# Calcular o MSE
mse = mean_squared_error(y_true, y_pred)

# Função para calcular o QLIKE
def calculate_qlike(y_true, y_pred):
    return np.mean(np.log(y_pred ** 2) + (y_true ** 2) / (y_pred ** 2))

y_true = df["Proxy Values"].values
y_pred = df["Predicted Mean"].values

qlike = calculate_qlike(y_true, y_pred)

# Exibir os resultados
print(f"MSE: {mse:.6f}")
print(f"QLIKE: {qlike:.6f}")
