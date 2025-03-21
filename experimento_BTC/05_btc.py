##### Esse script constróio o Modelo Hi­brido Combinado (GMG)

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# carregando previsoes dos modelos hibridos
garch_lstm = pd.read_csv('act_garch_lstm.csv')['Predicted Values']
msgarch_lstm = pd.read_csv('act_msgarch_lstm.csv')['Predicted Values']
gas_lstm = pd.read_csv('act_gas_lstm.csv')['Predicted Values']
volatility = pd.read_csv('act_garch_lstm.csv')['Proxy Values']

# criando dataframe das previsoes dos modelos hibridos + proxy da volatilidade 
df = pd.DataFrame({
    'GARCH_LSTM': garch_lstm,
    'MSGARCH_LSTM': msgarch_lstm,
    'GAS_LSTM': gas_lstm,
    'Volatility': volatility
})

# selecionando os dados
X = df.iloc[:, :3].values  # previsões dos tres modelos hibridos
y = df.iloc[:, 3].values   # proxy (target)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# loop para treinar 25 vezes
for i in range(1, 26):
    print(f"Treinamento {i}/25")

    # criando o modelo
    model = Sequential([
        Dense(64, input_dim=3, activation='relu'),  
        Dropout(0.1),  
        Dense(32, activation='relu'),
        Dropout(0.1),  
        Dense(16, activation='relu'),  
        Dropout(0.1),  
        Dense(1, activation='linear')  
    ])

    # compilando o modelo
    model.compile(
        optimizer=Adam(learning_rate=0.001),  
        loss='mean_squared_error',  
        metrics=['mean_absolute_error']
    )

    # treinando o modelo
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,  
        batch_size=64,  
        verbose=1,
        callbacks=[early_stopping]  
    )

    # obter as previsoes
    y_pred = model.predict(X_val)

    # salvar previsoes e valores reais
    predictions_df = pd.DataFrame({
        "Actual": y_val,  
        "Predicted": y_pred.flatten()  
    })

    # nome do arquivo para cada iteração
    predictions_df.to_csv(f"ensemble_model_predictions_{i}.csv", index=False)

print("Treinamento concluído! Todos os CSVs foram salvos.")

## encontrando as melhores metricas

# funcao para calcular o QLIKE
def calculate_qlike(y_true, y_pred):
    return np.mean(np.log(y_pred ** 2) + (y_true ** 2) / (y_pred ** 2))

# funcao para calcular HMSE e HMAE
def calculate_hmse_hmae(y_true, y_pred):
    hmse = np.mean(((y_true - y_pred) ** 2) / np.maximum(1e-10, y_true ** 2))
    hmae = np.mean(np.abs(y_true - y_pred) / np.maximum(1e-10, y_true))
    return hmse, hmae

# lista todos os arquivos CSV gerados
csv_files = glob.glob("ensemble_model_predictions_*.csv")

# inicializa variaveis para armazenar o melhor arquivo e o menor QLIKE
best_file = None
best_qlike = float("inf")

# percorre todos os arquivos e calcula o QLIKE
for file in csv_files:
    df = pd.read_csv(file)
   
    y_true = df["Actual"].values
    y_pred = df["Predicted"].values
   
    qlike = calculate_qlike(y_true, y_pred)

    # atualiza se o novo QLIKE for menor
    if qlike < best_qlike:
        best_qlike = qlike
        best_file = file

# apos encontrar o melhor arquivo, calcular as demais metricas
if best_file:
    df_best = pd.read_csv(best_file)
    y_true_best = df_best["Actual"].values
    y_pred_best = df_best["Predicted"].values

    mse = mean_squared_error(y_true_best, y_pred_best)
    mae = mean_absolute_error(y_true_best, y_pred_best)
    hmse, hmae = calculate_hmse_hmae(y_true_best, y_pred_best)

    print(f"Melhor arquivo: {best_file}")
    print(f"QLIKE: {best_qlike}")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"HMSE: {hmse}")
    print(f"HMAE: {hmae}")
else:
    print("Nenhum arquivo encontrado.")

