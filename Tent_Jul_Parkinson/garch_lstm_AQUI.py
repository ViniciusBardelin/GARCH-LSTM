import os
import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

# ————— Reprodutibilidade —————
os.environ['TF_DETERMINISTIC_OPS'] = '1'
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# ————— Carrega e prepara os dados —————
df = pd.read_csv("vol_GARCH_1_1.csv")
df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d")

# ————— Adiciona lags de Parkinson como novas features —————
df['Parkinson_lag1'] = df['Parkinson'].shift(1)
df['Parkinson_lag5'] = df['Parkinson'].shift(5)
df[['Parkinson_lag1','Parkinson_lag5']] = df[['Parkinson_lag1','Parkinson_lag5']].bfill()

df = df.sort_values('Date').reset_index(drop=True)

# ————— Configuração do walk‐forward —————
window_size   = 10     # look-back de 1 mês (~22 pregões)
initial_train = 1500   # treino inicial
retrain_every = 252    # retreina a cada 252 pregões (~1 ano)
feature_cols  = ['Sigma_GARCH','Parkinson_lag1','Parkinson_lag5']

# ————— Prepara arrays de features, target e datas —————
features = df[feature_cols].values                 # shape (N, n_features)
raw_target = df['Parkinson'].values + 1e-8         # offset para evitar log(0)
dates    = df['Date']
N        = len(df)

# ————— Transformação log do target + padronização —————
log_target = np.log(raw_target).reshape(-1,1)
scaler_y   = StandardScaler()
scaler_y.fit(log_target[:initial_train])
scaled_target = scaler_y.transform(log_target).flatten()

# ————— Padronização das features —————
scaler_X = StandardScaler()
scaler_X.fit(features[:initial_train])
scaled_features = scaler_X.transform(features)

# ————— Função para criar janelas multivariadas —————
def make_windows(X_arr, y_arr, size):
    X, y = [], []
    for i in range(size, len(X_arr)):
        X.append(X_arr[i-size:i])
        y.append(y_arr[i])
    return np.array(X), np.array(y)

# ————— Prepara janelas para o treino inicial —————
X_tr, y_tr = make_windows(
    scaled_features[:initial_train],
    scaled_target[:initial_train],
    window_size
)

# ————— Callback de EarlyStopping —————
es = EarlyStopping(monitor='val_qlike_loss', mode='min', patience=10, restore_best_weights=True)

# ————— Loss customizado QLIKE —————
import tensorflow.keras.backend as K
def qlike_loss(y_true, y_pred):
    # y_pred in log-scale ⇒ transform to original
    y_pred_orig = K.exp(y_pred)  # already offset by exp
    y_true_orig = K.exp(y_true)
    # evita zeros
    y_pred_orig = K.maximum(y_pred_orig, 1e-8)
    return K.mean(K.log(y_pred_orig) + y_true_orig / y_pred_orig)

# ————— Função para construir o modelo LSTM —————
def build_model(input_shape):
    model = keras.models.Sequential([
        keras.layers.LSTM(128, return_sequences=True,  input_shape=input_shape),
        #keras.layers.Dropout(0.2),
        keras.layers.LSTM(64, return_sequences=False),
        #keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='linear')
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=qlike_loss,
        metrics=[qlike_loss, 'mse']
    )
    return model

# ————— Treino inicial (cold-start) —————
model = build_model((window_size, len(feature_cols)))
history = model.fit(
    X_tr, y_tr,
    epochs=100,
    batch_size=32,
    shuffle=False,
    validation_split=0.1,
    callbacks=[es],
    verbose=1
)

# ————— Walk-forward com warm-start e retrain periódico —————
preds, pred_dates = [], []
for t in range(initial_train, N):
    # re-treina incrementalmente a cada bloco
    if (t - initial_train) % retrain_every == 0:
        # reescala X e y até o ponto t
        scaler_X.fit(features[:t])
        scaled_features[:t] = scaler_X.transform(features[:t])
        log_target_up = np.log(raw_target[:t]).reshape(-1,1)
        scaler_y.fit(log_target_up)
        scaled_target = scaler_y.transform(np.log(raw_target).reshape(-1,1)).flatten()

        # monta novas janelas e aquece pesos
        X_new, y_new = make_windows(
            scaled_features[:t],
            scaled_target[:t],
            window_size
        )
        model.fit(
            X_new, y_new,
            epochs=10,
            batch_size=32,
            shuffle=False,
            validation_split=0.1,   # usa 10% desse bloco como validação
            callbacks=[es],
            verbose=1
            )

    # previsão one-step-ahead
    window = scaled_features[t-window_size:t]
    x_in   = window.reshape(1, window_size, len(feature_cols))
    p_log  = model.predict(x_in, verbose=0)
    # inverter log + scaler
    p_log_orig = scaler_y.inverse_transform(p_log)[0,0]
    p_orig     = np.exp(p_log_orig) - 1e-8
    preds.append(p_orig)
    pred_dates.append(dates.iloc[t])

# ————— DataFrame final de previsões —————
df_pred = pd.DataFrame({
    'Date':       pred_dates,
    'Prediction': preds
})

# Salva CSV
df_pred.to_csv("DF_PREDS/T39.csv", index=False)

# ————— Métricas ————— #
import numpy as np
from sklearn.metrics import mean_squared_error

# 1) Define y_true como a série “Parkinson” fora da amostra
y_true = df['Parkinson'][initial_train:].reset_index(drop=True).values

# 2) Previsões GARCH e GARCH-LSTM, alinhadas com y_true
y_pred_garch = df['Sigma_GARCH'][initial_train:].reset_index(drop=True).values
y_pred_lstm  = df_pred['Prediction'].reset_index(drop=True).values

# 3) Máscaras de valores válidos (evita zeros/NaN)
eps = 1e-8
valid_garch = (y_pred_garch > eps) & ~np.isnan(y_true)
valid_lstm  = (y_pred_lstm  > eps) & ~np.isnan(y_true)

# 4) Métricas para GARCH
y_t_g, y_p_g = y_true[valid_garch], y_pred_garch[valid_garch]
mse_garch    = mean_squared_error(y_t_g, y_p_g)
qlike_garch  = np.mean(
    y_t_g / y_p_g
    - np.log(y_t_g / y_p_g)
    - 1
)

# 5) Métricas para LSTM-GARCH
y_t_l, y_p_l = y_true[valid_lstm], y_pred_lstm[valid_lstm]
mse_lstm     = mean_squared_error(y_t_l, y_p_l)
qlike_lstm   = np.mean(
    y_t_l / y_p_l
    - np.log(y_t_l / y_p_l)
    - 1
)

# 6) Exibe resultados
print(f"GARCH Univariado → MSE: {mse_garch:.6e}, QLIKE: {qlike_garch:.6f}")
print(f"LSTM–GARCH      → MSE: {mse_lstm:.6e}, QLIKE: {qlike_lstm:.6f}")

# ————— Gráfico final —————
plt.figure(figsize=(12,6))
plt.plot(df['Date'][:initial_train], df['Parkinson'][:initial_train],
         label='Parkinson In-sample', color='tab:blue')
plt.plot(df['Date'][initial_train:], df['Parkinson'][initial_train:],
         label='Parkinson OOS', color='tab:orange')
plt.plot(df_pred['Date'], df_pred['Prediction'],
         label='LSTM–GARCH', color='tab:green')
plt.title("LSTM–GARCH vs Parkinson")
plt.xlabel("Data")
plt.ylabel("Volatilidade (Parkinson)")
plt.legend()
plt.grid("--", alpha=0.4)
plt.tight_layout()
plt.savefig("Resultados/T37.pdf", format="pdf", bbox_inches="tight")
plt.show()
