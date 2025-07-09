import os
import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

# ————— Configurações de reprodutibilidade —————
os.environ['TF_DETERMINISTIC_OPS'] = '1'
seed = 42
random.seed(seed)
np.random.seed(seed)
keras.utils.set_random_seed(seed)

# ————— Carrega e prepara os dados —————
df = pd.read_csv("volatilidades_previstas_completo_corrigido_GARCH_1_1.csv")
df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d")
df = df.sort_values('Date').reset_index(drop=True)

# ————— Configuração do walk‐forward —————
window_size   = 22    # look-back de 1 mês (22 pregões)
initial_train = 1900  # treina com 1500 primeiras observações
retrain_every = 252   # retrena a cada 252 pregões (~1 ano)
feature_cols   = ['Sigma_GARCH', 'Returns']  # features para o LSTM

# Prepara arrays de features e datas
features = df[feature_cols].values  # shape (N, 2)
dates    = df['Date']
N        = len(features)

# ————— Preprocessamento inicial —————
scaler = StandardScaler()
scaler.fit(features[:initial_train])
scaled_features = scaler.transform(features)

# ————— Função para criar janelas multivariadas —————
def make_windows(arr, size):
    X, y = [], []
    for i in range(size, len(arr)):
        X.append(arr[i-size:i, :])  # todas as features
        y.append(arr[i, 0])         # target = Sigma_GARCH (coluna 0)
    return np.array(X), np.array(y)

# ————— Monta janelas para o treino inicial —————
X_tr, y_tr = make_windows(scaled_features[:initial_train], window_size)

# ————— Callback de EarlyStopping —————
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# ————— Função para construir o modelo —————
def build_model(input_shape):
    model = keras.models.Sequential([
        keras.layers.LSTM(256, return_sequences=True, input_shape=input_shape),
        keras.layers.LSTM(128, return_sequences=False),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(1, activation='linear')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mse']
    )
    return model

# ————— Treino inicial (cold-start) —————
model = build_model((window_size, len(feature_cols)))
model.fit(
    X_tr, y_tr,
    epochs=100,
    batch_size=64,
    shuffle=False,
    validation_split=0.1,
    callbacks=[es],
    verbose=1
)

# ————— Walk-forward com retrain a cada retrain_every —————
preds, pred_dates = [], []
for t in range(initial_train, N):
    # retrain cold-start a cada bloco de retrain_every pregões
    if (t - initial_train) % retrain_every == 0:
        # atualiza scaler com dados até t
        scaler.fit(features[:t])
        scaled_features[:t] = scaler.transform(features[:t])
        # recria janelas e modelo
        X_tr, y_tr = make_windows(scaled_features[:t], window_size)
        model = build_model((window_size, len(feature_cols)))
        model.fit(
            X_tr, y_tr,
            epochs=100,
            batch_size=64,
            shuffle=False,
            validation_split=0.1,
            callbacks=[es],
            verbose=1
        )
    # previsão one-step-ahead
    window = scaled_features[t-window_size:t]
    x_in   = window.reshape(1, window_size, len(feature_cols))
    p_s    = model.predict(x_in, verbose=0)
    # inversão apenas do target (Sigma_GARCH)
    inv = scaler.inverse_transform(np.hstack([p_s, np.zeros((1, len(feature_cols)-1))]))
    preds.append(inv[0, 0])
    pred_dates.append(dates.iloc[t])

# ————— Agrupa previsões em DataFrame —————
df_pred = pd.DataFrame({
    'Date':       pred_dates,
    'Prediction': preds
})

# salva sem a coluna de índice no arquivo “predicoes.csv” na pasta de trabalho atual
df_pred.to_csv("DF_PREDS/T14.csv", index=False)

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
qlike_garch = np.mean(
    y_t_g / y_p_g
    - np.log(y_t_g / y_p_g)
    - 1
)

# 5) Métricas para LSTM-GARCH
y_t_l, y_p_l = y_true[valid_lstm], y_pred_lstm[valid_lstm]
mse_lstm     = mean_squared_error(y_t_l, y_p_l)
qlike_lstm = np.mean(
    y_t_l / y_p_l
    - np.log(y_t_l / y_p_l)
    - 1
)

# 6) Resultados
print(f"GARCH Univariado → MSE: {mse_garch:.15f}, QLIKE: {qlike_garch:.6f}")
print(f"LSTM-GARCH      → MSE: {mse_lstm:.15f}, QLIKE: {qlike_lstm:.6f}")

# ————— Gráfico LSTM-GARCH vs. Parkinson ————— #

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    df['Date'][:initial_train],
    df['Parkinson'][:initial_train],
    label='Parkinson In-sample',
    color='tab:blue'
)


ax.plot(
    df['Date'][initial_train:],
    df['Parkinson'][initial_train:],
    label='Parkinson Out-of-sample',
    color='tab:orange'
)

ax.plot(
    df_pred['Date'],
    df_pred['Prediction'],
    label='LSTM-GARCH',
    color='tab:green'
)

ax.set_title("LSTM-GARCH vs. Parkinson")
ax.set_xlabel("Data")
ax.set_ylabel("Volatilidade (Proxy)")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.xticks(rotation=45, ha="right")
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend()
plt.savefig("Resultados/T14.pdf", format="pdf", bbox_inches="tight")
plt.show()

# ————— Gráfico GARCH vs. Parkinson ————— #

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    df['Date'][:initial_train],
    df['Parkinson'][:initial_train],
    label='Parkinson In-sample',
    color='tab:blue'
)

ax.plot(
    df['Date'][initial_train:],
    df['Parkinson'][initial_train:],
    label='Parkinson Out-of-sample',
    color='tab:orange'
)

ax.plot(
    df['Date'][initial_train:],
    df['Sigma_GARCH'][initial_train:],
    label='GARCH Univariado',
    color='tab:red'
)

ax.set_title("GARCH Univariado vs. Parkinson")
ax.set_xlabel("Data")
ax.set_ylabel("Volatilidade (Proxy)")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.xticks(rotation=45, ha="right")
ax.legend()
plt.tight_layout()
plt.show()

