# treino incremental (“warm-start”)

import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# ————— Parâmetros de reproducibilidade —————
seed = 42
np.random.seed(seed)
random.seed(seed)
keras.utils.set_random_seed(seed)

# ————— Carrega e prepara os dados —————
df = pd.read_csv("volatilidades_previstas_completo_corrigido_GARCH_1_1.csv")
df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d")
df = df.sort_values('Date').reset_index(drop=True)

series = df['Sigma_GARCH'].values.reshape(-1, 1)
dates  = df['Date']
N = len(series)  # e.g. 2499

# ————— Configuração do walk‐forward —————
window_size   = 22    # look-back de 1 mês
retrain_every = 252  # retrain a cada 1500 pregões (incremental)
train_frac    = 0.75  # 75% dos dados no fit inicial
initial_train = int(N * train_frac)

# ————— Funções auxiliares —————
def build_model(input_shape):
    model = keras.models.Sequential([
        keras.layers.LSTM(256, return_sequences=True,  input_shape=input_shape),
        keras.layers.LSTM(128, return_sequences=False),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mse'])
    return model

def make_windows(arr, size):
    X, y = [], []
    for i in range(size, len(arr)):
        X.append(arr[i-size:i, 0])
        y.append(arr[i, 0])
    X = np.array(X)[..., np.newaxis]
    y = np.array(y)
    return X, y

# ————— Walk-forward com treino incremental —————
scaler = StandardScaler()
preds, pred_dates = [], []

# 1) Fit inicial (cold-start)
scaler.fit(series[:initial_train])
train_scaled = scaler.transform(series[:initial_train])
X_tr, y_tr    = make_windows(train_scaled, window_size)

model = build_model((window_size, 1))
model.fit(
    X_tr, y_tr,
    epochs=50,
    batch_size=64,
    shuffle=False,
    verbose=1
)

# 2) Loop de predição com treino incremental a cada retrain_every
for t in range(initial_train, N):
    # 2.1) Treino incremental: re-fit modelo com TODAS as janelas até o ponto t
    if (t - initial_train) % retrain_every == 0:
        # re-ajusta o scaler até t
        scaler.fit(series[:t])
        ts = scaler.transform(series[:t])
        X_tr, y_tr = make_windows(ts, window_size)
        # não recria o modelo: damos mais fit (“fine-tuning”) sobre pesos atuais
        model.fit(
            X_tr, y_tr,
            epochs=10,         # menos épocas no fine-tuning
            batch_size=64,
            shuffle=False,
            verbose=1
        )

    # 2.2) Previsão one-step-ahead
    window = series[t-window_size:t]
    w_s    = scaler.transform(window)
    x_in   = w_s.reshape(1, window_size, 1)
    p_s    = model.predict(x_in, verbose=0)
    p      = scaler.inverse_transform(p_s)[0, 0]

    preds.append(p)
    pred_dates.append(dates.iloc[t])

# ————— Monta DataFrame e plota —————
df_pred = pd.DataFrame({
    'Date':       pred_dates,
    'GARCH_real': series[initial_train:].flatten(),
    'Prediction': np.array(preds)
})

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['Date'][:initial_train], df['Sigma_GARCH'][:initial_train],
        label='Treino', color='tab:blue')
ax.plot(df_pred['Date'], df_pred['GARCH_real'],
        label='Teste', color='tab:orange')
ax.plot(df_pred['Date'], df_pred['Prediction'],
        label='Previsões', color='tab:red')

ax.set_title("Walk-Forward LSTM-GARCH (treino incremental a cada 252 dias)")
ax.set_xlabel("Data")
ax.set_ylabel("Volatilidade")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.xticks(rotation=45, ha="right")
ax.legend()
plt.tight_layout()
plt.show()

# ————— Métricas —————
y_true = df_pred['GARCH_real']
y_pred = df_pred['Prediction']
valid  = (y_pred.notna() & y_true.notna() & (y_pred > 1e-8))

mse   = mean_squared_error(y_true[valid], y_pred[valid])
qlike = np.mean(np.log(y_pred[valid]) + y_true[valid] / y_pred[valid])

print(f"MSE:   {mse:.6f}")
print(f"QLIKE: {qlike:.6f}")
