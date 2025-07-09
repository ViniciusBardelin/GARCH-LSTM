# re-treina do zero (“cold-start”) a cada 252 pregões.

import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping

# ————— Parâmetros de reproducibilidade —————
seed = 42
np.random.seed(seed)
random.seed(seed)
keras.utils.set_random_seed(seed)

# ————— Callback de EarlyStopping —————
es = EarlyStopping(
    monitor="val_loss",        
    patience=10, # para após 10 épocas sem melhora
    restore_best_weights=True # volta aos pesos da melhor época
)

# ————— Carrega e prepara os dados —————
df = pd.read_csv("volatilidades_previstas_completo_corrigido_GARCH_1_1.csv")
df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d")
df = df.sort_values('Date').reset_index(drop=True)

series = df['Sigma_GARCH'].values.reshape(-1, 1)
dates  = df['Date']
N = len(series)  # esperado = 2499

# ————— Configuração do walk‐forward —————
window_size   = 22         # look-back de 1 mês
retrain_every = 252        # retrain a cada 252 pregões
train_frac    = 0.75       # usar 75% dos dados no primeiro fit

initial_train = 1500

# ————— Funções auxiliares —————
def build_model(input_shape):
    model = keras.models.Sequential([
        keras.layers.LSTM(256, return_sequences=True, 
                          input_shape=input_shape),
        keras.layers.LSTM(128, return_sequences=False),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
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

# ————— Walk-forward com cold-start —————
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
    validation_split=0.1,  # reserva 10% do X_tr/y_tr para validação
    callbacks=[es],
    verbose=1
)
# 2) Loop de predição com retrain a cada retrain_every
for t in range(initial_train, N):
    # 2.1) Re-treina cold-start se atingiu retrain_every
    if (t - initial_train) % retrain_every == 0:
        scaler.fit(series[:t])
        ts = scaler.transform(series[:t])
        X_tr, y_tr = make_windows(ts, window_size)
        
        model = build_model((window_size, 1))
        model.fit(
            X_tr, y_tr,
            epochs=30,
            batch_size=64,
            shuffle=False,
            validation_split=0.1,
            callbacks=[es],        
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

# ————— Monta DataFrame de previsões e plota —————
df_pred = pd.DataFrame({
    'Date':        pred_dates,
    'GARCH_real':  series[initial_train:].flatten(),
    'Prediction':  np.array(preds)
})

# salva sem a coluna de índice no arquivo “predicoes.csv” na pasta de trabalho atual
df_pred.to_csv("DF_PREDS/T3.csv", index=False)

#-- Métricas --#
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
qlike_garch = np.mean(np.log(y_p_g) + y_t_g / y_p_g)

# 5) Métricas para LSTM-GARCH
y_t_l, y_p_l = y_true[valid_lstm], y_pred_lstm[valid_lstm]
mse_lstm     = mean_squared_error(y_t_l, y_p_l)
qlike_lstm  = np.mean(np.log(y_p_l) + y_t_l / y_p_l)

# 6) Exibe resultados
print(f"GARCH Univariado → MSE: {mse_garch:.15f}, QLIKE: {qlike_garch:.6f}")
print(f"LSTM-GARCH      → MSE: {mse_lstm:.15f}, QLIKE: {qlike_lstm:.6f}")

#-- Gráfico LSTM-GARCH vs. Parkinson --#

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

fig, ax = plt.subplots(figsize=(12, 6))

# 1) Verdade in-sample (Parkinson)
ax.plot(
    df['Date'][:initial_train],
    df['Parkinson'][:initial_train],
    label='Parkinson In-sample',
    color='tab:blue'
)

# 2) Verdade out-of-sample (Parkinson)
ax.plot(
    df['Date'][initial_train:],
    df['Parkinson'][initial_train:],
    label='Parkinson Out-of-sample',
    color='tab:orange'
)

# 3) Previsões LSTM-GARCH
ax.plot(
    df_pred['Date'],
    df_pred['Prediction'],
    label='LSTM-GARCH',
    color='tab:green'
)

# formatação
ax.set_title("LSTM-GARCH vs. Parkinson")
ax.set_xlabel("Data")
ax.set_ylabel("Volatilidade (Proxy)")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.xticks(rotation=45, ha="right")
ax.legend()
plt.tight_layout()
plt.show()

#-- Gráfico GARCH vs. Parkinson --#

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ↑ supondo que você já tenha:
# df            – DataFrame com colunas ['Date', 'Parkinson', 'Sigma_GARCH']
# initial_train = int(len(df)*TRAIN_FRAC)

fig, ax = plt.subplots(figsize=(12, 6))

# 1) Verdade in-sample (Parkinson)
ax.plot(
    df['Date'][:initial_train],
    df['Parkinson'][:initial_train],
    label='Parkinson In-sample',
    color='tab:blue'
)

# 2) Verdade out-of-sample (Parkinson)
ax.plot(
    df['Date'][initial_train:],
    df['Parkinson'][initial_train:],
    label='Parkinson Out-of-sample',
    color='tab:orange'
)

# 3) Previsões GARCH Univariado
ax.plot(
    df['Date'][initial_train:],
    df['Sigma_GARCH'][initial_train:],
    label='GARCH Univariado',
    color='tab:red'
)

# formatação
ax.set_title("GARCH Univariado vs. Parkinson")
ax.set_xlabel("Data")
ax.set_ylabel("Volatilidade (Proxy)")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.xticks(rotation=45, ha="right")
ax.legend()
plt.tight_layout()
plt.show()