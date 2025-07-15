import os
import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Bidirectional, LSTM

# ————— Reprodutibilidade —————
os.environ['TF_DETERMINISTIC_OPS'] = '1'
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# ————— Carrega e prepara os dados —————
df = pd.read_csv("vol_GARCH_1_1.csv")
df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d")
df = df.sort_values('Date').reset_index(drop=True)

# ————— Cria covariáveis —————
df['Parkinson_lag1'] = df['Parkinson'].shift(1)
df['Parkinson_lag5'] = df['Parkinson'].shift(5)
df[['Parkinson_lag1','Parkinson_lag5']] = df[['Parkinson_lag1','Parkinson_lag5']].bfill()

# ————— Parâmetros do walk‐forward —————
window_size   = 22
initial_train = 1500
N             = len(df)

# ————— Features e target —————
feature_cols = ['Sigma_GARCH','Parkinson_lag1','Parkinson_lag5','Residuals_garch','Returns']
features     = df[feature_cols].values
raw_target   = df['Parkinson'].values + 1e-8  # offset evita log(0)
dates        = df['Date']

# ————— Escalonamento Min-Max —————
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_X.fit(features[:initial_train])
X_scaled = scaler_X.transform(features)

log_y    = np.log(raw_target).reshape(-1, 1)
scaler_y = MinMaxScaler(feature_range=(0, 1))
scaler_y.fit(log_y[:initial_train])
y_scaled = scaler_y.transform(log_y).flatten()

# ————— Cria janelas para treino in‐sample —————
def make_windows(X, y, size):
    XX, yy = [], []
    for i in range(size, initial_train):
        XX.append(X[i-size:i])
        yy.append(y[i])
    return np.array(XX), np.array(yy)

X_tr, y_tr = make_windows(X_scaled, y_scaled, window_size)

# ————— Callbacks —————
es = EarlyStopping(
    monitor='val_loss', mode='min',
    patience=10, restore_best_weights=True, verbose=1
)
rlp = ReduceLROnPlateau(
    monitor='loss', factor=0.5,
    patience=5, min_lr=1e-6, verbose=1
)

# ————— Loss customizado QLIKE —————
def qlike_loss(y_true, y_pred):
    p = K.exp(y_pred); t = K.exp(y_true)
    p = K.maximum(p, 1e-8)
    return K.mean(K.log(p) + t / p)

# ————— Constrói o modelo —————
def build_model(input_shape):
    m = keras.models.Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        keras.layers.Dropout(0.2),
        Bidirectional(LSTM(32, return_sequences=False)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='linear')
    ])
    m.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=qlike_loss,
        metrics=[qlike_loss, 'mse']
    )
    return m

model = build_model((window_size, len(feature_cols)))

# ————— Treino inicial (cold-start) —————
model.fit(
    X_tr, y_tr,
    epochs=100,
    batch_size=32,
    shuffle=False,
    validation_split=0.1,
    callbacks=[es],
    verbose=1
)

# ————— Resíduos in-sample —————
p_log_in = model.predict(X_tr, verbose=0)
p_in = scaler_y.inverse_transform(p_log_in)
p_in = np.exp(p_in).flatten() - 1e-8
rets_in = df['Returns'].values[window_size:initial_train]
resid_in = rets_in / p_in

dates_in = df['Date'].iloc[window_size:initial_train].reset_index(drop=True)
pd.DataFrame({
    'Date':     dates_in,
    'Residual': resid_in
}).to_csv("Res/LSTM_residuals_in_sample_76.csv", index=False)

# ————— Walk-forward sem retrain —————
preds, pred_dates = [], []
for t in range(initial_train, N):
    window = X_scaled[t-window_size:t]
    x_in   = window.reshape(1, window_size, len(feature_cols))
    p_log  = model.predict(x_in, verbose=0)
    p_orig = np.exp(scaler_y.inverse_transform(p_log)[0,0]) - 1e-8
    preds.append(p_orig)
    pred_dates.append(dates.iloc[t])

df_pred = pd.DataFrame({
    'Date':       pred_dates,
    'Prediction': preds
})
df_pred.to_csv("DF_PREDS/T76.csv", index=False)

# ————— Métricas finais —————
from sklearn.metrics import mean_squared_error

y_true   = raw_target[initial_train:]
y_pred_g = df['Sigma_GARCH'][initial_train:].values
y_pred_l = df_pred['Prediction'].values

mask_g = (y_pred_g > 1e-8) & ~np.isnan(y_true)
mask_l = (y_pred_l > 1e-8) & ~np.isnan(y_true)

mse_g = mean_squared_error(y_true[mask_g], y_pred_g[mask_g])
qlike_g = np.mean(y_true[mask_g]/y_pred_g[mask_g]
                  - np.log(y_true[mask_g]/y_pred_g[mask_g])
                  - 1)

mse_l = mean_squared_error(y_true[mask_l], y_pred_l[mask_l])
qlike_l = np.mean(y_true[mask_l]/y_pred_l[mask_l]
                  - np.log(y_true[mask_l]/y_pred_l[mask_l])
                  - 1)

print(f"GARCH → MSE: {mse_g:.2e}, QLIKE: {qlike_g:.6f}")
print(f"LSTM →  MSE: {mse_l:.2e}, QLIKE: {qlike_l:.6f}")

# ————— Gráfico final —————
plt.figure(figsize=(12,6))
plt.plot(df['Date'][:initial_train], raw_target[:initial_train],
         label='In-sample', color='tab:blue')
plt.plot(df['Date'][initial_train:], raw_target[initial_train:],
         label='OOS true',   color='tab:orange')
plt.plot(df_pred['Date'], df_pred['Prediction'],
         label='LSTM–GARCH', color='tab:green')
plt.title("LSTM–GARCH vs Parkinson")
plt.xlabel("Data"); plt.ylabel("Parkinson")
plt.legend(); plt.grid("--", alpha=0.4)
plt.tight_layout()
plt.savefig("Resultados/T76.pdf", bbox_inches="tight")
plt.show()
