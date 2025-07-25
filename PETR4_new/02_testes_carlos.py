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
from tensorflow.keras.layers import Bidirectional, LSTM, Dense

# ————— Reprodutibilidade —————
os.environ['TF_DETERMINISTIC_OPS'] = '1'
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# ———————————————————————— #
# —————— GARCH-LSTM —————— #
# ———————————————————————— #

# ————— Carrega e prepara os dados —————
df = pd.read_csv("sigma2_ajustado_e_previsto_completo.csv")
df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d")

# ————— Adiciona lags de Parkinson como novas features —————
df['Parkinson_lag1'] = df['Parkinson'].shift(1)
df['Parkinson_lag5'] = df['Parkinson'].shift(5)
df[['Parkinson_lag1','Parkinson_lag5']] = df[['Parkinson_lag1','Parkinson_lag5']].bfill()

df = df.sort_values('Date').reset_index(drop=True)

# ————— Configuração do modelo  —————
window_size   = 22 # look-back
initial_train = 1500 # treino inicial
retrain_every = 100 # retreino
feature_cols  = ['Sigma2_GARCH','Parkinson_lag1','Parkinson_lag5', 'Returns'] 

# ————— Prepara features, target e datas —————
features = df[feature_cols].values # shape: (N, n_features)
raw_target = df['Parkinson'].values + 1e-8 # para evitar log(0)
dates = df['Date']
N = len(df)

# ————— Transformação log do target + padronização —————
log_target = np.log(raw_target).reshape(-1,1)
scaler_y   = StandardScaler()
scaler_y.fit(log_target[:initial_train])
scaled_target = scaler_y.transform(log_target).flatten()

# ————— Padronização das features —————
scaler_X = StandardScaler()
scaler_X.fit(features[:initial_train])
scaled_features = scaler_X.transform(features)

# ————— Função para criar janelas —————
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
    y_pred_orig = K.exp(y_pred)  # transformando y_pred de log para escala original
    y_true_orig = K.exp(y_true)
    y_pred_orig = K.maximum(y_pred_orig, 1e-8)
    return K.mean(K.log(y_pred_orig) + y_true_orig / y_pred_orig)

def build_model(input_shape):
    model = keras.models.Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        Bidirectional(LSTM(32, return_sequences=False)),
        keras.layers.Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=qlike_loss,
        metrics=[qlike_loss, 'mse']
    )
    return model

# ————— Treino inicial —————
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

# ————— Resíduos in-sample da LSTM —————
p_log_in = model.predict(X_tr, verbose=0)  # previsões sobre o treino inicial
p_in = scaler_y.inverse_transform(p_log_in)[:, 0]
p_in = np.exp(p_in) - 1e-8 
p_in = np.sqrt(p_in) 
rets_in = df['Returns'].values[window_size:initial_train]
resid_in = (rets_in) / p_in 
dates_in = df['Date'].iloc[window_size:initial_train].reset_index(drop=True)
os.makedirs("Res", exist_ok=True)
pd.DataFrame({
    'Date': dates_in,
    'Residual': resid_in
}).to_csv("Res/GARCH_LSTM_residuals_in_sample_T101_tst_carlos_1_padronizado_COM_MEAN.csv", index=False)

# ————— Walk-forward  —————
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
            epochs=100,
            batch_size=32,
            shuffle=False,
            validation_split=0.1,
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

# ————— Dataframe final de previsões —————
df_pred = pd.DataFrame({
    'Date':       pred_dates,
    'Prediction': preds
})

# Salva CSV
df_pred.to_csv("DF_PREDS/GARCH_LSTM_T101_tst_carlos_1.csv", index=False)

# ————— Métricas ————— #
import numpy as np
from sklearn.metrics import mean_squared_error

# Previsões OoS
oos = pd.read_csv("previsoes_oos_vol_var_es.csv")

# y_true como a série Parkinson OoS
y_true = df['Parkinson'][initial_train:].reset_index(drop=True).values

# Previsões OoS GARCH e GARCH-LSTM, alinhadas com y_true
y_pred_garch = oos['Vol_GARCH'].reset_index(drop=True).values 
y_pred_garch = y_pred_garch ** 2
y_pred_lstm  = df_pred['Prediction'].reset_index(drop=True).values

# Valores válidos (evita zeros/NaN)
eps = 1e-8
valid_garch = (y_pred_garch > eps) & ~np.isnan(y_true)
valid_lstm  = (y_pred_lstm  > eps) & ~np.isnan(y_true)

# Métricas GARCH
y_t_g, y_p_g = y_true[valid_garch], y_pred_garch[valid_garch]
mse_garch    = mean_squared_error(y_t_g, y_p_g)
qlike_garch  = np.mean(
    y_t_g / y_p_g
    - np.log(y_t_g / y_p_g)
    - 1
)

# Métricas LSTM-GARCH
y_t_l, y_p_l = y_true[valid_lstm], y_pred_lstm[valid_lstm]
mse_lstm     = mean_squared_error(y_t_l, y_p_l)
qlike_lstm   = np.mean(
    y_t_l / y_p_l
    - np.log(y_t_l / y_p_l)
    - 1
)

# Resultados
print(f"GARCH Univariado → MSE: {mse_garch:.6e}, QLIKE: {qlike_garch:.6f}")
print(f"GARCH-LSTM      → MSE: {mse_lstm:.6e}, QLIKE: {qlike_lstm:.6f}")

# ————— Gráfico final —————
plt.figure(figsize=(14, 6))

# Parkinson (toda a série, com alpha)
plt.plot(df['Date'], df['Parkinson'],
         label='Parkinson', color='tab:blue', linewidth=1, alpha=0.5)

# GARCH-LSTM (previsão)
plt.plot(df_pred['Date'], df_pred['Prediction'],
         label='GARCH-LSTM', color='tab:green', linewidth=2)

# GARCH (pré-processado como sigma² já!)
plt.plot(df_pred['Date'], y_pred_garch,
         label='GARCH', color='tab:red', linewidth=1.2, linestyle='--')

# Título e eixos
plt.title("Previsão de Volatilidade - GARCH vs GARCH-LSTM", fontsize=14)
plt.xlabel("Data", fontsize=12)
plt.ylabel("Volatilidade (Parkinson)", fontsize=12)
plt.ylim(0, 0.03)

# Legenda posicionada com clareza
plt.legend(loc="upper right", frameon=True, fontsize=10)

# Grade e layout
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig("Resultados/GARCH_LSTM_T101_tst_carlos1.pdf", format="pdf", bbox_inches="tight")
plt.show()

# —————————————————————————— #
# —————— MSGARCH-LSTM —————— #
# —————————————————————————— #

# ————— Carrega e prepara os dados —————
df = pd.read_csv("sigma2_ajustado_e_previsto_completo.csv")
df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d")

# ————— Adiciona lags de Parkinson como novas features —————
df['Parkinson_lag1'] = df['Parkinson'].shift(1)
df['Parkinson_lag5'] = df['Parkinson'].shift(5)
df[['Parkinson_lag1','Parkinson_lag5']] = df[['Parkinson_lag1','Parkinson_lag5']].bfill()

df = df.sort_values('Date').reset_index(drop=True)

# ————— Configuração do modelo  —————
window_size   = 22 # look-back
initial_train = 1500 # treino inicial
retrain_every = 100 # retreino
feature_cols  = ['Sigma2_MSGARCH','Parkinson_lag1','Parkinson_lag5', 'Returns'] 

# ————— Prepara arrays de features, target e datas —————
features = df[feature_cols].values # shape: (N, n_features)
raw_target = df['Parkinson'].values + 1e-8 # para evitar log(0)
dates = df['Date']
N = len(df)

# ————— Transformação log do target + padronização —————
log_target = np.log(raw_target).reshape(-1,1)
scaler_y   = StandardScaler()
scaler_y.fit(log_target[:initial_train])
scaled_target = scaler_y.transform(log_target).flatten()

# ————— Padronização das features —————
scaler_X = StandardScaler()
scaler_X.fit(features[:initial_train])
scaled_features = scaler_X.transform(features)

# ————— Função para criar janelas —————
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
    y_pred_orig = K.exp(y_pred)  # transformando y_pred de log para escala original
    y_true_orig = K.exp(y_true)
    y_pred_orig = K.maximum(y_pred_orig, 1e-8)
    return K.mean(K.log(y_pred_orig) + y_true_orig / y_pred_orig)

def build_model(input_shape):
    model = keras.models.Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        Bidirectional(LSTM(32, return_sequences=False)),
        keras.layers.Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=qlike_loss,
        metrics=[qlike_loss, 'mse']
    )
    return model

# ————— Treino inicial —————
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

# ————— Resíduos in-sample da LSTM —————
p_log_in = model.predict(X_tr, verbose=0)  # previsões sobre o treino inicial
p_in = scaler_y.inverse_transform(p_log_in)[:, 0]
p_in = np.exp(p_in) - 1e-8 
p_in = np.sqrt(p_in)  
rets_in = df['Returns'].values[window_size:initial_train]
resid_in = rets_in / p_in # resíduos padronizados
dates_in = df['Date'].iloc[window_size:initial_train].reset_index(drop=True)
os.makedirs("Res", exist_ok=True)
pd.DataFrame({
    'Date': dates_in,
    'Residual': resid_in
}).to_csv("Res/MSGARCH_LSTM_residuals_in_sample_T101_tst_carlos_1_padronizado_COM_MEAN.csv", index=False)

# ————— Walk-forward  —————
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
            epochs=100,
            batch_size=32,
            shuffle=False,
            validation_split=0.1,
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

# ————— Dataframe final de previsões —————
df_pred = pd.DataFrame({
    'Date':       pred_dates,
    'Prediction': preds
})

# Salva CSV
df_pred.to_csv("DF_PREDS/MSGARCH_LSTM_T101_tst_carlos_1.csv", index=False)

# ————— Métricas ————— #
import numpy as np
from sklearn.metrics import mean_squared_error

# Previsões OoS

oos = pd.read_csv("previsoes_oos_vol_var_es.csv")


# y_true como a série Parkinson OoS
y_true = df['Parkinson'][initial_train:].reset_index(drop=True).values

# Previsões OoS GARCH e GARCH-LSTM, alinhadas com y_true
y_pred_msgarch = oos['Vol_MSGARCH'].reset_index(drop=True).values 
y_pred_msgarch = y_pred_msgarch ** 2

y_pred_lstm  = df_pred['Prediction'].reset_index(drop=True).values

# Valores válidos (evita zeros/NaN)
eps = 1e-8
valid_msgarch = (y_pred_msgarch > eps) & ~np.isnan(y_true)
valid_lstm  = (y_pred_lstm  > eps) & ~np.isnan(y_true)

# Métricas GARCH
y_t_g, y_p_g = y_true[valid_msgarch], y_pred_garch[valid_msgarch]
mse_msgarch    = mean_squared_error(y_t_g, y_p_g)
qlike_msgarch  = np.mean(
    y_t_g / y_p_g
    - np.log(y_t_g / y_p_g)
    - 1
)

# Métricas LSTM-GARCH
y_t_l, y_p_l = y_true[valid_lstm], y_pred_lstm[valid_lstm]
mse_lstm     = mean_squared_error(y_t_l, y_p_l)
qlike_lstm   = np.mean(
    y_t_l / y_p_l
    - np.log(y_t_l / y_p_l)
    - 1
)

# Resultados
print(f"MSGARCH Univariado → MSE: {mse_msgarch:.6e}, QLIKE: {qlike_msgarch:.6f}")
print(f"MSGARCH-LSTM      → MSE: {mse_lstm:.6e}, QLIKE: {qlike_lstm:.6f}")

# ————— Gráfico final —————
plt.figure(figsize=(14, 6))

# Parkinson (toda a série, com alpha)
plt.plot(df['Date'], df['Parkinson'],
         label='Parkinson', color='tab:blue', linewidth=1, alpha=0.5)

# GARCH-LSTM (previsão)
plt.plot(df_pred['Date'], df_pred['Prediction'],
         label='MSGARCH-LSTM', color='tab:green', linewidth=2)

# GARCH (pré-processado como sigma² já!)
plt.plot(df_pred['Date'], y_pred_msgarch,
         label='MSGARCH', color='tab:red', linewidth=1.2, linestyle='--')

# Título e eixos
plt.title("Previsão de Volatilidade - MSGARCH vs MSGARCH-LSTM", fontsize=14)
plt.xlabel("Data", fontsize=12)
plt.ylabel("Volatilidade (Parkinson)", fontsize=12)
plt.ylim(0, 0.03)

# Legenda posicionada com clareza
plt.legend(loc="upper right", frameon=True, fontsize=10)

# Grade e layout
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig("Resultados/MSGARCH_LSTM_T101_tst_carlos1.pdf", format="pdf", bbox_inches="tight")
plt.show()

# —————————————————————— #
# —————— GAS-LSTM —————— #
# —————————————————————— #

# ————— Carrega e prepara os dados —————
df = pd.read_csv("sigma2_ajustado_e_previsto_completo.csv")
df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d")

# ————— Adiciona lags de Parkinson como novas features —————
df['Parkinson_lag1'] = df['Parkinson'].shift(1)
df['Parkinson_lag5'] = df['Parkinson'].shift(5)
df[['Parkinson_lag1','Parkinson_lag5']] = df[['Parkinson_lag1','Parkinson_lag5']].bfill()

df = df.sort_values('Date').reset_index(drop=True)

# ————— Configuração do modelo  —————
window_size   = 22 # look-back
initial_train = 1500 # treino inicial
retrain_every = 100 # retreino
feature_cols  = ['Sigma2_GAS','Parkinson_lag1','Parkinson_lag5', 'Returns'] ##### sem resi

# ————— Prepara arrays de features, target e datas —————
features = df[feature_cols].values # shape: (N, n_features)
raw_target = df['Parkinson'].values + 1e-8 # para evitar log(0)
dates = df['Date']
N = len(df)

# ————— Transformação log do target + padronização —————
log_target = np.log(raw_target).reshape(-1,1)
scaler_y   = StandardScaler()
scaler_y.fit(log_target[:initial_train])
scaled_target = scaler_y.transform(log_target).flatten()

# ————— Padronização das features —————
scaler_X = StandardScaler()
scaler_X.fit(features[:initial_train])
scaled_features = scaler_X.transform(features)

# ————— Função para criar janelas —————
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
    y_pred_orig = K.exp(y_pred)  # transformando y_pred de log para escala original
    y_true_orig = K.exp(y_true)
    y_pred_orig = K.maximum(y_pred_orig, 1e-8)
    return K.mean(K.log(y_pred_orig) + y_true_orig / y_pred_orig)

def build_model(input_shape):
    model = keras.models.Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        Bidirectional(LSTM(32, return_sequences=False)),
        keras.layers.Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=qlike_loss,
        metrics=[qlike_loss, 'mse']
    )
    return model

# ————— Treino inicial —————
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

# ————— Resíduos in-sample da LSTM —————
p_log_in = model.predict(X_tr, verbose=0)  # previsões sobre o treino inicial
p_in = scaler_y.inverse_transform(p_log_in)[:, 0]
p_in = np.exp(p_in) - 1e-8  
p_in = np.sqrt(p_in) 
rets_in = df['Returns'].values[window_size:initial_train]
resid_in = rets_in / p_in # resíduos padronizados
dates_in = df['Date'].iloc[window_size:initial_train].reset_index(drop=True)
os.makedirs("Res", exist_ok=True)
pd.DataFrame({
    'Date': dates_in,
    'Residual': resid_in
}).to_csv("Res/GAS_LSTM_residuals_in_sample_T101_tst_carlos_1_padronizado_COM_MEAN.csv", index=False)

# ————— Walk-forward  —————
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
            epochs=100,
            batch_size=32,
            shuffle=False,
            validation_split=0.1,
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

# ————— Dataframe final de previsões —————
df_pred = pd.DataFrame({
    'Date':       pred_dates,
    'Prediction': preds
})

# Salva CSV
df_pred.to_csv("DF_PREDS/GAS_LSTM_T101_tst_carlos_1.csv", index=False)

# ————— Métricas ————— #
import numpy as np
from sklearn.metrics import mean_squared_error

# Previsões OoS

oos = pd.read_csv("previsoes_oos_vol_var_es.csv")


# y_true como a série Parkinson OoS
y_true = df['Parkinson'][initial_train:].reset_index(drop=True).values

# Previsões OoS GARCH e GARCH-LSTM, alinhadas com y_true
y_pred_gas = oos['Vol_GAS'].reset_index(drop=True).values 
y_pred_gas = y_pred_gas ** 2

y_pred_lstm  = df_pred['Prediction'].reset_index(drop=True).values

# Valores válidos (evita zeros/NaN)
eps = 1e-8
valid_gas = (y_pred_gas > eps) & ~np.isnan(y_true)
valid_lstm  = (y_pred_lstm  > eps) & ~np.isnan(y_true)

# Métricas GARCH
y_t_g, y_p_g = y_true[valid_gas], y_pred_garch[valid_gas]
mse_gas    = mean_squared_error(y_t_g, y_p_g)
qlike_gas  = np.mean(
    y_t_g / y_p_g
    - np.log(y_t_g / y_p_g)
    - 1
)

# Métricas LSTM-GARCH
y_t_l, y_p_l = y_true[valid_lstm], y_pred_lstm[valid_lstm]
mse_lstm     = mean_squared_error(y_t_l, y_p_l)
qlike_lstm   = np.mean(
    y_t_l / y_p_l
    - np.log(y_t_l / y_p_l)
    - 1
)

# Resultados
print(f"GAS Univariado → MSE: {mse_gas:.6e}, QLIKE: {qlike_gas:.6f}")
print(f"GAS-LSTM      → MSE: {mse_lstm:.6e}, QLIKE: {qlike_lstm:.6f}")

# ————— Gráfico final —————
plt.figure(figsize=(14, 6))

# Parkinson (toda a série, com alpha)
plt.plot(df['Date'], df['Parkinson'],
         label='Parkinson', color='tab:blue', linewidth=1, alpha=0.5)

# GARCH-LSTM (previsão)
plt.plot(df_pred['Date'], df_pred['Prediction'],
         label='GAS-LSTM', color='tab:green', linewidth=2)

# GARCH (pré-processado como sigma² já!)
plt.plot(df_pred['Date'], y_pred_msgarch,
         label='GAS', color='tab:red', linewidth=1.2, linestyle='--')

# Título e eixos
plt.title("Previsão de Volatilidade - GAS vs GAS-LSTM", fontsize=14)
plt.xlabel("Data", fontsize=12)
plt.ylabel("Volatilidade (Parkinson)", fontsize=12)
plt.ylim(0, 0.03)

# Legenda posicionada com clareza
plt.legend(loc="upper right", frameon=True, fontsize=10)

# Grade e layout
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig("Resultados/GAS_LSTM_T101_tst_carlos1.pdf", format="pdf", bbox_inches="tight")
plt.show()

# ———————————————————#
# —————— LSTM —————— #
# ———————————————————#

# ————— Carrega e prepara os dados —————
df = pd.read_csv("sigma2_ajustado_e_previsto_completo.csv")
df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d")

# ————— Adiciona lags de Parkinson como novas features —————
df['Parkinson_lag1'] = df['Parkinson'].shift(1)
df['Parkinson_lag5'] = df['Parkinson'].shift(5)
df[['Parkinson_lag1','Parkinson_lag5']] = df[['Parkinson_lag1','Parkinson_lag5']].bfill()

df = df.sort_values('Date').reset_index(drop=True)

# ————— Configuração do modelo  —————
window_size   = 22 # look-back
initial_train = 1500 # treino inicial
retrain_every = 100 # retreino
feature_cols = ['Returns', 'Parkinson_lag1', 'Parkinson_lag5']

# ————— Prepara arrays de features, target e datas —————
features = df[feature_cols].values # shape: (N, n_features)
raw_target = df['Parkinson'].values + 1e-8 # para evitar log(0)
dates = df['Date']
N = len(df)

# ————— Transformação log do target + padronização —————
log_target = np.log(raw_target).reshape(-1,1)
scaler_y   = StandardScaler()
scaler_y.fit(log_target[:initial_train])
scaled_target = scaler_y.transform(log_target).flatten()

# ————— Padronização das features —————
scaler_X = StandardScaler()
scaler_X.fit(features[:initial_train])
scaled_features = scaler_X.transform(features)

# ————— Função para criar janelas —————
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
    y_pred_orig = K.exp(y_pred)  # transformando y_pred de log para escala original
    y_true_orig = K.exp(y_true)
    y_pred_orig = K.maximum(y_pred_orig, 1e-8)
    return K.mean(K.log(y_pred_orig) + y_true_orig / y_pred_orig)

def build_model(input_shape):
    model = keras.models.Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        Bidirectional(LSTM(32, return_sequences=False)),
        keras.layers.Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=qlike_loss,
        metrics=[qlike_loss, 'mse']
    )
    return model

# ————— Treino inicial —————
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

# ————— Resíduos in-sample da LSTM —————
p_log_in = model.predict(X_tr, verbose=0)  # previsões sobre o treino inicial
p_in = scaler_y.inverse_transform(p_log_in)[:, 0]
p_in = np.exp(p_in) - 1e-8  
rets_in = df['Returns'].values[window_size:initial_train]
resid_in = rets_in / p_in # resíduos padronizados
dates_in = df['Date'].iloc[window_size:initial_train].reset_index(drop=True)
os.makedirs("Res", exist_ok=True)
pd.DataFrame({
    'Date': dates_in,
    'Residual': resid_in
}).to_csv("Res/LSTM_puro_residuals_in_sample_T101_tst_carlos_1.csv", index=False)

# ————— Walk-forward  —————
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
            epochs=100,
            batch_size=32,
            shuffle=False,
            validation_split=0.1,
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

# ————— Dataframe final de previsões —————
df_pred = pd.DataFrame({
    'Date':       pred_dates,
    'Prediction': preds
})

# Salva CSV
df_pred.to_csv("DF_PREDS/lstm_puro_T101_tst_carlos_1.csv", index=False)

# ————— Métricas para LSTM puro —————
from sklearn.metrics import mean_squared_error

eps = 1e-8
y_true = df['Parkinson'][initial_train:].reset_index(drop=True).values
y_pred_lstm = df_pred['Prediction'].reset_index(drop=True).values
valid_lstm = (y_pred_lstm > eps) & ~np.isnan(y_true)

y_t_l, y_p_l = y_true[valid_lstm], y_pred_lstm[valid_lstm]
mse_lstm = mean_squared_error(y_t_l, y_p_l)
qlike_lstm = np.mean(
    y_t_l / y_p_l
    - np.log(y_t_l / y_p_l)
    - 1
)

print(f"LSTM puro → MSE: {mse_lstm:.6e}, QLIKE: {qlike_lstm:.6f}")

# ————— Gráfico final —————
plt.figure(figsize=(14, 6))

# Parkinson (toda a série com alpha reduzido)
plt.plot(df['Date'], df['Parkinson'],
         label='Parkinson', color='tab:blue', linewidth=1, alpha=0.5)

# LSTM puro (linha destacada)
plt.plot(df_pred['Date'], df_pred['Prediction'],
         label='LSTM puro', color='tab:green', linewidth=2)

# Título e eixos
plt.title("Previsão de Volatilidade - LSTM puro", fontsize=14)
plt.xlabel("Data", fontsize=12)
plt.ylabel("Volatilidade (Parkinson)", fontsize=12)
plt.ylim(0, 0.03)

# Legenda e grade
plt.legend(loc="upper right", frameon=True, fontsize=10)
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()

# Salvar
plt.savefig("Resultados/LSTM_puro_T101_tst_carlos_1.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.show()
