import os
import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import keras_tuner as kt            # pip install keras-tuner
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

# ————— Reprodutibilidade —————
os.environ['TF_DETERMINISTIC_OPS'] = '1'
seed = 42
random.seed(seed)
np.random.seed(seed)
keras.utils.set_random_seed(seed)

# ————— Carrega e prepara os dados —————
df = pd.read_csv("volatilidades_previstas_completo_corrigido_GARCH_1_1.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# ————— Configuração do treino inicial —————
WINDOW_SIZE   = 22
initial_train = 1500

# Seleciona as features de input e o target
feature_cols = ['Sigma_GARCH', 'Residuals_garch']  # ou outras que queira testar
series_feat  = df[feature_cols].values      # shape (N, n_features)
series_target = df['Sigma_GARCH'].values    # target = GARCH proxy

# ————— Escalonamento só no treino inicial —————
scaler = StandardScaler()
scaler.fit(series_feat[:initial_train])
scaled_feat = scaler.transform(series_feat)

# ————— Função para criar janelas —————
def make_windows(X, y, window_size):
    XX, yy = [], []
    for i in range(window_size, len(X[:initial_train])):
        XX.append(X[i-window_size:i])
        yy.append(y[i])
    return np.array(XX), np.array(yy)

X_train, y_train = make_windows(scaled_feat, series_target, WINDOW_SIZE)

# ————— Função de construção para o tuner —————
def build_model(hp):
    model = keras.models.Sequential()
    # 1–3 camadas LSTM
    n_layers = hp.Int("n_layers", 1, 3)
    for i in range(n_layers):
        units = hp.Choice(f"units_{i}", [64, 128, 256, 512])
        return_seq = (i < n_layers - 1)
        if i == 0:
            model.add(keras.layers.LSTM(
                units,
                return_sequences=return_seq,
                input_shape=(WINDOW_SIZE, len(feature_cols))
            ))
        else:
            model.add(keras.layers.LSTM(
                units,
                return_sequences=return_seq
            ))
        model.add(keras.layers.Dropout(
            hp.Float(f"dropout_{i}", 0.0, 0.2, step=0.05)
        ))
    model.add(keras.layers.Dense(1, activation="linear"))

    lr = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mse"]
    )
    return model

# ————— Configura o tuner (Hyperband) —————
tuner = kt.Hyperband(
    build_model,
    objective="val_mse",
    max_epochs=30,
    factor=3,
    directory="ktuner_dir2",
    project_name="initial_fit_tuning2"
)

# ————— Callback de EarlyStopping para o search —————
stop_early = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# ————— Executa o tuning —————
tuner.search(
    X_train, y_train,
    epochs=30,
    validation_split=0.1,  # mantém temporalidade (shuffle=False)
    batch_size=64,
    callbacks=[stop_early],
    shuffle=False
)

# ————— Extrai os melhores hiperparâmetros —————
best_hp = tuner.get_best_hyperparameters(1)[0]
print("Melhores HPs no fit inicial:")
for k, v in best_hp.values.items():
    print(f" • {k}: {v}")
