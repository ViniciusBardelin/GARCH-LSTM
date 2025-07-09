import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping

# ————— Dados —————
data = pd.read_csv("volatilidades_previstas_completo_corrigido_GARCH_1_1.csv")
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

garch = data[['Sigma_GARCH']].values
split = int(len(garch) * 0.95)

# ————— Escalonamento —————
scaler = StandardScaler()
train_scaled = scaler.fit_transform(garch[:split])
test_scaled  = scaler.transform(garch[split - 22:])

# ————— Construção das janelas —————
def build_window(arr, window_size=22):
    X, y = [], []
    for i in range(window_size, len(arr)):
        X.append(arr[i-window_size:i, 0])
        y.append(arr[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = build_window(train_scaled)
X_test,  _       = build_window(test_scaled)

X_train = X_train[..., np.newaxis]
X_test  = X_test[...,  np.newaxis]

# ————— Modelo —————
model = keras.models.Sequential([
    keras.layers.LSTM(256, return_sequences=True,
                      input_shape=(X_train.shape[1], 1)),
    keras.layers.LSTM(128, return_sequences=False),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.summary()

# ————— Treino com EarlyStopping —————
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    shuffle=False,
    validation_split=0.1,
    callbacks=[es]
)

# ————— Previsão —————
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# ————— Plot —————
train = data.iloc[:split]
test  = data.iloc[split:].copy()
test['Pred'] = predictions

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

fig, ax = plt.subplots(figsize=(12,6))
ax.plot(train['Date'], train['Sigma_GARCH'], label='Treino')
ax.plot(test ['Date'], test ['Sigma_GARCH'], label='Teste')
ax.plot(test ['Date'], test ['Pred'], label='Previsões')
ax.set_title("Previsões volatilidade – PETR4")
ax.set_xlabel("Data")
ax.legend()
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

