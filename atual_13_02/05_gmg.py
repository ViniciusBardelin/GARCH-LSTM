## Modelo Hi­brido Combinado (GMG)
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

## Dados

# Carregando previsÃµes dos modelos hÃ­bridos
garch_lstm = pd.read_csv('teste_petr4_predictions_garch_lstm_parkinson.csv')['Predictions']
msgarch_lstm = pd.read_csv('teste_petr4_predictions_msgarch_lstm_parkinson.csv')['Predictions']
gas_lstm = pd.read_csv('teste_petr4_predictions_gas_lstm_parkinson.csv')['Predictions']
volatility = pd.read_csv('teste_petr4_predictions_garch_lstm_parkinson.csv')['Proxy_PK']

# Criando dataframe
df = pd.DataFrame({
    'GARCH_LSTM': garch_lstm,
    'MSGARCH-LSTM': msgarch_lstm,
    'GAS-LSTM': gas_lstm,
    'Volatility': volatility}
)

## Construindo rede progressiva para juntar as previsÃµes

# Definindo o modelo
model = Sequential([
    Dense(64, input_dim=3, activation='relu'),  
    Dropout(0.1),  
    Dense(32, activation='relu'),
    Dropout(0.1),  
    Dense(16, activation='relu'),  
    Dropout(0.1),  
    Dense(1, activation='linear')  
])

# Compilando o modelo
model.compile(
    optimizer=Adam(learning_rate=0.001),  
    loss='mean_squared_error',  
    metrics=['mean_absolute_error']
)

# Resumo do modelo
model.summary()


# Selecionando os dados
X = df.iloc[:, :3].values  # PrevisÃµes dos trÃªs modelos hÃ­bridos
y = df.iloc[:, 3].values  # Proxy

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinando o modelo
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,  
    batch_size=64,  
    verbose=1  
)

# Obter as previsÃµes do conjunto de validaÃ§Ã£o
y_pred = model.predict(X_val)

# Salvar as previsÃµes e os valores reais
predictions_df = pd.DataFrame({
    "Actual": y_val,  # Proxy
    "Predicted": y_pred.flatten()  # PrevisÃµes do modelo
})

# Salvar as previsÃµes em um CSV
predictions_df.to_csv("ensemble_model_predictions_64batch.csv", index=False)

# Calcular mÃ©tricas de erro
mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
hmae = np.mean(np.abs(y_val - y_pred.flatten()) / np.maximum(1e-10, y_val))
hmse = np.mean(((y_val - y_pred.flatten()) ** 2) / np.maximum(1e-10, y_val))
qlike = np.mean(np.log(y_pred ** 2) + (y_val ** 2) / (y_pred ** 2))

# Exibir as mÃ©tricas
metrics_df = pd.DataFrame({
    "Metric": ["MSE", "MAE", "HMAE", "HMSE", "QLIKE"],
    "Value": [mse, mae, hmae, hmse, qlike]
})

metrics_df