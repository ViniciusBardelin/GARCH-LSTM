import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os

'''
Script para treinar modelos híbridos (GARCH-LSTM, MSGARCH-LSTM, GAS-LSTM)
Usando retornos ao quadrado como variável alvo em vez do Parkinson
'''

# Configurações globais
WINDOW_SIZE = 30  # Tamanho da janela temporal
MODELS = ['garch', 'msgarch', 'gas']  # Tipos de modelos híbridos
NUM_RUNS = 25  # Número de execuções por modelo

def create_windows(data, target, window_size):
    """Cria janelas deslizantes para os dados de entrada e saída"""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(target[i + window_size])
    return np.array(X), np.array(y)

def build_lstm_model(window_size):
    """Constroi a arquitetura da LSTM"""
    model = Sequential([
        LSTM(512, input_shape=(window_size, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(256, return_sequences=False),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def train_hybrid_models():
    """Treina todos os modelos híbridos"""
    # Carregar dados de treino
    train_df = pd.read_csv('final_train_data.csv')
    
    for model_type in MODELS:
        print(f"\n=== Treinando {model_type.upper()}-LSTM ===")
        
        # Preparar dados - AGORA USANDO RETORNOS AO QUADRADO
        sigma_col = f'{model_type}_sigma'
        X_train, y_train = create_windows(
            train_df[sigma_col].values,
            train_df['Returns_sq'].values,  # Alvo agora é Returns_sq
            WINDOW_SIZE
        )
        X_train = np.expand_dims(X_train, axis=-1)
        
        # Pasta para salvar os modelos
        os.makedirs(f'models/{model_type}', exist_ok=True)
        
        # Treinar múltiplas vezes
        for run in range(1, NUM_RUNS + 1):
            print(f"Execução {run}/{NUM_RUNS}")
            model = build_lstm_model(WINDOW_SIZE)
            
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            
            model.fit(
                X_train, y_train,
                validation_split=0.2,
                epochs=100,
                batch_size=64,
                callbacks=[early_stop],
                verbose=1
            )
            
            # Salvar modelo
            model.save(f'models/{model_type}/{model_type}_lstm_run_{run}.h5')

def validate_hybrid_models():
    """Valida todos os modelos nos dados de validação"""
    from tensorflow.keras.losses import MeanSquaredError
    from tensorflow.keras.optimizers import Adam

    val_df = pd.read_csv('val_sigma_hat.csv')
    results = []
    
    for model_type in MODELS:
        print(f"\n=== Validando {model_type.upper()}-LSTM ===")
        
        # Preparar dados de validação - AGORA USANDO RETORNOS AO QUADRADO
        sigma_col = f'{model_type}_sigma'
        X_val, y_val = create_windows(
            val_df[sigma_col].values,
            val_df['Returns_sq'].values,  # Alvo agora é Returns_sq
            WINDOW_SIZE
        )
        X_val = np.expand_dims(X_val, axis=-1)
        
        # Pasta para salvar previsões
        os.makedirs(f'predictions/{model_type}', exist_ok=True)
        
        # Validar cada execução
        for run in range(1, NUM_RUNS + 1):
            model_path = f'models/{model_type}/{model_type}_lstm_run_{run}.h5'
            
            model = load_model(model_path, compile=False)
            model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())
            
            y_pred = model.predict(X_val)
            
            # Salvar previsões
            pred_df = pd.DataFrame({
                'Date': val_df['Date'].iloc[WINDOW_SIZE:WINDOW_SIZE+len(y_val)],
                'Returns_sq': y_val,  # Nome da coluna atualizado
                'Predicted': y_pred.flatten()
            })
            pred_df.to_csv(f'predictions/{model_type}/run_{run}_predictions.csv', index=False)
            
            print(f"Execução {run} concluída")

if __name__ == "__main__":
    train_hybrid_models()
    validate_hybrid_models()

train_hybrid_models()
validate_hybrid_models()