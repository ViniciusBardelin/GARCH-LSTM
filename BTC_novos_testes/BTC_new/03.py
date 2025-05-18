import pandas as pd
import numpy as np
import os
from tqdm import tqdm

"""
Script para consolidar previsões de modelos híbridos GARCH/MSGARCH/GAS-LSTM

1. Carrega as execuções (25x) de cada modelo
2. Calcula a média diária das previsões para cada modelo
3. Gera um arquivo CSV consolidado por modelo com:
   - Data
   - Previsão média
   - Valor real
4. Verifica consistência do período (31/01/2023 a 31/10/2024)

Saída: Arquivos '[model]_mean_predictions.csv' na pasta 'mean_predictions'
"""

# Configurações
MODELS = ['garch', 'msgarch', 'gas']
NUM_RUNS = 25
PREDICTIONS_DIR = 'predictions'
MEAN_PREDICTIONS_DIR = 'mean_predictions'
os.makedirs(MEAN_PREDICTIONS_DIR, exist_ok=True)

def create_mean_predictions():
    """Cria um CSV com as previsões médias diárias para cada modelo"""
    for model in MODELS:
        print(f"\nProcessando {model.upper()}-LSTM...")
        
        # Lista para armazenar DataFrames temporários
        dfs = []
        
        # Carregar todas as execuções
        for run in tqdm(range(1, NUM_RUNS + 1), desc="Carregando execuções"):
            file_path = f"{PREDICTIONS_DIR}/{model}/run_{run}_predictions.csv"
            
            if not os.path.exists(file_path):
                print(f"\nAviso: Arquivo não encontrado - {file_path}")
                continue
                
            df = pd.read_csv(file_path)
            
            # Converter coluna Date para datetime e garantir formato consistente
            df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
            df = df[['Date', 'Predicted']]  # Manter apenas data e previsão
            df = df.rename(columns={'Predicted': f'Run_{run}'})
            dfs.append(df)
        
        if not dfs:
            print(f"Nenhum arquivo encontrado para {model}")
            continue
        
        # Combinar todos os DataFrames
        combined = dfs[0]
        for df in dfs[1:]:
            combined = combined.merge(df, on='Date', how='outer')
        
        # Verificar período esperado (31/01/2023 a 31/10/2024)
        start_date = pd.to_datetime('2023-01-31')
        end_date = pd.to_datetime('2024-10-31')
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Dias úteis
        
        if not combined['Date'].isin(date_range).all():
            print("\nAviso: Algumas datas estão fora do período esperado")
        
        # Ordenar por data e preencher possíveis valores faltantes
        combined = combined.sort_values('Date')
        combined = combined[combined['Date'].between(start_date, end_date)]
        
        # Calcular média das previsões
        prediction_cols = [col for col in combined.columns if col.startswith('Run_')]
        combined['Mean_Predicted'] = combined[prediction_cols].mean(axis=1)
        
        # Criar DataFrame final
        final_df = combined[['Date', 'Mean_Predicted']].copy()
        
        # Adicionar valores reais do primeiro arquivo (se existir)
        first_file = f"{PREDICTIONS_DIR}/{model}/run_1_predictions.csv"
        if os.path.exists(first_file):
            actual_df = pd.read_csv(first_file)
            actual_df['Date'] = pd.to_datetime(actual_df['Date'], format='%Y-%m-%d')
            final_df = final_df.merge(actual_df[['Date', 'Actual']], on='Date', how='left')
        
        # Garantir ordem cronológica
        final_df = final_df.sort_values('Date')
        
        # Salvar arquivo final
        output_file = f"{MEAN_PREDICTIONS_DIR}/{model}_mean_predictions.csv"
        final_df.to_csv(output_file, index=False)
        
        print(f"\nArquivo criado: {output_file}")
        print("Período coberto:", final_df['Date'].min(), "a", final_df['Date'].max())
        print("Número de dias:", len(final_df))
        print("\nExemplo das primeiras linhas:")
        print(final_df.head(3).to_string(index=False))
        print("\nExemplo das últimas linhas:")
        print(final_df.tail(3).to_string(index=False))

if __name__ == "__main__":
    print("=== Criando arquivos de previsões médias ===")
    create_mean_predictions()
    print("\nProcesso concluído com sucesso!")