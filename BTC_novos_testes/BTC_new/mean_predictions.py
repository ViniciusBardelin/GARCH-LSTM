import pandas as pd
import numpy as np
import os
from tqdm import tqdm

"""
Script consolidado para processar previsões de modelos híbridos
"""

# Configurações
MODELS = ['garch', 'msgarch', 'gas']
NUM_RUNS = 25
PREDICTIONS_DIR = 'predictions'
MEAN_PREDICTIONS_DIR = 'mean_predictions'
os.makedirs(MEAN_PREDICTIONS_DIR, exist_ok=True)

def create_mean_predictions():
    """Versão corrigida sem ambiguidade de índice/coluna"""
    for model in MODELS:
        print(f"\nProcessando {model.upper()}-LSTM...")
        
        all_data = []
        
        for run in tqdm(range(1, NUM_RUNS + 1), desc="Execuções"):
            file_path = f"{PREDICTIONS_DIR}/{model}/run_{run}_predictions.csv"
            
            if not os.path.exists(file_path):
                print(f"Aviso: Arquivo {file_path} não encontrado")
                continue
                
            try:
                df = pd.read_csv(file_path, parse_dates=['Date'])
                df = df[['Date', 'Predicted', 'Returns_sq']].copy()
                df = df.rename(columns={'Predicted': f'Run_{run}'})
                all_data.append(df)
            except Exception as e:
                print(f"Erro ao processar {file_path}: {str(e)}")
                continue
        
        if not all_data:
            print("Nenhum dado válido encontrado.")
            continue
            
        # Concatenar mantendo 'Date' como coluna
        combined = pd.concat(all_data, axis=0)
        
        # Agrupar por data e calcular estatísticas
        stats = combined.groupby('Date').agg({
            **{f'Run_{i}': 'first' for i in range(1, NUM_RUNS + 1)},
            'Returns_sq': 'first'
        })
        
        # Calcular métricas consolidadas
        run_cols = [col for col in stats.columns if col.startswith('Run_')]
        stats['Mean_Predicted'] = stats[run_cols].mean(axis=1)
        stats['Median_Predicted'] = stats[run_cols].median(axis=1)
        stats['Std_Predicted'] = stats[run_cols].std(axis=1)
        
        # Resetar índice para tornar 'Date' uma coluna
        stats = stats.reset_index()
        
        # Filtrar período de interesse
        start_date = pd.to_datetime('2023-01-31')
        end_date = pd.to_datetime('2024-10-31')
        stats = stats[(stats['Date'] >= start_date) & (stats['Date'] <= end_date)]
        
        # Selecionar colunas finais
        output_cols = ['Date', 'Mean_Predicted', 'Median_Predicted', 'Std_Predicted', 'Returns_sq']
        final_df = stats[output_cols].copy()
        
        # Salvar resultados
        output_file = f"{MEAN_PREDICTIONS_DIR}/{model}_mean_predictions.csv"
        final_df.to_csv(output_file, index=False)
        
        print(f"\nArquivo salvo: {output_file}")
        print(f"Período: {final_df['Date'].min().date()} a {final_df['Date'].max().date()}")
        print(f"Dias úteis: {len(final_df)}")
        print("\nPrimeiras linhas:")
        print(final_df.head(3).to_string(index=False))

if __name__ == "__main__":
    print("=== Consolidação de Previsões ===")
    create_mean_predictions()
    print("\nProcesso concluído com sucesso!")

create_mean_predictions()