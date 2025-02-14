## Nesse script eu coloco as datas nas previsões dos modelos híbridos; os modelos híbridos retornam apenas as previsões (sem datas)
## O CSV das previsões com as datas se chamam ´teste_petr4_predictions_garch_lstm_parkinson.csv´
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

########################################
########### Proxy: Parkinson ###########
########################################

####################################
## PETR4 - PARKINSON - GARCH-LSTM ##
####################################

# Importando datas
dates = pd.read_csv('petrobras_log_returns.csv')['Date']

# Importando as previsões e proxy
garch_lstm_predictions = pd.read_csv('teste_predictions_garch_lstm_parkinson.csv')['Predicted Values']
garch_lstm_proxy = pd.read_csv('teste_predictions_garch_lstm_parkinson.csv')['Proxy Values']

# Alinhando datas
aligned_dates = dates[-len(garch_lstm_predictions):]  # Selecionar apenas as últimas datas correspondentes às previsões

# Criando dataframe com as datas
petr4_predictions_garch_lstm_parkinson = pd.DataFrame({
    'Date': aligned_dates.values,
    'Predictions': garch_lstm_predictions.values,
    'Proxy_PK': garch_lstm_proxy.values
})

# Salvando o df como CSV
petr4_predictions_garch_lstm_parkinson.to_csv("teste_petr4_predictions_garch_lstm_parkinson.csv", index=False)

# Convertendo a coluna de datas para o tipo Date
petr4_predictions_garch_lstm_parkinson["Date"] = pd.to_datetime(petr4_predictions_garch_lstm_parkinson["Date"])

# Configurar o gráfico
plt.figure(figsize=(12, 6))

# Plotar a proxy
plt.plot(
    petr4_predictions_garch_lstm_parkinson["Date"], 
    petr4_predictions_garch_lstm_parkinson["Proxy_PK"], 
    label="Proxy (Parkinson)", 
    color="blue", 
    alpha=0.7
)

# Plotar as previsões
plt.plot(
    petr4_predictions_garch_lstm_parkinson["Date"], 
    petr4_predictions_garch_lstm_parkinson["Predictions"], 
    label="Predicted Values (GARCH-LSTM)", 
    color="red", 
    linestyle="--", 
    alpha=0.7
)

# Configurar os eixos e título
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.title("Comparison: Proxy vs Predicted Values (GARCH-LSTM)")

# Ajustar o formato do eixo X (datas)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Mostrar rótulos a cada 3 meses
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Formato do rótulo: ano-mês
plt.gcf().autofmt_xdate()  # Rotacionar as datas no eixo X para melhor visualização

# Adicionar legenda e grade
plt.legend()
plt.grid()

# Ajustar layout e salvar o gráfico
plt.tight_layout()
plt.savefig("comparison_proxy_garch_lstm_petr4_parkinson.png")
plt.savefig("comparison_proxy_garch_lstm_petr4_parkinson.pdf")

# Mostrar o gráfico
plt.show()

######################################
## PETR4 - PARKINSON - MSGARCH-LSTM ##
######################################

# Importando datas
dates = pd.read_csv('petrobras_log_returns.csv')['Date']

# Importando as previsões e proxy
msgarch_lstm_predictions = pd.read_csv('teste_predictions_msgarch_lstm_parkinson.csv')['Predicted Values']
msgarch_lstm_proxy = pd.read_csv('teste_predictions_msgarch_lstm_parkinson.csv')['Proxy Values']

# Alinhando datas
aligned_dates = dates[-len(msgarch_lstm_predictions):]  # Selecionar apenas as últimas datas correspondentes às previsões

# Criando dataframe com as datas
petr4_predictions_msgarch_lstm_parkinson = pd.DataFrame({
    'Date': aligned_dates.values,
    'Predictions': msgarch_lstm_predictions.values,
    'Proxy_PK': msgarch_lstm_proxy.values
})

# Salvando o df como CSV
petr4_predictions_msgarch_lstm_parkinson.to_csv("teste_petr4_predictions_msgarch_lstm_parkinson.csv", index=False)

# Convertendo a coluna de datas para o tipo Date
petr4_predictions_msgarch_lstm_parkinson["Date"] = pd.to_datetime(petr4_predictions_msgarch_lstm_parkinson["Date"])

# Configurar o gráfico
plt.figure(figsize=(12, 6))

# Plotar a proxy (PK)
plt.plot(
    petr4_predictions_msgarch_lstm_parkinson["Date"], 
    petr4_predictions_msgarch_lstm_parkinson["Proxy_PK"], 
    label="Proxy (Parkinson)", 
    color="blue", 
    alpha=0.7
)

# Plotar as previsões
plt.plot(
    petr4_predictions_msgarch_lstm_parkinson["Date"], 
    petr4_predictions_msgarch_lstm_parkinson["Predictions"], 
    label="Predicted Values (MSGARCH-LSTM)", 
    color="red", 
    linestyle="--", 
    alpha=0.7
)

# Configurar os eixos e título
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.title("Volatility vs Predicted Values (MSGARCH-LSTM)")
# Ajustar o formato do eixo X (datas)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Mostrar rótulos a cada 3 meses
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Formato do rótulo: ano-mês
plt.gcf().autofmt_xdate()  # Rotacionar as datas no eixo X para melhor visualização

# Adicionar legenda e grade
plt.legend()
plt.grid()

# Ajustar layout e salvar o gráfico
plt.tight_layout()
plt.savefig("comparison_proxy_msgarch_lstm_petr4_parkinson.png")
plt.savefig("comparison_proxy_msgarch_lstm_petr4_parkinson.pdf")

# Mostrar o gráfico
plt.show()

##################################
## PETR4 - PARKINSON - GAS-LSTM ##
##################################

# Importando datas
dates = pd.read_csv('petrobras_log_returns.csv')['Date']

# Importando as previsões e proxy
gas_lstm_predictions = pd.read_csv('teste_predictions_gas_lstm_parkinson.csv')['Predicted Values']
gas_lstm_proxy = pd.read_csv('teste_predictions_gas_lstm_parkinson.csv')['Proxy Values']

# Alinhando datas
aligned_dates = dates[-len(gas_lstm_predictions):]  # Selecionar apenas as últimas datas correspondentes às previsões

# Criando dataframe com as datas
petr4_predictions_gas_lstm_parkinson = pd.DataFrame({
    'Date': aligned_dates.values,
    'Predictions': gas_lstm_predictions.values,
    'Proxy_PK': gas_lstm_proxy.values
})

# Salvando o df como CSV
petr4_predictions_gas_lstm_parkinson.to_csv("teste_petr4_predictions_gas_lstm_parkinson.csv", index=False)

# Convertendo a coluna de datas para o tipo Date
petr4_predictions_gas_lstm_parkinson["Date"] = pd.to_datetime(petr4_predictions_gas_lstm_parkinson["Date"])

# Configurar o gráfico
plt.figure(figsize=(12, 6))

# Plotar a proxy (PK)
plt.plot(
    petr4_predictions_gas_lstm_parkinson["Date"], 
    petr4_predictions_gas_lstm_parkinson["Proxy_PK"], 
    label="Proxy (Parkinson)", 
    color="blue", 
    alpha=0.7
)

# Plotar as previsões
plt.plot(
    petr4_predictions_gas_lstm_parkinson["Date"], 
    petr4_predictions_gas_lstm_parkinson["Predictions"], 
    label="Predicted Values (GAS-LSTM)", 
    color="red", 
    linestyle="--", 
    alpha=0.7
)

# Configurar os eixos e título
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.title("Volatility vs Predicted Values (GAS-LSTM)")
# Ajustar o formato do eixo X (datas)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Mostrar rótulos a cada 3 meses
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Formato do rótulo: ano-mês
plt.gcf().autofmt_xdate()  # Rotacionar as datas no eixo X para melhor visualização

# Adicionar legenda e grade
plt.legend()
plt.grid()

# Ajustar layout e salvar o gráfico
plt.tight_layout()
plt.savefig("comparison_proxy_gas_lstm_petr4_parkinson.png")
plt.savefig("comparison_proxy_gas_lstm_petr4_parkinson.pdf")

# Mostrar o gráfico
plt.show()








