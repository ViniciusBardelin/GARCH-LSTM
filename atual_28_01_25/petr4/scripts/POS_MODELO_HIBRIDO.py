# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 18:04:00 2025

@author: vinic
"""

## Nesse script eu coloco as datas nas previsões dos modelos híbridos; os modelos híbridos retornam apenas as previsões x proxy (não tem datas)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


#################################
########### Proxy: SR ###########
#################################


#############################
## PETR4 - SR - GARCH-LSTM ##
#############################

# Importando datas
dates = pd.read_csv('petrobras_log_returns.csv')['Date']

# Importando as previsões e proxy
garch_lstm_predictions = pd.read_csv('predictions_garch_lstm.csv')['Predicted Values']
garch_lstm_proxy = pd.read_csv('predictions_garch_lstm.csv')['Proxy Values']

# Alinhando datas
aligned_dates = dates[-len(garch_lstm_predictions):]  # Selecionar apenas as últimas datas correspondentes às previsões

# Criando dataframe com as datas
petr4_predictions_garch_lstm_sr = pd.DataFrame({
    'Date': aligned_dates.values,
    'Predictions': garch_lstm_predictions.values,
    'Proxy_SR': garch_lstm_proxy.values
})

# Salvando o df como CSV
petr4_predictions_garch_lstm_sr.to_csv("petr4_predictions_garch_lstm.csv", index=False)

# Convertendo a coluna de datas para o tipo Date
petr4_predictions_garch_lstm_sr["Date"] = pd.to_datetime(petr4_predictions_garch_lstm_sr["Date"])

# Configurar o gráfico
plt.figure(figsize=(12, 6))

# Plotar a proxy (SR)
plt.plot(
    petr4_predictions_garch_lstm_sr["Date"], 
    petr4_predictions_garch_lstm_sr["Proxy_SR"], 
    label="Proxy (Squared Returns)", 
    color="blue", 
    alpha=0.7
)

# Plotar as previsões
plt.plot(
    petr4_predictions_garch_lstm_sr["Date"], 
    petr4_predictions_garch_lstm_sr["Predictions"], 
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
plt.savefig("comparison_proxy_garch_lstm_petr4_fixed.png")
plt.savefig("comparison_proxy_garch_lstm_petr4_fixed.pdf")

# Mostrar o gráfico
plt.show()

###############################
## PETR4 - SR - MSGARCH-LSTM ##
###############################

# Importando datas
dates = pd.read_csv('petrobras_log_returns.csv')['Date']

# Importando as previsões e proxy
msgarch_lstm_predictions = pd.read_csv('predictions_msgarch_lstm.csv')['Predicted Values']
msgarch_lstm_proxy = pd.read_csv('predictions_msgarch_lstm.csv')['Proxy Values']

# Alinhando datas
aligned_dates = dates[-len(garch_lstm_pred):]  # Selecionar apenas as últimas datas correspondentes às previsões

# Criando dataframe com as datas
petr4_predictions_msgarch_lstm_sr = pd.DataFrame({
    'Date': aligned_dates.values,
    'Predictions': msgarch_lstm_predictions.values,
    'Proxy_SR': msgarch_lstm_proxy.values
})

# Salvando o df como CSV
petr4_predictions_msgarch_lstm_sr.to_csv("petr4_predictions_msgarch_lstm.csv", index=False)

# Convertendo a coluna de datas para o tipo Date
petr4_predictions_msgarch_lstm_sr["Date"] = pd.to_datetime(petr4_predictions_msgarch_lstm_sr["Date"])

# Configurar o gráfico
plt.figure(figsize=(12, 6))

# Plotar a proxy (SR)
plt.plot(
    petr4_predictions_msgarch_lstm_sr["Date"], 
    petr4_predictions_msgarch_lstm_sr["Proxy_SR"], 
    label="Proxy (Squared Returns)", 
    color="blue", 
    alpha=0.7
)

# Plotar as previsões
plt.plot(
    petr4_predictions_msgarch_lstm_sr["Date"], 
    petr4_predictions_msgarch_lstm_sr["Predictions"], 
    label="Predicted Values (MSGARCH-LSTM)", 
    color="red", 
    linestyle="--", 
    alpha=0.7
)

# Configurar os eixos e título
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.title("Comparison: Proxy vs Predicted Values (MSGARCH-LSTM)")

# Ajustar o formato do eixo X (datas)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Mostrar rótulos a cada 3 meses
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Formato do rótulo: ano-mês
plt.gcf().autofmt_xdate()  # Rotacionar as datas no eixo X para melhor visualização

# Adicionar legenda e grade
plt.legend()
plt.grid()

# Ajustar layout e salvar o gráfico
plt.tight_layout()
plt.savefig("comparison_proxy_msgarch_lstm_petr4_fixed.png")
plt.savefig("comparison_proxy_msgarch_lstm_petr4_fixed.pdf")

# Mostrar o gráfico
plt.show()

###########################
## PETR4 - SR - GAS-LSTM ##
###########################

# Importando datas
dates = pd.read_csv('petrobras_log_returns.csv')['Date']

# Importando as previsões e proxy
gas_lstm_predictions = pd.read_csv('predictions_gas_lstm.csv')['Predicted Values']
gas_lstm_proxy = pd.read_csv('predictions_gas_lstm.csv')['Proxy Values']

# Alinhando datas
aligned_dates = dates[-len(garch_lstm_pred):]  # Selecionar apenas as últimas datas correspondentes às previsões

# Criando dataframe com as datas
petr4_predictions_gas_lstm_sr = pd.DataFrame({
    'Date': aligned_dates.values,
    'Predictions': gas_lstm_predictions.values,
    'Proxy_SR': gas_lstm_proxy.values
})

# Salvando o df como CSV
petr4_predictions_gas_lstm_sr.to_csv("petr4_predictions_gas_lstm.csv", index=False)

# Convertendo a coluna de datas para o tipo Date
petr4_predictions_gas_lstm_sr["Date"] = pd.to_datetime(petr4_predictions_gas_lstm_sr["Date"])

# Configurar o gráfico
plt.figure(figsize=(12, 6))

# Plotar a proxy (SR)
plt.plot(
    petr4_predictions_gas_lstm_sr["Date"], 
    petr4_predictions_gas_lstm_sr["Proxy_SR"], 
    label="Proxy (Squared Returns)", 
    color="blue", 
    alpha=0.7
)

# Plotar as previsões
plt.plot(
    petr4_predictions_gas_lstm_sr["Date"], 
    petr4_predictions_gas_lstm_sr["Predictions"], 
    label="Predicted Values (GAS-LSTM)", 
    color="red", 
    linestyle="--", 
    alpha=0.7
)

# Configurar os eixos e título
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.title("Comparison: Proxy vs Predicted Values (GAS-LSTM)")

# Ajustar o formato do eixo X (datas)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Mostrar rótulos a cada 3 meses
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Formato do rótulo: ano-mês
plt.gcf().autofmt_xdate()  # Rotacionar as datas no eixo X para melhor visualização

# Adicionar legenda e grade
plt.legend()
plt.grid()

# Ajustar layout e salvar o gráfico
plt.tight_layout()
plt.savefig("comparison_proxy_gas_lstm_petr4_fixed.png")
plt.savefig("comparison_proxy_gas_lstm_petr4_fixed.pdf")

# Mostrar o gráfico
plt.show()

###################################
########### Proxy: EWMA ###########
###################################

###############################
## PETR4 - EWMA - GARCH-LSTM ##
###############################

# Importando datas
dates = pd.read_csv('petrobras_log_returns.csv')['Date']

# Importando as previsões e proxy
garch_lstm_predictions = pd.read_csv('predictions_garch_lstm_ewma.csv')['Predicted Values']
garch_lstm_proxy = pd.read_csv('predictions_garch_lstm_ewma.csv')['Proxy Values']

# Alinhando datas
aligned_dates = dates[-len(garch_lstm_predictions):]  # Selecionar apenas as últimas datas correspondentes às previsões

# Criando dataframe com as datas
petr4_predictions_garch_lstm_ewma = pd.DataFrame({
    'Date': aligned_dates.values,
    'Predictions': garch_lstm_predictions.values,
    'Proxy_SR': garch_lstm_proxy.values
})

# Salvando o df como CSV
petr4_predictions_garch_lstm_ewma.to_csv("petr4_predictions_garch_lstm_ewma,csv", index=False)

# Convertendo a coluna de datas para o tipo Date
petr4_predictions_garch_lstm_ewma["Date"] = pd.to_datetime(petr4_predictions_garch_lstm_ewma["Date"])

# Configurar o gráfico
plt.figure(figsize=(12, 6))

# Plotar a proxy (SR)
plt.plot(
    petr4_predictions_garch_lstm_ewma["Date"], 
    petr4_predictions_garch_lstm_ewma["Proxy_SR"], 
    label="Proxy (EWMA)", 
    color="blue", 
    alpha=0.7
)

# Plotar as previsões
plt.plot(
    petr4_predictions_garch_lstm_ewma["Date"], 
    petr4_predictions_garch_lstm_ewma["Predictions"], 
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
plt.savefig("comparison_proxy_garch_lstm_petr4_fixed_ewma.png")
plt.savefig("comparison_proxy_garch_lstm_petr4_fixed_ewma.pdf")

# Mostrar o gráfico
plt.show()








