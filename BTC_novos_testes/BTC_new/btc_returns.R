### ----------------------------
### OBTENDO DADOS
### ----------------------------

# Carregar dados
prices = read.csv('BTCUSDT_1d.csv')

# Visualização
prices$OpenTime <- as.Date(prices$OpenTime)
plot(prices$OpenTime, prices$Close, type = 'l', xlab = 'Date', ylab = 'Close')

# Calculando retornos
returns <- diff(log(prices$Close))

df_returns <- data.frame(
  Date = prices$OpenTime[-1],  
  Returns = returns)

# Gerando o CSV
write.csv(df_returns, file = "retornos_btc.csv", row.names = FALSE)


