## Versão 4: usar dados de ações brasileiras; a ideia agora é ajustar os 
## modelos (MS)GARCH e GAS, obter as previsões da volatilidade condicional,
## e aplicar o método FHS para calcular o VaR e o ES.

## Bibliotecas
library(quantmod)
library(rugarch)
library(MSGARCH)
library(gasmodel)
library(dplyr)
library(xts)

## Dados
# Baixar dados da PETR4.SA
getSymbols("PETR4.SA", src = "yahoo", from = "2010-01-01", to = Sys.Date())

# Selecionando os preços de fechamento ajustados
df <- data.frame(Date = index(PETR4.SA), Close = as.numeric(PETR4.SA$PETR4.SA.Adjusted))
plot(df$Date, df$Close, type = "l", main = "Preços de Fechamento - Petrobras", xlab = "Data", ylab = "Preço")

# Log-retornos
log_returns <- diff(log(df$Close)) * 100
log_returns <- na.omit(log_returns)

# Visualizar os log-retornos
plot(df$Date[-1], log_returns, type = "l", main = "Log-retornos da Petrobras", xlab = "Data", ylab = "Log-retorno")

## Modelo GARCH

# Especificar o modelo GARCH(1,1)-t
spec <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(0, 0), include.mean = FALSE),
  distribution.model = "std"
)

# Janela deslizante
window_size <- 1000  
n <- length(log_returns)
n_windows <- n - window_size

# Inicializar um vetor para armazenar as previsões de volatilidade
predictions <- numeric(n_windows)

# Vetor de datas para as previsões
dates <- df$Date[(window_size + 1):n]

# Janela deslizante
for (i in 1:n_windows) {
  # Definir a janela atual
  train_data <- log_returns[i:(i + window_size - 1)]
  
  # Ajustar o modelo GARCH na janela atual
  fit <- ugarchfit(spec, train_data)
  
  # Obter a previsão de volatilidade para o próximo dia
  forecast <- ugarchforecast(fit, n.ahead = 1)
  
  # Armazenar a previsão de volatilidade
  predictions[i] <- sqrt(forecast@forecast$sigmaFor)
}

# Visualizar as previsões de volatilidade
plot(dates, predictions, type = "l", main = "Previsões de Volatilidade - Petrobras", xlab = "Data", ylab = "Volatilidade")

# Criar o objeto xts para análise
vol_forecasts_ts <- xts(predictions, order.by = dates)

# Visualizar o objeto xts
print(head(vol_forecasts_ts))

########################################

## Modelo MSGARCH

# Especificação do MSGARCH com 2 regimes e distribuição t-Student
msgarch_spec <- CreateSpec(
  variance.spec = list(model = "sGARCH"),
  distribution.spec = list(distribution = "std"),
  switch.spec = list(do.mix = FALSE, K = 2)
)

# Janela deslizante
window_size <- 1000 
n <- length(log_returns)
n_windows <- n - window_size

# Vetor para armazenar as previsões de volatilidade
msgarch_predictions <- numeric(n_windows)

# Vetor de datas para as previsões
dates_msgarch <- df$Date[(window_size + 1):n]

# Loop com janela deslizante
for (i in 1:n_windows) {
  # Dados na janela atual
  train_data <- log_returns[i:(i + window_size - 1)]
  
  # Ajustar o modelo MSGARCH na janela atual
  msgarch_fit <- tryCatch({
    FitML(spec = msgarch_spec, data = train_data)
  }, error = function(e) {
    message(sprintf("Erro ao ajustar o modelo na janela %d: %s", i, e$message))
    return(NULL)
  })
  
  # Verificar se o modelo foi ajustado com sucesso
  if (!is.null(msgarch_fit)) {
    # Prever a volatilidade condicional para o próximo período
    msgarch_forecast <- tryCatch({
      predict(msgarch_fit, nahead = 1)
    }, error = function(e) {
      message(sprintf("Erro ao prever a volatilidade na janela %d: %s", i, e$message))
      return(NULL)
    })
    
    # Armazenar a previsão se válida
    if (!is.null(msgarch_forecast$Volatility) && length(msgarch_forecast$Volatility) > 0) {
      msgarch_predictions[i] <- msgarch_forecast$Volatility
    } else {
      message(sprintf("Previsão de volatilidade vazia na janela %d", i))
      msgarch_predictions[i] <- NA
    }
  } else {
    msgarch_predictions[i] <- NA  
  }
}



