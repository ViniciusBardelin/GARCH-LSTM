## Versão 5: Arrumando o código para o MSGARCH e ajustando GAS.

## Bibliotecas
library(quantmod)
library(rugarch)
library(MSGARCH)
library(GAS)
library(dplyr)
library(xts)
library(zoo)
library(magrittr)
library(keras)

library(tensorflow)
install_tensorflow()


## Dados
getSymbols("PETR4.SA", src = "yahoo", from = "2010-01-01", to = Sys.Date())

# Selecionando os preços de fechamento ajustados
df <- data.frame(Date = index(PETR4.SA), Close = as.numeric(PETR4.SA$PETR4.SA.Adjusted))
plot(df$Date, df$Close, type = "l", main = "Preços de Fechamento - Petrobras", xlab = "Data", ylab = "Preço")

# Log-retornos
log_returns <- diff(log(df$Close)) * 100
log_returns <- na.omit(log_returns)

# Visualizar os log-retornos
plot(df$Date[-1], log_returns, type = "l", main = "Log-retornos da Petrobras", xlab = "Data", ylab = "Log-retorno")

############
## GARCH ##
###########

# Especificar o modelo GARCH(1,1)-t
spec <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(0, 0), include.mean = FALSE),
  distribution.model = "std"
)

# Janela deslizante
window_size <- 1500  
n <- length(log_returns)
n_windows <- n - window_size

# Vetores para previsões e datas
garch_predictions <- numeric(n_windows)
dates_garch <- df$Date[(window_size + 1):n]

# Janela deslizante
for (i in 1:n_windows) {
  # Definir a janela atual
  train_data <- log_returns[i:(i + window_size - 1)]
  
  # Ajustar o modelo GARCH na janela atual
  fit <- ugarchfit(spec, train_data)
  
  # Obter a previsão de volatilidade para o próximo dia
  forecast <- ugarchforecast(fit, n.ahead = 1)
  
  # Armazenar a previsão de volatilidade
  garch_predictions[i] <- sqrt(forecast@forecast$sigmaFor)
}

# Visualizar as previsões de volatilidade
plot(dates_garch, garch_predictions, type = "l",
     main = "Previsões de Volatilidade - GARCH",
     xlab = "Data", ylab = "Volatilidade",
     col = "blue", lwd = 2)

# Criar o objeto xts para análise
vol_forecasts_garch <- xts(garch_predictions, order.by = dates_garch)

# Salvar o objeto xts como CSV
#write.zoo(vol_forecasts_garch, file = "vol_forecasts_garch.csv", sep = ",", col.names = TRUE)

# Visualizar o objeto xts
print(head(vol_forecasts_garch))

##############
## MSGARCH ##
#############

# Especificação do MSGARCH com 2 regimes e distribuição t-Student
msgarch_spec <- CreateSpec(
  variance.spec = list(model = "sGARCH"),
  distribution.spec = list(distribution = "std"),
  switch.spec = list(do.mix = FALSE, K = 2)  # Modelo com 2 regimes
)

# Janela deslizante
window_size <- 1500
n <- length(log_returns)
n_windows <- n - window_size

# Vetores para previsões e datas
msgarch_predictions <- numeric(n_windows)
dates_msgarch <- df$Date[(window_size + 1):n]

# Loop de ajuste
for (i in 1:n_windows) {
  train_data <- log_returns[i:(i + window_size - 1)]
  
  # Ajustar o modelo MSGARCH
  msgarch_fit <- tryCatch({
    FitML(spec = msgarch_spec, data = train_data)
  }, error = function(e) {
    message(sprintf("Erro ao ajustar o modelo na janela %d: %s", i, e$message))
    return(NULL)
  })
  
  # Previsão
  if (!is.null(msgarch_fit)) {
    msgarch_forecast <- tryCatch({
      predict(msgarch_fit, nahead = 1)
    }, error = function(e) {
      message(sprintf("Erro ao prever a volatilidade na janela %d: %s", i, e$message))
      return(NULL)
    })
    
    if (!is.null(msgarch_forecast$vol) && length(msgarch_forecast$vol) > 0) {
      msgarch_predictions[i] <- msgarch_forecast$vol
    } else {
      message(sprintf("Previsão de volatilidade vazia na janela %d", i))
      msgarch_predictions[i] <- NA
    }
  } else {
    msgarch_predictions[i] <- NA
  }
}

# Identificar índices válidos (previsões não são NA)
valid_indices <- !is.na(msgarch_predictions)
sum(valid_indices)

# Filtrar previsões e datas
msgarch_predictions <- msgarch_predictions[valid_indices]
dates_msgarch <- dates_msgarch[valid_indices]

# Plot das previsões de volatilidade
plot(dates_msgarch, msgarch_predictions, type = "l",
     main = "Previsões de Volatilidade - MSGARCH",
     xlab = "Data", ylab = "Volatilidade",
     col = "blue", lwd = 2)

# Criar o objeto xts
vol_forecasts_msgarch <- xts(msgarch_predictions, order.by = dates_msgarch)

# Visualizar as primeiras entradas do objeto xts
head(vol_forecasts_msgarch)

# Salvar o objeto xts como CSV
# write.zoo(vol_forecasts_msgarch, file = "vol_forecasts_msgarch.csv", sep = ",", col.names = TRUE)

##########
## GAS ##
#########

# Especificação do modelo GAS
gas_spec <- UniGASSpec(Dist = "std", ScalingType = "Identity", GASPar = list(scale = TRUE))

# Previsões com janela deslizante usando UniGASRoll
gas_roll <- UniGASRoll(
  data = log_returns,
  GASSpec = gas_spec,
  ForecastLength = n - window_size,  # período out-of-sample
  RefitEvery = 1,                   # reestimação a cada passo
  RefitWindow = "moving"            # janela móvel
)

# Extrair os dados da 4ª coluna (previsões da volatilidade)
gas_volatility_predictions <- gas_roll@Forecast$Moments[, 4]

# Dataframe com datas e previsões
gas_forecast_df <- data.frame(
  Date = dates_gas,  # Ensure dates_gas matches the length of the predictions
  GAS_Volatility = gas_volatility_predictions
)

# Gerando CSV
#write.csv(gas_forecast_df, "gas_volatility_predictions.csv", row.names = FALSE)

# Gráfico
plot(dates_gas, gas_volatility_predictions, type = "l",
     main = "Volatility Predictions - GAS", xlab = "Date", ylab = "Volatility")

############################
## Ensemble GARCH + LSTM ##
###########################

# Carregar as previsões de volatilidade de cada modelo
garch_volatility <- read.csv("vol_forecasts_garch.csv")$V1
msgarch_volatility <- read.csv("vol_forecasts_msgarch.csv")$V1
gas_volatility <- read.csv("gas_volatility_predictions.csv")$GAS_Volatility

# Criar um data frame com os dados
ensemble_data <- data.frame(
  GARCH = garch_volatility,
  MSGARCH = msgarch_volatility,
  GAS = gas_volatility
)

# Gerando CSV
#write.csv(ensemble_data, "ensemble_data.csv", row.names = FALSE)
