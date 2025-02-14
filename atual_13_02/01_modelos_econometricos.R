## Petrobras 

# Bibliotecas
library(quantmod)
library(rugarch)
library(MSGARCH)
library(GAS)
library(dplyr)
library(xts)
library(zoo)
library(magrittr)

## Dados
df_returns <- read.csv("petrobras_log_returns.csv")
log_returns <- df_returns$Log_Returns

############
## GARCH ##
###########

# Modelo GARCH(1,1)-t
spec <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(0, 0), include.mean = FALSE),
  distribution.model = "std"
)

# Parâmetros janela deslizante
window_size <- 1500  
n <- nrow(df_returns)
n_windows <- n - window_size

# Vetores para previsões e datas
garch_predictions <- numeric(n_windows)
dates_garch <- df_returns$Date[(window_size + 1):n]

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

# Converter as datas para tipo date
dates_garch <- as.Date(dates_garch)

# Plot
plot(dates_garch, garch_predictions, type = "l",
     main = "Volatility Forecasts - PETR4 - GARCH",
     xlab = "Date", ylab = "Volatility",
     col = "blue", lwd = 2)

# Criar o objeto xts
vol_forecasts_garch <- xts(garch_predictions, order.by = dates_garch)

# Salvar o objeto xts como CSV
write.zoo(vol_forecasts_garch, file = "garch_forecasts.csv", sep = ",", col.names = TRUE)

##############
## MSGARCH ##
#############

# MSGARCH-t com 2 regimes
msgarch_spec <- CreateSpec(
  variance.spec = list(model = "sGARCH"),
  distribution.spec = list(distribution = "std"),
  switch.spec = list(do.mix = FALSE, K = 2)
)

# Janela deslizante
window_size <- 1500
n <- nrow(df_returns)
n_windows <- n - window_size

# Vetores para previsões e datas
msgarch_predictions <- numeric(n_windows)
dates_msgarch <- df_returns$Date[(window_size + 1):n]

# Janela deslizante
for (i in 1:n_windows) {
  # Definir a janela atual
  train_data <- log_returns[i:(i + window_size - 1)]
  
  # Ajustar o modelo MSGARCH na janela atual
  msgarch_fit <- FitML(spec = msgarch_spec, data = train_data)
  
  # Obter a previsão de volatilidade para o próximo dia
  msgarch_forecast <- predict(msgarch_fit, nahead = 1)
    
  # Armazenar a previsão de volatilidade 
  msgarch_predictions[i] <- msgarch_forecast$vol
    
}

# Verificar índices válidos
valid_indices <- !is.na(msgarch_predictions)
sum(valid_indices)

# Filtrar previsões e datas
msgarch_predictions <- msgarch_predictions[valid_indices]
dates_msgarch <- dates_msgarch[valid_indices]

# Converter as datas
dates_msgarch <- as.Date(dates_msgarch)

# Plot 
plot(dates_msgarch, msgarch_predictions, type = "l",
     main = "Volatility Forecasts - PETR4 - MSGARCH",
     xlab = "Date", ylab = "Volatility",
     col = "blue", lwd = 2)

# Criar o objeto xts
vol_forecasts_msgarch <- xts(msgarch_predictions, order.by = dates_msgarch)

# Salvar o objeto xts como CSV
write.zoo(vol_forecasts_msgarch, file = "msgarch_forecasts.csv", sep = ",", col.names = TRUE)

##########
## GAS ##
#########

# Especificação do modelo GAS
gas_spec <- UniGASSpec(Dist = "std", ScalingType = "Identity",
                       GASPar = list(scale = TRUE))

# Parâmetros da janela deslizante
window_size <- 1500
n <- nrow(df_returns)
n_windows <- n - window_size

# Vetores para previsões e datas
gas_predictions <- numeric(n_windows)
dates_gas <- df_returns$Date[(window_size + 1):n]

# Janela deslizante
for (i in 1:n_windows) {
  # Dados para a janela atual
  train_data <- log_returns[i:(i + window_size - 1)]
  
  # Ajustar o modelo GAS para a janela atual
  fit_gas <- UniGASFit(gas_spec, train_data)
  
  # Obter a previsão de volatilidade para o próximo dia
  gas_forecast <- UniGASFor(fit_gas, H = 1)
  
  # Armazenar a previsão de volatilidade
  gas_predictions[i] <- sqrt(gas_forecast@Forecast$PointForecast[2])  # Raiz da variância
}

# Converter as datas para tipo date
dates_gas <- as.Date(dates_gas)

# Plot
plot(dates_gas, gas_predictions, type = "l",
     main = "Previsões de Volatilidade - GAS",
     xlab = "Data", ylab = "Volatilidade",
     col = "blue", lwd = 2)

# Criar o objeto xts para análise
vol_forecasts_gas <- xts(gas_predictions, order.by = as.Date(dates_gas))

# Salvar o objeto xts como CSV
write.zoo(vol_forecasts_gas, file = "gas_forecasts.csv", sep = ",", col.names = TRUE)

