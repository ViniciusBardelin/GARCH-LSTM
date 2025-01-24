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

# Especificar o modelo GARCH(1,1)-t
spec <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(0, 0), include.mean = FALSE),
  distribution.model = "std"
)

# Janela deslizante
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

# Converter as datas
dates_garch <- as.Date(dates_garch)

# Visualizar as previsões de volatilidade
plot(dates_garch, garch_predictions, type = "l",
     main = "Previsões de Volatilidade - GARCH",
     xlab = "Data", ylab = "Volatilidade",
     col = "blue", lwd = 2)

# Criar o objeto xts para análise
vol_forecasts_garch <- xts(garch_predictions, order.by = dates_garch)

# Salvar o objeto xts como CSV
write.zoo(vol_forecasts_garch, file = "garch_forecasts.csv", sep = ",", col.names = TRUE)

print(head(vol_forecasts_garch))

'''
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
n <- nrow(df_returns)
n_windows <- n - window_size

# Vetores para previsões e datas
msgarch_predictions <- numeric(n_windows)
dates_msgarch <- df_returns$Date[(window_size + 1):n]

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

# Converter as datas
dates_msgarch <- as.Date(dates_msgarch)

# Plot 
plot(dates_msgarch, msgarch_predictions, type = "l",
     main = "Previsões de Volatilidade - MSGARCH",
     xlab = "Data", ylab = "Volatilidade",
     col = "blue", lwd = 2)

# Criar o objeto xts
vol_forecasts_msgarch <- xts(msgarch_predictions, order.by = dates_msgarch)

head(vol_forecasts_msgarch)

# Salvar o objeto xts como CSV
write.zoo(vol_forecasts_msgarch, file = "vol_forecasts_msgarch_atual.csv", sep = ",", col.names = TRUE)

##########
## GAS ##
#########

df_returns <- read.csv("returns_data.csv")
log_returns <- df_returns$Log_Returns

# Especificação do modelo GAS
gas_spec <- UniGASSpec(Dist = "std", ScalingType = "Identity",
                       GASPar = list(scale = TRUE))

# Janela deslizante
window_size <- 1500
n <- nrow(df_returns)
n_windows <- n - window_size

# Vetores para previsões e datas
gas_predictions <- numeric(n_windows)
dates_gas <- df_returns$Date[(window_size + 1):n]

for (i in 1:n_windows) {
  # Dados para a janela atual
  train_data <- log_returns[i:(i + window_size - 1)]
  
  # Ajustar o modelo GAS
  fit_gas <- UniGASFit(gas_spec, train_data)
  
  # Previsão
  gas_forecast <- UniGASFor(fit_gas, H = 1)
  
  # Armazenar a previsão de volatilidade
  gas_predictions[i] <- sqrt(gas_forecast@Forecast$PointForecast[2])  # Raiz da variância (volatilidade)
}

dates_gas <- as.Date(dates_gas)

# Criar o objeto xts para análise
vol_forecasts_gas <- xts(gas_predictions, order.by = as.Date(dates_gas))

# Salvar o objeto xts como CSV
write.zoo(vol_forecasts_gas, file = "vol_forecasts_gas_certo1.csv", sep = ",", col.names = TRUE)

# Gerar o gráfico
plot(dates_gas, gas_predictions, type = "l",
     main = "Previsões de Volatilidade - GAS",
     xlab = "Data", ylab = "Volatilidade",
     col = "blue", lwd = 2)
'''