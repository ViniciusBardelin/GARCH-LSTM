library(rugarch)
library(MSGARCH)
library(GAS)
library(dplyr)
library(xts)
library(zoo)
library(magrittr)

## pre processing

# Data
df = read.csv('returns_data.csv')
df$Date <- as.Date(df$Date)

# Treino + Teste: 18/08/2017 a 31/12/2022
df_train_test <- df %>%
  filter(Date >= as.Date("2017-08-18") & Date <= as.Date("2022-12-31"))

# Validação: 01/01/2023 a 31/10/2024
df_val <- df %>%
  filter(Date >= as.Date("2023-01-01") & Date <= as.Date("2024-10-31"))

## models

#-------#
# GARCH #
#-------#

# Definir janela
window_size <- 1000  
val_dates <- df_val$Date
n_windows <- length(val_dates)

# Vetores para armazenar previsões
garch_predictions <- numeric(n_windows)

# Sliding window: treinando no passado, prevendo no primeiro dia da validação em diante
for (i in 1:n_windows) {
  current_date <- val_dates[i]
  
  # Determinar o índice do último dia da janela de treino
  end_idx <- which(df$Date == current_date) - 1
  start_idx <- end_idx - window_size + 1
  
  # Capturar a janela de treino
  train_data <- df$Returns[start_idx:end_idx]
  
  # Ajustar o modelo
  fit <- ugarchfit(garch_spec, train_data)
  
  # Prever um passo à frente
  forecast <- ugarchforecast(fit, n.ahead = 1)
  garch_predictions[i] <- sqrt(forecast@forecast$sigmaFor)
}

# Criar o xts com as previsões e datas de validação
vol_forecasts_garch <- xts(garch_predictions, order.by = val_dates)
colnames(vol_forecasts_garch) <- c("Forecasts")

# Plotar
plot(val_dates, garch_predictions, type = "l",
     main = "Volatility Forecasts - GARCH",
     xlab = "Date", ylab = "Volatility",
     col = "blue", lwd = 2)

# Salvar como CSV
write.zoo(vol_forecasts_garch, file = "garch_forecasts.csv", sep = ",", col.names = TRUE)

#---------#
# MSGARCH #
#---------#

# Especificação do MSGARCH com 2 regimes e distribuição t-Student
msgarch_spec <- CreateSpec(
  variance.spec = list(model = "sGARCH"),
  distribution.spec = list(distribution = "std"),
  switch.spec = list(do.mix = FALSE, K = 2)  # Modelo com 2 regimes
)

# Janela deslizante
window_size <- 1000
dates_val <- df_val$Date
n_val <- length(dates_val)

# Vetores para previsões e datas
msgarch_predictions <- numeric(n_val)

# Loop sobre datas da validação
for (i in 1:n_val) {
  forecast_date <- dates_val[i]
  
  # Selecionar a janela de treino com base na data
  end_index <- which(df_train_test$Date == forecast_date) - 1
  if (length(end_index) == 0 || end_index < window_size) {
    msgarch_predictions[i] <- NA
    next
  }
  
  start_index <- end_index - window_size + 1
  train_data <- df_train_test$Returns[start_index:end_index]
  
  # Ajustar o modelo MSGARCH
  msgarch_fit <- tryCatch({
    FitML(spec = msgarch_spec, data = train_data)
  }, error = function(e) {
    message(sprintf("Erro ao ajustar o modelo na janela para %s: %s", forecast_date, e$message))
    return(NULL)
  })
  
  # Previsão
  if (!is.null(msgarch_fit)) {
    msgarch_forecast <- tryCatch({
      predict(msgarch_fit, nahead = 1)
    }, error = function(e) {
      message(sprintf("Erro ao prever a volatilidade na data %s: %s", forecast_date, e$message))
      return(NULL)
    })
    
    if (!is.null(msgarch_forecast$vol) && length(msgarch_forecast$vol) > 0) {
      msgarch_predictions[i] <- msgarch_forecast$vol
    } else {
      message(sprintf("Previsão de volatilidade vazia na data %s", forecast_date))
      msgarch_predictions[i] <- NA
    }
  } else {
    msgarch_predictions[i] <- NA
  }
}

# Filtrar previsões válidas
valid_indices <- !is.na(msgarch_predictions)
msgarch_predictions <- msgarch_predictions[valid_indices]
dates_msgarch <- dates_val[valid_indices]

# Converter para Date
dates_msgarch <- as.Date(dates_msgarch)

# Plot
plot(dates_msgarch, msgarch_predictions, type = "l",
     main = "Volatility Forecasts - PETR4 - MSGARCH",
     xlab = "Date", ylab = "Volatility",
     col = "blue", lwd = 2)

# Criar o objeto xts
vol_forecasts_msgarch <- xts(msgarch_predictions, order.by = dates_msgarch)
colnames(vol_forecasts_msgarch) <- c("Forecasts")

# Salvar o objeto xts como CSV
write.zoo(vol_forecasts_msgarch, file = "act_msgarch_forecasts.csv", sep = ",", col.names = TRUE)



##########
## GAS ##
#########

# Especificação do modelo GAS
gas_spec <- UniGASSpec(Dist = "std", ScalingType = "Identity",
                       GASPar = list(scale = TRUE))

# Janela deslizante
window_size <- 1000
n <- nrow(df_returns)
n_windows <- n - window_size

# Vetores para previsões e datas
gas_predictions <- numeric(n_windows)
dates_gas <- df_returns$Date[(window_size + 1):n]

for (i in 1:n_windows) {
  # Dados para a janela atual
  train_data <- returns[i:(i + window_size - 1)]
  
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
colnames(vol_forecasts_gas) <- c("Forecasts")

# Salvar o objeto xts como CSV
write.zoo(vol_forecasts_gas, file = "act_gas_forecasts.csv", sep = ",", col.names = TRUE)

# Gerar o gráfico
plot(dates_gas, gas_predictions, type = "l",
     main = "Previsões de Volatilidade - GAS",
     xlab = "Data", ylab = "Volatilidade",
     col = "blue", lwd = 2)