library(rugarch)
library(MSGARCH)
library(GAS)
library(dplyr)
library(xts)
library(zoo)
library(magrittr)

#----------------#
# pre processing #
#----------------#

# data
prices = read.csv('BTCUSDT_1d.csv')

sum(prices$NA.)

prices$OpenTime <- as.Date(prices$OpenTime)
plot(prices$OpenTime, prices$Close, type = 'l', xlab = 'Date', ylab = 'Close')

# computing log returns
returns <- diff(log(prices$Close)) * 100

# creating data frame
df_returns <- data.frame(
  Date = prices$OpenTime[-1],  
  Returns = returns)

plot(df_returns$Date, df_returns$Returns, type = 'l')

#--------------#
# garch(1,1)-t #
#--------------#

garch_spec <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(0, 0), include.mean = FALSE),
  distribution.model = "std"
)

# sliding window parameters
window_size <- 1000  
n <- nrow(df_returns)
n_windows <- n - window_size

# vectors for store dates and forecasts
garch_predictions <- numeric(n_windows)
dates_garch <- df_returns$Date[(window_size + 1):n]

# sliding window
for (i in 1:n_windows) {
  # define the actual window
  train_data <- returns[i:(i + window_size - 1)]
  
  # fit the model 
  fit <- ugarchfit(garch_spec, train_data)
  
  # obtaining one-day-ahead forecast
  forecast <- ugarchforecast(fit, n.ahead = 1)
  
  # store the forecast
  garch_predictions[i] <- sqrt(forecast@forecast$sigmaFor)
}

# converter as datas para tipo date
dates_garch <- as.Date(dates_garch)

# plot
plot(dates_garch, garch_predictions, type = "l",
     main = "Volatility Forecasts - GARCH",
     xlab = "Date", ylab = "Volatility",
     col = "blue", lwd = 2)

# criar o objeto xts
vol_forecasts_garch <- xts(garch_predictions, order.by = dates_garch)
colnames(vol_forecasts_garch) <- c("Forecasts")


# salvar o objeto como CSV
write.zoo(vol_forecasts_garch, file = "act_garch_forecasts.csv", sep = ",", col.names = TRUE)

#---------#
# MSGARCH #
#---------#

# especificação do MSGARCH com 2 regimes e distribuição t-Student
msgarch_spec <- CreateSpec(
  variance.spec = list(model = "sGARCH"),
  distribution.spec = list(distribution = "std"),
  switch.spec = list(do.mix = FALSE, K = 2)  # Modelo com 2 regimes
)

# janela deslizante
window_size <- 1000
n <- nrow(df_returns)
n_windows <- n - window_size

# vetores para previsões e datas
msgarch_predictions <- numeric(n_windows)
dates_msgarch <- df_returns$Date[(window_size + 1):n]

# Loop de ajuste
for (i in 1:n_windows) {
  train_data <- returns[i:(i + window_size - 1)]
  
  # ajustar o modelo MSGARCH
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

# identificar índices válidos (previsões não são NA)
valid_indices <- !is.na(msgarch_predictions)
sum(valid_indices)

# filtrar previsões e datas
msgarch_predictions <- msgarch_predictions[valid_indices]
dates_msgarch <- dates_msgarch[valid_indices]

# converter as datas
dates_msgarch <- as.Date(dates_msgarch)

# plot
plot(dates_msgarch, msgarch_predictions, type = "l",
     main = "Volatility Forecasts - PETR4 - MSGARCH",
     xlab = "Date", ylab = "Volatility",
     col = "blue", lwd = 2)

# criar o objeto xts
vol_forecasts_msgarch <- xts(msgarch_predictions, order.by = dates_msgarch)
colnames(vol_forecasts_msgarch) <- c("Forecasts")
head(vol_forecasts_msgarch)

# salvar como CSV
write.zoo(vol_forecasts_msgarch, file = "act_msgarch_forecasts.csv", sep = ",", col.names = TRUE)

#-----#
# GAS #
#-----#

gas_spec <- UniGASSpec(Dist = "std", ScalingType = "Identity",
                       GASPar = list(scale = TRUE))

# janela deslizante
window_size <- 1000
n <- nrow(df_returns)
n_windows <- n - window_size

# vetores para previsões e datas
gas_predictions <- numeric(n_windows)
dates_gas <- df_returns$Date[(window_size + 1):n]

for (i in 1:n_windows) {
  # dados para a janela atual
  train_data <- returns[i:(i + window_size - 1)]
  
  # ajustar o modelo GAS
  fit_gas <- UniGASFit(gas_spec, train_data)
  
  # previsão
  gas_forecast <- UniGASFor(fit_gas, H = 1)
  
  # armazenar a previsão de volatilidade
  gas_predictions[i] <- sqrt(gas_forecast@Forecast$PointForecast[2])  # Raiz da variância (volatilidade)
}

dates_gas <- as.Date(dates_gas)

# criar o objeto xts
vol_forecasts_gas <- xts(gas_predictions, order.by = as.Date(dates_gas))
colnames(vol_forecasts_gas) <- c("Forecasts")

# salvar como CSV
write.zoo(vol_forecasts_gas, file = "act_gas_forecasts.csv", sep = ",", col.names = TRUE)

# gráfico
plot(dates_gas, gas_predictions, type = "l",
     main = "Previsões de Volatilidade - GAS",
     xlab = "Data", ylab = "Volatilidade",
     col = "blue", lwd = 2)
