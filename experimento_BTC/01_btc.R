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

returns <- diff(log(prices$Close)) * 100

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

window_size <- 1000  
n <- nrow(df_returns)
n_windows <- n - window_size

# vectors for dates and forecasts
garch_predictions <- numeric(n_windows)
dates_garch <- df_returns$Date[(window_size + 1):n]

# sliding window
for (i in 1:n_windows) {
  
  train_data <- returns[i:(i + window_size - 1)]
  
  fit <- ugarchfit(garch_spec, train_data)
  
  forecast <- ugarchforecast(fit, n.ahead = 1)
  
  garch_predictions[i] <- sqrt(forecast@forecast$sigmaFor)
}

dates_garch <- as.Date(dates_garch)


plot(dates_garch, garch_predictions, type = "l",
     main = "Volatility Forecasts - GARCH",
     xlab = "Date", ylab = "Volatility",
     col = "blue", lwd = 2)

vol_forecasts_garch <- xts(garch_predictions, order.by = dates_garch)
colnames(vol_forecasts_garch) <- c("Forecasts")

write.zoo(vol_forecasts_garch, file = "act_garch_forecasts.csv", sep = ",", col.names = TRUE)

#---------#
# MSGARCH #
#---------#

msgarch_spec <- CreateSpec(
  variance.spec = list(model = "sGARCH"),
  distribution.spec = list(distribution = "std"),
  switch.spec = list(do.mix = FALSE, K = 2)  # modelo com 2 regimes e distribuição t
)

window_size <- 1000
n <- nrow(df_returns)
n_windows <- n - window_size

msgarch_predictions <- numeric(n_windows)
dates_msgarch <- df_returns$Date[(window_size + 1):n]

for (i in 1:n_windows) {
  train_data <- returns[i:(i + window_size - 1)]
  
  msgarch_fit <- tryCatch({
    FitML(spec = msgarch_spec, data = train_data)
  }, error = function(e) {
    message(sprintf("Erro ao ajustar o modelo na janela %d: %s", i, e$message))
    return(NULL)
  })
  
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

msgarch_predictions <- msgarch_predictions[valid_indices]
dates_msgarch <- dates_msgarch[valid_indices]

dates_msgarch <- as.Date(dates_msgarch)

plot(dates_msgarch, msgarch_predictions, type = "l",
     main = "Volatility Forecasts - PETR4 - MSGARCH",
     xlab = "Date", ylab = "Volatility",
     col = "blue", lwd = 2)

vol_forecasts_msgarch <- xts(msgarch_predictions, order.by = dates_msgarch)
colnames(vol_forecasts_msgarch) <- c("Forecasts")
head(vol_forecasts_msgarch)

write.zoo(vol_forecasts_msgarch, file = "act_msgarch_forecasts.csv", sep = ",", col.names = TRUE)

#-----#
# GAS #
#-----#

gas_spec <- UniGASSpec(Dist = "std", ScalingType = "Identity",
                       GASPar = list(scale = TRUE))

window_size <- 1000
n <- nrow(df_returns)
n_windows <- n - window_size

gas_predictions <- numeric(n_windows)
dates_gas <- df_returns$Date[(window_size + 1):n]

for (i in 1:n_windows) {
  train_data <- returns[i:(i + window_size - 1)]
  
  fit_gas <- UniGASFit(gas_spec, train_data)
  
  gas_forecast <- UniGASFor(fit_gas, H = 1)
  
  gas_predictions[i] <- sqrt(gas_forecast@Forecast$PointForecast[2]) 
}

dates_gas <- as.Date(dates_gas)

vol_forecasts_gas <- xts(gas_predictions, order.by = as.Date(dates_gas))
colnames(vol_forecasts_gas) <- c("Forecasts")

write.zoo(vol_forecasts_gas, file = "act_gas_forecasts.csv", sep = ",", col.names = TRUE)

plot(dates_gas, gas_predictions, type = "l",
     main = "Previsões de Volatilidade - GAS",
     xlab = "Data", ylab = "Volatilidade",
     col = "blue", lwd = 2)
