library(TTR)
library(quantmod)
library(xts)

# data
prices = read.csv('BTCUSDT_1d.csv')

# data frame in the TTR's format
ohlc_prices <- data.frame(Open = as.numeric(prices$Open),
                 High = as.numeric(prices$High),
                 Low = as.numeric(prices$Low),
                 Close = as.numeric(prices$Close))

Date <- as.Date(prices$OpenTime)

# calculating proxy
parkinson <- volatility(ohlc_prices, calc = "parkinson")

sum(is.na(parkinson))
na_indices <- which(is.na(parkinson))

plot(Date, parkinson, type = 'l')

# generating CSV file
vol_forecasts_parkinson <- xts(parkinson, order.by = Date)
colnames(vol_forecasts_parkinson) <- c("Forecasts")
write.zoo(vol_forecasts_parkinson, file = "act_btc_proxy.csv", sep = ",", col.names = TRUE)
