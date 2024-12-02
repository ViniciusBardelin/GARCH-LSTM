## Conversa 21/11 - Estimar a volatilidade usando uma janela maior e por
## enquanto, só prever volatilidade, sem usar variáveis exógenas, nem 
## parâmetros.

## Bibliotecas
library(rugarch)
library(dplyr)
library(magrittr)
library(xts)
library(openxlsx)

## Dados
setwd("C:/Users/vinic/OneDrive/Área de Trabalho/Estudos/IC/Codes")
df <- read.csv("BTCUSDT_1d.csv") # período 17/08/2017 até 31/10/2024
df %<>% select(-`NA.`)

d1 <- df %>% select(OpenTime, Close) # considerando só fechamento

plot(d1$Close, type = 'l')

## Log retornos
log_returns <- diff(log(d1$Close)) * 100 # Lu et al. 2023
log_returns <- na.omit(log_returns)

plot(log_returns, type = "l", main = "Log-retornos do Bitcoin")


## GARCH

# Especificar o modelo GARCH(1,1)-t
spec <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
                   mean.model = list(armaOrder = c(0, 0), include.mean = FALSE),
                   distribution.model = "std")  # Distribuição t-Student

# Parâmetros para a janela deslizante
window_size <- 1000 
n <- length(log_returns)
n_windows <- n - window_size

# Inicializar um vetor para armazenar as previsões de volatilidade
predictions <- numeric(n_windows)

# Vetor de datas para as previsões (considerando a data da última observação da janela)
dates <- as.Date(d1$OpenTime[(window_size + 1):n])  # Ajuste conforme o formato de data em seu dataframe

# Janela deslizante
for (i in 1:n_windows) {
  # Definir a janela atual
  train_data <- log_returns[i:(i + window_size - 1)]
  
  # Ajustar o modelo GARCH na janela atual
  fit <- ugarchfit(spec, train_data)
  
  # Obter a previsão de volatilidade para o próximo dia
  forecast <- ugarchforecast(fit, n.ahead = 1)
  
  # Armazenar a previsão de volatilidade (estimativa da raiz quadrada da variância condicional)
  predictions[i] <- sqrt(forecast@forecast$sigmaFor)
}

# Visualizar as previsões de volatilidade
plot(predictions, type = "l", main = "Previsões de Volatilidade")

# Criar o objeto xts
vol_forecasts_ts <- xts(predictions, order.by = dates)

'''
# Converter xts para dataframe
vol_forecasts_df <- data.frame(Date = index(vol_forecasts_ts), Volatility = coredata(vol_forecasts_ts))

# Especificar o caminho do arquivo Excel
output_file <- "C:/Users/vinic/OneDrive/Área de Trabalho/Estudos/IC/Codes/vol_forecasts.xlsx"

# Escrever no arquivo Excel
write.xlsx(vol_forecasts_df, output_file)
'''

