### ----------------------------
### BIBLIOTECAS E DADOS
### ----------------------------

library(rugarch)
library(MSGARCH)
library(GAS)
library(dplyr)
library(xts)
library(zoo)
library(ggplot2)
library(tidyr)
library(dplyr)

# Carregar dados
df <- read.csv('retornos_btc.csv')
df$Date <- as.Date(df$Date)

# Definir períodos 
train_start <- as.Date("2017-08-18")
train_end <- as.Date("2022-12-31")  # período completo de treino
val_start <- as.Date("2023-01-01")
val_end <- as.Date("2024-10-31")

# Filtrar dados
train_data <- df %>% filter(Date >= train_start & Date <= train_end)
val_data <- df %>% filter(Date >= val_start & Date <= val_end)

### ----------------------------
### ESPECIFICAÇÕES DOS MODELOS
### ----------------------------

garch_spec <- ugarchspec(
  variance.model = list(model = "sGARCH"),
  mean.model = list(armaOrder = c(1,1)),
  distribution.model = "std"
)

msgarch_spec <- CreateSpec(
  variance.spec = list(model = "sGARCH"),
  distribution.spec = list(distribution = "std"),
  switch.spec = list(do.mix = FALSE, K = 2)
)

gas_spec <- UniGASSpec(
  Dist = "std",
  ScalingType = "Identity",
  GASPar = list(scale = TRUE)
)

### ----------------------------
### FUNÇÃO PARA GERAR SIGMA_HAT
### ----------------------------

generate_sigma_hat <- function(returns, model_type, window_size = 1500) {
  n <- length(returns)
  sigma_hat <- numeric(n) 
  
  for (i in (window_size+1):n) {
    window_returns <- returns[(i-window_size):(i-1)]
    
    if (model_type == "garch") {
      fit <- ugarchfit(garch_spec, window_returns)
      sigma_hat[i] <- as.numeric(sigma(fit)[window_size])^2  # Conversão para σ²
    } 
    else if (model_type == "msgarch") {
      fit <- FitML(msgarch_spec, window_returns)
      sigma_hat[i] <- (predict(fit, nahead = 1)$vol)^2  # Conversão para σ²
    }
    else if (model_type == "gas") {
      fit <- UniGASFit(gas_spec, window_returns)
      forecast <- UniGASFor(fit, H = 1)
      nu <- fit@GASDyn$mTheta[3, 1]  # Graus de liberdade (ν)
      sigma_hat[i] <- forecast@Forecast$PointForecast[2] * nu / (nu - 2)  # σ² ajustada
    }
  }
  
  return(sigma_hat)
}

# Gerar sigma_hat para cada modelo
train_data$garch_sigma <- generate_sigma_hat(train_data$Returns, "garch")
train_data$msgarch_sigma <- generate_sigma_hat(train_data$Returns, "msgarch")
train_data$gas_sigma <- generate_sigma_hat(train_data$Returns, "gas")

# Salvar em CSV
write.csv(train_data, "train_sigma_hat.csv", row.names = FALSE)

### ----------------------------
### VALIDAÇÃO (RETORNOS AO QUADRADO)
### ----------------------------

# Carregar dados
train_sigma_hat <- read.csv('train_sigma_hat.csv')
returns_data <- read.csv('returns_data.csv') %>%  # Arquivo com coluna 'Returns'
  mutate(Returns_sq = Returns^2)  # Criando coluna de retornos ao quadrado

# Juntar valores ajustados dos modelos com retornos²
train_data <- train_sigma_hat %>%
  left_join(returns_data %>% select(Date, Returns_sq), by = "Date")

# Salvando arquivo CSV final do período de treino
write.csv(train_data, "final_train_data.csv", row.names = FALSE)

# Carregar dados de validação
returns_val <- read.csv('returns_data.csv') %>%
  mutate(
    Date = as.Date(Date),
    Returns_sq = Returns^2
  ) %>%
  filter(Date >= as.Date("2023-01-01") & Date <= as.Date("2024-10-31"))

# Criar dataframe de resultados com retornos²
val_results <- data.frame(
  Date = returns_val$Date,
  Returns_sq = returns_val$Returns_sq  # Usando retornos ao quadrado como proxy
)

# Validação one-step-ahead (ajustada)
for (i in 1:nrow(returns_val)) {
  current_date <- returns_val$Date[i]
  
  # Dados até o dia anterior
  history_returns <- df$Returns[df$Date < current_date]
  
  if (length(history_returns) >= 1500) {
    window_returns <- tail(history_returns, 1500)
    
    # GARCH (mantém saída como desvio padrão)
    garch_fit <- ugarchfit(garch_spec, window_returns)
    val_results$garch_sigma[i] <- sigma(ugarchforecast(garch_fit, n.ahead = 1))[1]^2  # Convertendo para variância
    
    # MSGARCH
    msgarch_fit <- FitML(msgarch_spec, window_returns)
    val_results$msgarch_sigma[i] <- (predict(msgarch_fit, nahead = 1)$vol)^2  # Convertendo para variância
    
    # GAS (já ajustado para variância)
    gas_fit <- UniGASFit(gas_spec, window_returns)
    forecast <- UniGASFor(gas_fit, H = 1)
    nu <- gas_fit@GASDyn$mTheta[3, 1]
    val_results$gas_sigma[i] <- forecast@Forecast$PointForecast[2] * nu / (nu - 2)  # Variância já ajustada
  }
}

# Salvar resultados
write.csv(val_results, "val_sigma_hat.csv", row.names = FALSE)