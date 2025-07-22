# ----------------------------
# Bibliotecas e dados
# ----------------------------

library(rugarch)
library(MSGARCH)
library(GAS)
library(dplyr)
library(tidyr)
library(Metrics)

# Carregar dados
df <- read.csv("petr4_returns.csv")
df$Data <- as.Date(df$Data)
returns <- df$log_return
N <- length(returns)
n_ins <- 1500
window_size <- 1500

# Especificações dos modelos
garch_spec <- ugarchspec(
  variance.model = list(model= "sGARCH", garchOrder = c(1,1)),
  mean.model = list(armaOrder  = c(0,0), include.mean = FALSE),
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
  GASPar = list(locate = FALSE, scale = TRUE, shape = FALSE)
)

# Valores ajustados e previsões 
generate_sigma_hat_completo <- function(returns, model_type, window_size = 1500) {
  n <- length(returns)
  sigma_hat_adjusted <- rep(NA_real_, n)
  sigma_hat_forecast <- rep(NA_real_, n)
  
  # --- Ajuste inicial in-sample ---
  initial_window <- returns[1:window_size]
  mu0 <- mean(initial_window, na.rm = TRUE)
  centered0 <- initial_window - mu0
  
  fit_tudo <- switch(model_type,
                     "garch"   = ugarchfit(garch_spec, centered0),
                     "msgarch" = FitML(msgarch_spec, centered0),
                     "gas"     = UniGASFit(gas_spec, centered0)
  )
  
  sigma_hat_adjusted[1:window_size] <- switch(model_type,
                                              "garch"   = as.numeric(sigma(fit_tudo))^2,
                                              "msgarch" = as.numeric(Volatility(fit_tudo))^2,
                                              "gas"     = {
                                                vol_in_sample <- getMoments(fit_tudo)[, "M2"]
                                                nu <- fit_tudo@GASDyn$mTheta[3, 1]
                                                vol_in_sample * nu / (nu - 2)
                                              }
  )
  
  sigma_hat_forecast[1:window_size] <- sigma_hat_adjusted[1:window_size]
  
  # --- Rolling forecast OoS ---
  for (i in (window_size + 1):n) {
    w <- returns[(i - window_size):(i - 1)]
    mu <- mean(w, na.rm = TRUE)
    wc <- w - mu
    
    fit <- switch(model_type,
                  "garch"   = ugarchfit(garch_spec, wc),
                  "msgarch" = FitML(msgarch_spec, wc),
                  "gas"     = UniGASFit(gas_spec, wc)
    )
    
    # Valor ajustado em t
    sigma_hat_adjusted[i] <- switch(model_type,
                                    "garch"   = as.numeric(sigma(fit)[window_size])^2,
                                    "msgarch" = as.numeric(Volatility(fit))[window_size]^2,
                                    "gas"     = {
                                      m2 <- getMoments(fit)[, "M2"]
                                      nu <- fit@GASDyn$mTheta[3, 1]
                                      m2[window_size] * nu / (nu - 2)
                                    }
    )
    
    # Previsão para t+1
    sigma_hat_forecast[i] <- switch(model_type,
                                    "garch" = {
                                      fc <- ugarchforecast(fit, n.ahead = 1)
                                      as.numeric(sigma(fc))^2
                                    },
                                    "msgarch" = {
                                      (predict(fit, nahead = 1)$vol)^2
                                    },
                                    "gas" = {
                                      fc <- UniGASFor(fit, H = 1)
                                      nu <- fit@GASDyn$mTheta[3, 1]
                                      fc@Forecast$PointForecast[2] * nu / (nu - 2)
                                    }
    )
  }
  
  return(data.frame(
    Index = 1:n,
    Sigma_Adjusted = sigma_hat_adjusted,
    Sigma_Forecast = sigma_hat_forecast
  ))
}

# Gerar resultados
df_garch_sigma <- generate_sigma_hat_completo(returns, "garch", window_size = 1500)

df_msgarch_sigma <- generate_sigma_hat_completo(returns, "msgarch", window_size = 1500)

df_gas_sigma <- generate_sigma_hat_completo(returns, "gas", window_size = 1500)


# --- GARCH --- #

# Criar dataframe com datas e sigma_garch
df_garch <- data.frame(
  Date = df$Data,
  Sigma_Adjusted = df_garch_sigma$Sigma_Adjusted,
  Sigma_GARCH = df_garch_sigma$Sigma_Forecast
)

resultados_garch <- data.frame(
  Date = df$Data,
  Returns = df$log_return,
  Sigma_Adjusted = df_garch_sigma$Sigma_Adjusted,
  Sigma_GARCH = df_garch_sigma$Sigma_Forecast
)

# Calcular resíduos
resultados_garch <- resultados_garch %>%
  mutate(
    Residuals_garch = if_else(!is.na(Sigma_GARCH) & Sigma_GARCH > 0,
                              Returns / sqrt(Sigma_GARCH),
                              NA_real_)
  )

# Exportar para CSV
#write.csv(resultados_garch, "vol_GARCH_1_1.csv", row.names = FALSE)

# --- --- #

#resultados <- data.frame(
#  Date = df$Date,
#  Returns = df$Returns,
#  Returns_sq = returns^2,
#  GARCH = sigma_garch,
#  GAS = sigma_gas,
#  MSGARCH = sigma_msgarch
#)

# Cálculo dos resíduos
#resultados <- resultados %>%
#  mutate(
#    Residuals_garch = if_else(!is.na(GARCH) & GARCH > 0, Returns / sqrt(GARCH), NA_real_),
#    Residuals_msgarch = if_else(!is.na(MSGARCH) & MSGARCH > 0, Returns / sqrt(MSGARCH), NA_real_),
#    Residuals_gas = if_else(!is.na(GAS) & GAS > 0, Returns / sqrt(GAS), NA_real_)
#  )

# --- Adicionando coluna Parkinson no CSV com as previsões do GARCH --- #

#dat  <- read.csv("vol_GARCH_1_1.csv", stringsAsFactors = FALSE)
park <- read.csv("petr4_parkinson.csv", stringsAsFactors = FALSE)

resultados_garch$Date <- as.Date(resultados_garch$Date, format = "%Y-%m-%d")
park$Date <- as.Date(park$Date, format = "%Y-%m-%d")

park_sub <- park[, c("Date", "Parkinson")]

dat_merged <- merge(
  resultados_garch,
  park_sub,
  by = "Date",
  all.x = TRUE
)

head(dat_merged)

write.csv(dat_merged, "vol_GARCH_1_1_new.csv", row.names = FALSE)

# ----------------------------
# EXPORTAR
# ----------------------------

#write.csv(resultados, "volatilidades_previstas_completo.csv", row.names = FALSE)


library(zoo)

# -- Obtendo as médias dos retornos -- #
df <- read.csv("petr4_returns.csv")
df$Data <- as.Date(df$Data)
returns <- df$log_return
N <- length(returns)
window_size <- 1500

# Salva as médias para calcular o VaR depois
mu_vec <- rollapply(
  returns,
  width = window_size,
  FUN = mean,
  align = "right",
  fill = NA
)

mu_windows <- mu_vec[window_size:N]

out <- data.frame(
  Date = df$Data[window_size:N],
  Mu_Window = mu_windows
)

write.csv(out, "means_1500_garch_1_1.csv", row.names = FALSE)

# --- MSGARCH --- #

# Criar dataframe com datas e sigma_garch
df_msgarch <- data.frame(
  Date = df$Data,
  Sigma_Adjusted = df_msgarch_sigma$Sigma_Adjusted,
  Sigma_MSGARCH = df_msgarch_sigma$Sigma_Forecast
)

resultados_msgarch <- data.frame(
  Date = df$Data,
  Returns = df$log_return,
  Sigma_Adjusted = df_msgarch_sigma$Sigma_Adjusted,
  Sigma_MSGARCH = df_msgarch_sigma$Sigma_Forecast
)

# Calcular resíduos
resultados_msgarch <- resultados_msgarch %>%
  mutate(
    Residuals_msgarch = if_else(!is.na(Sigma_MSGARCH) & Sigma_MSGARCH > 0,
                              Returns / sqrt(Sigma_MSGARCH),
                              NA_real_)
  )

# Exportar para CSV
#write.csv(resultados_msgarch, "vol_MSGARCH_1_1.csv", row.names = FALSE)

# PARKINSON
#dat  <- read.csv("vol_MSGARCH_1_1.csv", stringsAsFactors = FALSE)
park <- read.csv("petr4_parkinson.csv", stringsAsFactors = FALSE)

resultados_msgarch$Date <- as.Date(resultados_msgarch$Date, format = "%Y-%m-%d")
park$Date <- as.Date(park$Date, format = "%Y-%m-%d")

park_sub <- park[, c("Date", "Parkinson")]

dat_merged <- merge(
  resultados_msgarch,
  park_sub,
  by = "Date",
  all.x = TRUE
)

head(dat_merged)

write.csv(dat_merged, "vol_MSGARCH_1_1_new.csv", row.names = FALSE)

plot(dat_merged$Sigma_MSGARCH, type = 'l')

# --- GAS --- #

# Criar dataframe com datas e sigma_gas
df_gas <- data.frame(
  Date = df$Data,
  Sigma_Adjusted = df_gas_sigma$Sigma_Adjusted,
  Sigma_GAS = df_gas_sigma$Sigma_Forecast
)

resultados_gas <- data.frame(
  Date = df$Data,
  Returns = df$log_return,
  Sigma_Adjusted = df_gas_sigma$Sigma_Adjusted,
  Sigma_GAS = df_gas_sigma$Sigma_Forecast
)

# Calcular resíduos
resultados_gas <- resultados_gas %>%
  mutate(
    Residuals_gas = if_else(!is.na(Sigma_GAS) & Sigma_GAS > 0,
                                Returns / sqrt(Sigma_GAS),
                                NA_real_)
  )

# Exportar para CSV
#write.csv(resultados_gas, "vol_GAS_1_1.csv", row.names = FALSE)

# PARKINSON
#dat  <- read.csv("vol_GAS_1_1.csv", stringsAsFactors = FALSE)
park <- read.csv("petr4_parkinson.csv", stringsAsFactors = FALSE)

resultados_gas$Date <- as.Date(resultados_gas$Date, format = "%Y-%m-%d")
park$Date <- as.Date(park$Date, format = "%Y-%m-%d")

park_sub <- park[, c("Date", "Parkinson")]

dat_merged <- merge(
  resultados_gas,
  park_sub,
  by = "Date",
  all.x = TRUE
)

head(dat_merged)

write.csv(dat_merged, "vol_GAS_1_1_new.csv", row.names = FALSE)

plot(df_gas$Sigma_GAS, type = 'l')
