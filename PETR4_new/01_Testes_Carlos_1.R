library(rugarch)
library(MSGARCH)
library(GAS)
library(dplyr)
library(tidyr)
library(arfima)

df <- read.csv("petr4_returns.csv")
df$Date <- as.Date(df$Date)
returns <- df$log_return

n_ins <- 1500
n_tot <- length(returns)
n_oos <- n_tot - n_ins

garch_spec<- ugarchspec(variance.model = list(model= "sGARCH", garchOrder = c(1,1)), mean.model = list(armaOrder  = c(0,0), include.mean = FALSE), distribution.model = "std")
msgarch_spec <- CreateSpec(variance.spec = list(model = "sGARCH"), distribution.spec = list(distribution = "std"), switch.spec = list(do.mix = FALSE, K = 2))
gas_spec <- UniGASSpec(Dist = "std", ScalingType = "Identity", GASPar = list(scale = TRUE))

#InS

# Matriz com valores ajustados + previsões (variância sigma²)
sigma2_completo <- matrix(NA_real_, nrow = n_tot, ncol = 3,
                          dimnames = list(NULL, c("GARCH", "MSGARCH", "GAS")))

# Ajuste único nos n_ins primeiros dados centrados
returns_c <- scale(returns[1:n_ins], scale = FALSE)

fit_GARCH <- ugarchfit(garch_spec, returns_c, solver = "hybrid")
fit_GAS <- UniGASFit(gas_spec, returns_c, Compute.SE = FALSE)
fit_MSGARCH <- FitML(msgarch_spec, returns_c, ctr = list(do.se = FALSE))

# Valores ajustados (sigma²) nas primeiras n_ins posições
sigma2_completo[1:n_ins, "GARCH"] <- sigma(fit_GARCH)^2
sigma2_completo[1:n_ins, "GAS"] <- fit_GAS@GASDyn$mTheta[2, 1:n_ins] * fit_GAS@GASDyn$mTheta[3, 1] / (fit_GAS@GASDyn$mTheta[3, 1] - 2)
sigma2_completo[1:n_ins, "MSGARCH"] <- Volatility(fit_MSGARCH)^2

# OoS
ES_1 <- ES_2 <- ES_5 <- VaR_1 <- VaR_2 <- VaR_5 <- sigma2 <- matrix(0, ncol = 3, nrow = n_oos, dimnames = list(NULL, c("GARCH", "MSGARCH", "GAS")))
r_oos <- c()
for (i in 1:n_oos) {
  print(i)
  returns_window <- returns[i:(i + n_ins - 1)]
  mu <- mean(returns_window)
  returns_c <- scale(returns_window, scale = FALSE)
  
  fit_GARCH <- ugarchfit(garch_spec, returns_c, solver = "hybrid")
  fit_GAS <- UniGASFit(gas_spec, returns_c, Compute.SE = FALSE)
  fit_MSGARCH <- FitML(msgarch_spec , returns_c, ctr = list(do.se = FALSE))
  
  # One-step-ahead Volatility**2 (variancia)
  sigma2[i, "GARCH"] <- ugarchforecast(fit_GARCH, n.ahead = 1)@forecast$sigmaFor[1]^2
  sigma2[i, "MSGARCH"] <- predict(fit_MSGARCH , nahead = 1)$vol^2
  sigma2[i, "GAS"] <- UniGASFor(fit_GAS, H = 1)@Forecast$PointForecast[, 2] * fit_GAS@GASDyn$mTheta[3, 1] /(fit_GAS@GASDyn$mTheta[3, 1] - 2)
    
  # Guardar valores ajustados na matriz completa
  sigma2_completo[i + n_ins, "GARCH"] <- sigma(fit_GARCH)[n_ins]^2
  sigma2_completo[i + n_ins, "GAS"] <- fit_GAS@GASDyn$mTheta[2, n_ins] * fit_GAS@GASDyn$mTheta[3, 1] / (fit_GAS@GASDyn$mTheta[3, 1] - 2)
  sigma2_completo[i + n_ins, "MSGARCH"] <- Volatility(fit_MSGARCH)[n_ins]^2
  
  # Residuals
  res_GARCH <- as.numeric(returns_c/sigma(fit_GARCH))
  res_GAS <-   as.numeric(returns_c/sqrt(fit_GAS@GASDyn$mTheta[2, 1:n_ins] * fit_GAS@GASDyn$mTheta[3, 1] /(fit_GAS@GASDyn$mTheta[3, 1] - 2)))
  res_MSGARCH <- as.numeric(returns_c/ Volatility(fit_MSGARCH))
  
  # VaR and ES 
  # 1%
  VaR_1[i, "GARCH"] = mu + sqrt(sigma2[i, "GARCH"]) * quantile(res_GARCH, 0.01)
  VaR_1[i, "GAS"] = mu + sqrt(sigma2[i, "GAS"] )* quantile(res_GAS, 0.01)
  VaR_1[i, "MSGARCH"] = mu + sqrt(sigma2[i, "MSGARCH"]) * quantile(res_MSGARCH, 0.01)
  
  ES_1[i, "GARCH"] <- mean(returns_window[returns_window < VaR_1[i, "GARCH"]])
  ES_1[i, "GAS"] <- mean(returns_window[returns_window < VaR_1[i, "GAS"]])
  ES_1[i, "MSGARCH"] <- mean(returns_window[returns_window < VaR_1[i, "MSGARCH"]])
  
  # 2.5%
  VaR_2[i, "GARCH"] = mu + sqrt(sigma2[i, "GARCH"]) * quantile(res_GARCH, 0.025)
  VaR_2[i, "GAS"] = mu + sqrt(sigma2[i, "GAS"] )* quantile(res_GAS, 0.025)
  VaR_2[i, "MSGARCH"] = mu + sqrt(sigma2[i, "MSGARCH"]) * quantile(res_MSGARCH, 0.025)
  
  ES_2[i, "GARCH"] <- mean(returns_window[returns_window < VaR_2[i, "GARCH"]])
  ES_2[i, "GAS"] <- mean(returns_window[returns_window < VaR_2[i, "GAS"]])
  ES_2[i, "MSGARCH"] <- mean(returns_window[returns_window < VaR_2[i, "MSGARCH"]])
  
  # 5%
  VaR_5[i, "GARCH"] = mu + sqrt(sigma2[i, "GARCH"]) * quantile(res_GARCH, 0.05)
  VaR_5[i, "GAS"] = mu + sqrt(sigma2[i, "GAS"] )* quantile(res_GAS, 0.05)
  VaR_5[i, "MSGARCH"] = mu + sqrt(sigma2[i, "MSGARCH"]) * quantile(res_MSGARCH, 0.05)
  
  ES_5[i, "GARCH"] <- mean(returns_window[returns_window < VaR_5[i, "GARCH"]])
  ES_5[i, "GAS"] <- mean(returns_window[returns_window < VaR_5[i, "GAS"]])
  ES_5[i, "MSGARCH"] <- mean(returns_window[returns_window < VaR_5[i, "MSGARCH"]])
  
  r_oos[i] <- returns[i + n_ins]
  
}

# InS
df_sigma2_completo <- data.frame(
  Date = df$Date,
  Returns = df$log_return,
  Sigma2_GARCH = sigma2_completo[, "GARCH"],
  Sigma2_GAS = sigma2_completo[, "GAS"],
  Sigma2_MSGARCH = sigma2_completo[, "MSGARCH"],
  Parkinson = df$Parkinson
)

write.csv(df_sigma2_completo, "sigma2_ajustado_e_previsto_completo.csv", row.names = FALSE)

# OoS
df_oos <- data.frame(
  Date = df$Date[(n_ins + 1):n_tot],
  Return = r_oos,
  Vol_GARCH = sqrt(sigma2[, "GARCH"]),
  Vol_MSGARCH = sqrt(sigma2[, "MSGARCH"]),
  Vol_GAS = sqrt(sigma2[, "GAS"]),
  
  VaR_GARCH_1 = VaR_1[, "GARCH"],
  VaR_MSGARCH_1 = VaR_1[, "MSGARCH"],
  VaR_GAS_1 = VaR_1[, "GAS"],
  ES_GARCH_1 = ES_1[, "GARCH"],
  ES_MSGARCH_1 = ES_1[, "MSGARCH"],
  ES_GAS_1 = ES_1[, "GAS"],
  
  VaR_GARCH_2_5 = VaR_2[, "GARCH"],
  VaR_MSGARCH_2_5 = VaR_2[, "MSGARCH"],
  VaR_GAS_2_5 = VaR_2[, "GAS"],
  ES_GARCH_2_5 = ES_2[, "GARCH"],
  ES_MSGARCH_2_5 = ES_2[, "MSGARCH"],
  ES_GAS_2_5 = ES_2[, "GAS"],
  
  VaR_GARCH_5 = VaR_5[, "GARCH"],
  VaR_MSGARCH_5 = VaR_5[, "MSGARCH"],
  VaR_GAS_5 = VaR_5[, "GAS"],
  ES_GARCH_5 = ES_5[, "GARCH"],
  ES_MSGARCH_5 = ES_5[, "MSGARCH"],
  ES_GAS_5 = ES_5[, "GAS"],
  
  Parkinson = df$Parkinson[(n_ins + 1):n_tot]
)

write.csv(df_oos, "previsoes_oos_vol_var_es.csv", row.names = FALSE)

# ARFIMA
df <- df %>%
  mutate(Log_Parkinson = log(Parkinson))
log_pks <- df$Log_Parkinson

sigma2 <- matrix(NA, nrow = n_oos, ncol = 1)
colnames(sigma2) <- "ARFIMA1"

sigma2_completo <- matrix(NA, nrow = n_total, ncol = 1)
colnames(sigma2_completo) <- "ARFIMA1"

# Ins
log_pks_ins <- df$Log_Parkinson[1:1500]
log_pks_ins_c <- scale(log_pks_ins, scale = FALSE)
mu <- attr(log_pks_c, "scaled:center") 

fit_0d1 <- try(arfima(log_pks_ins_c, order = c(0, 0, 1), back = TRUE), silent = TRUE)

resid_arfima <- resid(fit_0d1)
resid_arfima <- resid_arfima$Mode1
fitted_arfima <- fitted(fit_0d1)
fitted_arfima <- fitted_arfima$Mode1

data_arfima <- df$Date[1:n_ins]
df_arfima_insample <- data.frame(
  Date = data_arfima,
  Sigma2_ARFIMA_ajustado = fitted_arfima,
  Residuals = resid_arfima
)

write.csv(df_arfima_insample, "df_arfima_insample.csv", row.names = FALSE)

# OoS
for (i in 1:n_oos) {
  cat("Iteração:", i, "\n")
  
  pks_window <- log_pks[i:(i + n_ins - 1)]
  pks_window_c <- scale(pks_window, scale = FALSE)  
  mu <- attr(pks_window_c, "scaled:center")   
  
  fit_1d0 <- try(arfima(pks_window_c, order = c(1, 0, 0), back = TRUE), silent = TRUE)
  fit_0d1 <- try(arfima(pks_window_c, order = c(0, 0, 1), back = TRUE), silent = TRUE)
  fit_1d1 <- try(arfima(pks_window_c, order = c(1, 0, 1), back = TRUE), silent = TRUE)
  
  fits <- list(fit_1d0, fit_0d1, fit_1d1)
  bics <- sapply(fits, function(f) {
    if (inherits(f, "try-error") || is.null(f$bic) || !is.numeric(f$bic)) Inf else f$bic
  })
  best_idx <- which.min(bics)
  best_fit <- fits[[best_idx]]
  
  pred <- predict(best_fit, n.ahead = 1)
  log_sigma2_hat <- pred[[1]]$Forecast[1]
  sigma2[i, "ARFIMA1"] <- exp(log_sigma2_hat)  # previsão de variância OoS
}

df_arfima <- data.frame(
  Date = df$Date[(n_ins + 1):n_total],
  Sigma2_ARFIMA = sigma2[, "ARFIMA1"]
)

write.csv(df_arfima, "df_arfima.csv", row.names = FALSE)
