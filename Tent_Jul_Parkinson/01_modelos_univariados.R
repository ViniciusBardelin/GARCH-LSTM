# ----------------------------
# BIBLIOTECAS E DADOS
# ----------------------------

## Esse script salva todos os valores do dia 1 até 2632, onde [1,1500] são os valores ajustados e [1501,2632] são os valores preditos.
## esse script é uma adaptação de 01_modelos_univariados_1.R
## garch funciona! GAS não!

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

plot(df$log_return, type = 'l')

# ----------------------------
# PARÂMETROS
# ----------------------------

n_ins <- 1500

# ----------------------------
# ESPECIFICAÇÕES DOS MODELOS
# ----------------------------

# trucios
#g_spec <- ugarchspec(
#  variance.model     = list(model = "sGARCH"),
#  mean.model         = list(armaOrder = c(0,0), include.mean = FALSE),
#  distribution.model = "std"
#)

# novo -> mudei pois armaOrder é outra coisa! por padrão ele ja usa (1,1) para modelar a variância!
garch_spec <- ugarchspec(
  variance.model = list(model= "sGARCH", garchOrder = c(1,1)),
  mean.model = list(armaOrder  = c(0,0), include.mean = FALSE),
  distribution.model = "std"
)

# original
#garch_spec <- ugarchspec(
#  variance.model = list(model = "sGARCH"),
#  mean.model = list(armaOrder = c(1,1)),
#  distribution.model = "std"
#)

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

# ----------------------------
# FUNÇÃO PARA GERAR AJUSTADOS + PREVISÕES
# ----------------------------

generate_sigma_hat_completo <- function(returns, model_type, window_size = 1500) {
  n <- length(returns)
  sigma_hat <- rep(NA, n)
  
  # Valores ajustados (1 até window_size)
  fit_tudo <- switch(model_type,
                     "garch" = ugarchfit(garch_spec, returns[1:window_size]),
                     "msgarch" = FitML(msgarch_spec, returns[1:window_size]),
                     "gas" = UniGASFit(gas_spec, returns[1:window_size])
  )
  
  if (model_type == "garch") {
    sigma_hat[1:window_size] <- as.numeric(sigma(fit_tudo))^2
  } else if (model_type == "msgarch") {
    sigma_hat[1:window_size] <- as.numeric(fitted(fit_tudo))^2
  } else if (model_type == "gas") {
    nu <- fit_tudo@GASDyn$mTheta[3, 1]
    sigma_hat[1:window_size] <- as.numeric(fitted(fit_tudo)) * nu / (nu - 2)
  }
  
  # Previsões (window)
  for (i in (window_size + 1):n) {
    window_returns <- returns[(i - window_size):(i - 1)]
    
    fit <- switch(model_type,
                  "garch" = ugarchfit(garch_spec, window_returns),
                  "msgarch" = FitML(msgarch_spec, window_returns),
                  "gas" = UniGASFit(gas_spec, window_returns)
    )
    
    sigma_hat[i] <- switch(model_type,
                           "garch" = as.numeric(sigma(fit)[window_size])^2,
                           "msgarch" = (predict(fit, nahead = 1)$vol)^2,
                           "gas" = {
                             forecast <- UniGASFor(fit, H = 1)
                             nu <- fit@GASDyn$mTheta[3, 1]
                             forecast@Forecast$PointForecast[2] * nu / (nu - 2)
                           }
    )
  }
  
  return(sigma_hat)
}

# ----------------------------
# GERAR RESULTADOS
# ----------------------------

sigma_garch <- generate_sigma_hat_completo(returns, "garch", n_ins)
sigma_gas <- generate_sigma_hat_completo(returns, "gas", n_ins)
sigma_msgarch <- generate_sigma_hat_completo(returns, "msgarch", n_ins)


######## validando GARCH por enquanto
# Criar dataframe com datas e sigma_garch
df_garch <- data.frame(
  Date = df$Data,
  Sigma_GARCH = sigma_garch
)

# Exportar para CSV
#write.csv(df_garch, "sigma_garch_ajustado_e_previsto.csv", row.names = FALSE)

resultados_garch <- data.frame(
  Date = df$Data,
  Returns = df$log_return,
  Sigma_GARCH = sigma_garch
)

# Calcular resíduos
resultados_garch <- resultados_garch %>%
  mutate(
    Residuals_garch = if_else(!is.na(Sigma_GARCH) & Sigma_GARCH > 0,
                              Returns / sqrt(Sigma_GARCH),
                              NA_real_)
  )

# Exportar para CSV
write.csv(resultados_garch, "volatilidades_previstas_completo_corrigido_GARCH_1_1.csv", row.names = FALSE)

# Métricas só para o GARCH
preds_GARCH <- read.csv("volatilidades_previstas_completo_corrigido_GARCH_1_1.csv")
preds_GARCH <- tail(preds_GARCH, 1132) # considerando apenas o período de previsões do modelo híbrido

sigma2_hat_GARCH <- preds_GARCH$Sigma_GARCH # previsões de sigma^2
returns2_GARCH <- preds_GARCH$Returns_sq # retornos^2 como proxy da volatilidade

## QLIKE
valid_GARCH <- !is.na(sigma2_hat_GARCH) & !is.na(returns2_GARCH) & sigma2_hat_GARCH > 1e-8  # evitar divisão por números muito pequenos
qlike_GARCH <- mean(log(sigma2_hat_GARCH[valid_GARCH]) + (returns2_GARCH[valid_GARCH] / sigma2_hat_GARCH[valid_GARCH]))
print(paste("QLIKE_GARCH:", qlike_GARCH))

## MSE
mse_val_GARCH <- mse(returns2_GARCH, sigma2_hat_GARCH)
cat("MSE_GARCH:", mse_val_GARCH, "\n")


##########

resultados <- data.frame(
  Date = df$Date,
  Returns = df$Returns,
  Returns_sq = returns^2,
  GARCH = sigma_garch,
  GAS = sigma_gas,
  MSGARCH = sigma_msgarch
)

# Cálculo dos resíduos
resultados <- resultados %>%
  mutate(
    Residuals_garch = if_else(!is.na(GARCH) & GARCH > 0, Returns / sqrt(GARCH), NA_real_),
    Residuals_msgarch = if_else(!is.na(MSGARCH) & MSGARCH > 0, Returns / sqrt(MSGARCH), NA_real_),
    Residuals_gas = if_else(!is.na(GAS) & GAS > 0, Returns / sqrt(GAS), NA_real_)
  )

# ----------------------------
# EXPORTAR
# ----------------------------

write.csv(resultados, "volatilidades_previstas_completo_corrigido.csv", row.names = FALSE)
