################################################################################
#####                     Empirical Application                            #####
################################################################################


library(rugarch)
library(MSGARCH)
library(GAS)
library(dplyr)
library(tidyr)
library(keras)
library(tensorflow)
library(tfruns)


df <- read.csv('./BTC_novos_testes/BTC_new/retornos_btc.csv')
df$Date <- as.Date(df$Date)
returns <- df$Returns
N <- length(returns)


# ----------------------------
# ESPECIFICAÇÕES DOS MODELOS
# ----------------------------

garch_spec <- ugarchspec(variance.model = list(model = "sGARCH"), mean.model = list(armaOrder = c(0, 0), include.mean = FALSE), distribution.model = "std")
msgarch_spec <- CreateSpec(variance.spec = list(model = "sGARCH"), distribution.spec = list(distribution = "std"), switch.spec = list(do.mix = FALSE, K = 2))
gas_spec <- UniGASSpec(Dist = "std", ScalingType = "Identity", GASPar = list(scale = TRUE))





ins <- 1500
oos <- nrow(df) - ins
sigma2_fore <- matrix(0, ncol = 7, nrow = oos)
for (i in 1:oos) {
  r <- scale(returns[i:(i + ins - 1)], scale = FALSE)
  
  GARCH_fit <- ugarchfit(garch_spec, r)
  MSGARCH_fit <- FitML(msgarch_spec, r)
  GAS_fit <- UniGASFit(gas_spec, r, Compute.SE = FALSE)
  
  # Valores ajustados da volatilidade ao quadrado (que entrarao na LSTM)
  sigma2_GARCH <- sigma(GARCH_fit)^2
  sigma2_MSGARCH <- Volatility(MSGARCH_fit)^2
  sigma2_GAS <- GAS_fit@GASDyn$mTheta[2, 1:ins] * GAS_fit@GASDyn$mTheta[3, 1] /(GAS_fit@GASDyn$mTheta[3, 1] - 2)
  
  # Squared Volatility Forecasts
  sigma2_fore[i, 1:3] <- c(ugarchforecast(GARCH_fit, n.ahead = 1)@forecast$sigmaFor[1]^2,
                           as.numeric(predict(MSGARCH_fit, nahead = 1)$vol^2),
                           UniGASFor(GAS_fit, H = 1)@Forecast$PointForecast[, 2] * GAS_fit@GASDyn$mTheta[3, 1] /(GAS_fit@GASDyn$mTheta[3, 1] - 2))
  
  
  # LSTM
  tuning_run("lstm_tuning.R", flags = list(
    start_index = 1,
    window_size = 1500,
    lag = c(14, 30),
    units = c(32, 64),
    dropout = 0.2,
    patience = 10,
    batch_size = c(16, 32, 64),
    epochs = seq(50, 200, 10)
  ))
  
  
  
}



