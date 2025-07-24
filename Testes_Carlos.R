library(rugarch)
library(MSGARCH)
library(GAS)
library(dplyr)
library(tidyr)


df <- read.csv("/Users/ctrucios/Github/GARCH-LSTM/Tent_Jul_Parkinson/petr4_returns.csv")
df$Data <- as.Date(df$Data)
returns <- df$log_return


n_ins <- 1500
n_tot <- length(returns)
n_oos <- n_tot - n_ins
  
  
garch_spec<- ugarchspec(variance.model = list(model= "sGARCH", garchOrder = c(1,1)), mean.model = list(armaOrder  = c(0,0), include.mean = FALSE), distribution.model = "std")
msgarch_spec <- CreateSpec(variance.spec = list(model = "sGARCH"), distribution.spec = list(distribution = "std"), switch.spec = list(do.mix = FALSE, K = 2))
gas_spec <- UniGASSpec(Dist = "std", ScalingType = "Identity", GASPar = list(scale = TRUE))

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
    
  # One-step-ahead Volatility**2
  sigma2[i, "GARCH"] <- ugarchforecast(fit_GARCH, n.ahead = 1)@forecast$sigmaFor[1]^2
  sigma2[i, "MSGARCH"] <- predict(fit_MSGARCH , nahead = 1)$vol^2
  sigma2[i, "GAS"] <- UniGASFor(fit_GAS, H = 1)@Forecast$PointForecast[, 2] * fit_GAS@GASDyn$mTheta[3, 1] /(fit_GAS@GASDyn$mTheta[3, 1] - 2)
  
  # Residuals
  res_GARCH <- as.numeric(returns_c/sigma(fit_GARCH))
  res_GAS <-   as.numeric(returns_c/sqrt(fit_GAS@GASDyn$mTheta[2, 1:n_ins] * fit_GAS@GASDyn$mTheta[3, 1] /(fit_GAS@GASDyn$mTheta[3, 1] - 2)))
  res_MSGARCH <- as.numeric(returns_c/ Volatility(fit_MSGARCH))
    
  
  # VaR and ES FHS
  VaR_1[i, "GARCH"] = mu + sqrt(sigma2[i, "GARCH"]) * quantile(res_GARCH, 0.01)
  VaR_1[i, "GAS"] = mu + sqrt(sigma2[i, "GAS"] )* quantile(res_GAS, 0.01)
  VaR_1[i, "MSGARCH"] = mu + sqrt(sigma2[i, "MSGARCH"]) * quantile(res_MSGARCH, 0.01)
  
  ES_1[i, "GARCH"] <- mean(returns_window[returns_window < VaR_1[i, "GARCH"]])
  ES_1[i, "GAS"] <- mean(returns_window[returns_window < VaR_1[i, "GAS"]])
  ES_1[i, "MSGARCH"] <- mean(returns_window[returns_window < VaR_1[i, "MSGARCH"]])
  
  r_oos[i] <- returns[i + n_ins]
  
}

mean(r_oos < VaR_1[, "GARCH"])
mean(r_oos < VaR_1[, "GAS"])
mean(r_oos < VaR_1[, "MSGARCH"])