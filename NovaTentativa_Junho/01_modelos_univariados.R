# ----------------------------
# BIBLIOTECAS E DADOS
# ----------------------------

library(rugarch)
library(MSGARCH)
library(GAS)
library(dplyr)
library(tidyr)
library(ggplot2)
library(reshape2)

# Carregar dados
df <- read.csv('retornos_btc.csv')
df$Date <- as.Date(df$Date)
returns <- df$Returns 
N <- length(returns)

# ----------------------------
# PARÂMETROS
# ----------------------------

n_ins <- 1500 # tamanho da janela
n_out <- N - n_ins # qtd de janelas
sigma_garch <- numeric(n_out)
sigma_gas <- numeric(n_out)
sigma_msgarch <- numeric(n_out)
realized_variance <- numeric(n_out)

# ----------------------------
# ESPECIFICAÇÕES DOS MODELOS
# ----------------------------

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

# ----------------------------
# ROLLING WINDOW
# ----------------------------

for (i in 1:n_out) {
  cat("Processando janela", i, "de", n_out, "\n")
  
  if ((i + n_ins) > length(returns)) break
  
  ret_window <- returns[i:(i + n_ins - 1)]
  y_next <- returns[i + n_ins]
  realized_variance[i] <- y_next^2
  
  # GARCH
  garch_fit <- ugarchfit(spec = garch_spec, data = ret_window, solver = "hybrid", silent = TRUE)
  garch_fc <- ugarchforecast(garch_fit, n.ahead = 1)
  sigma_garch[i] <- sigma(garch_fc)^2
  
  # GAS
  gas_fit <- UniGASFit(gas_spec, ret_window)
  gas_fc <- UniGASFor(gas_fit, H = 1)
  nu <- gas_fit@GASDyn$mTheta[3, 1]
  sigma_gas[i] <- gas_fc@Forecast$PointForecast[2] * nu / (nu - 2)
  
  # MSGARCH
  msgarch_fit <- FitML(msgarch_spec, ret_window)
  msgarch_fc <- predict(msgarch_fit, nahead = 1)
  sigma_msgarch[i] <- msgarch_fc$vol^2
}

# ----------------------------
# RESULTADOS
# ----------------------------

resultados <- data.frame(
  Date = df$Date[(n_ins + 1):N],
  Returns_sq = realized_variance,
  GARCH = sigma_garch,
  GAS = sigma_gas,
  MSGARCH = sigma_msgarch
  )

retornos_btc <- read.csv("retornos_btc.csv")  
retornos_btc$Date <- as.Date(retornos_btc$Date)

resultados_completos <- merge(resultados, retornos_btc, by = "Date")

resultados_completos <- resultados_completos %>%
  mutate(
    Residuals_garch = if_else(!is.na(GARCH) & GARCH > 0, Returns / sqrt(GARCH), NA_real_),
    Residuals_msgarch = if_else(!is.na(MSGARCH) & MSGARCH > 0, Returns / sqrt(MSGARCH), NA_real_),
    Residuals_gas = if_else(!is.na(GAS) & GAS > 0, Returns / sqrt(GAS), NA_real_)
  )

# Salvar resultados
write.csv(resultados_completos, "volatilidades_previstas_completo.csv", row.names = FALSE) # não estou usando mais esse CSV no script das LSTM;
# estou usando um CSV chamado volatilidades_previstas_completo_corrigido.csv, que inclui todo o período da análise. Nesse CSV acima são consideradas
# apenas as observações a partir do dia 1500; o CSV corrigido é um "empilhamento" do dataframe de retornos do dia 1 até 1500 com o CSV acima.
# ----------------------------
# PLOT
# ----------------------------

plot_data <- melt(resultados_completos[, !(names(resultados_completos) %in% "Returns")], id.vars = "Date", variable.name = "Modelo", value.name = "Sigma2")

ggplot(plot_data, aes(x = Date, y = Sigma2, color = Modelo)) +
  geom_line() +
  theme_minimal() +
  labs(title = "Previsão de Volatilidade (Sigma^2)", y = "Volatilidade Condicional", x = "")
