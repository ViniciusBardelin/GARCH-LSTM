library(dplyr)
library(readr)
library(tidyr)
library(ggplot2)
library(plotly)
library(lubridate)
library(patchwork)
library(PerformanceAnalytics)
library(esback)
library(GAS)
library(knitr)
library(kableExtra)
library(Rcpp)
source("Function_VaR_VQR.R")
source("Optimizations.R")

#sourceCpp("scoring_functions.cpp")

# Parâmetros gerais
initial_train <- 1500
alphas <- c(0.01, 0.025, 0.05)
n_oos <- 2000
df <- read.csv("sigma2_ajustado_e_previsto_completo.csv")

# GARCH ----------------------------------------------------------------------------
df_garch <- read.csv("previsoes_oos_vol_var_es.csv") %>%
  mutate(
    Date = ymd(Date),
    Vol2_GARCH = Vol_GARCH^2,
    VaR_GARCH_2 = VaR_GARCH_2_5,
    ES_GARCH_2 = ES_GARCH_2_5,
    Exceed_VaR_GARCH_1 = Return < VaR_GARCH_1,
    Exceed_VaR_GARCH_2 = Return < VaR_GARCH_2,
    Exceed_VaR_GARCH_5 = Return < VaR_GARCH_5
  ) %>%
  select(
    Date, Return, Vol2_GARCH,
    VaR_GARCH_1, ES_GARCH_1,
    VaR_GARCH_2, ES_GARCH_2,
    VaR_GARCH_5, ES_GARCH_5,
    Exceed_VaR_GARCH_1, Exceed_VaR_GARCH_2, Exceed_VaR_GARCH_5
  )

df_garch <- df_garch %>%
  filter(!is.na(Return), !is.na(VaR_GARCH_1), !is.na(ES_GARCH_1), !is.na(Vol2_GARCH)) %>%
  filter(Vol2_GARCH > 0, ES_GARCH_1 < 0)

table(df_garch$Exceed_VaR_GARCH_1) # 25     espera: 20
table(df_garch$Exceed_VaR_GARCH_2) # 49     espera: 50
table(df_garch$Exceed_VaR_GARCH_5) # 82     espera: 100

# VaR Backtest: UC, CC, DQ, QL, 
Back_VaR_GARCH_1 <- BacktestVaR(df_garch$Return, df_garch$VaR_GARCH_1, 0.01)
Back_VaR_GARCH_2 <- BacktestVaR(df_garch$Return, df_garch$VaR_GARCH_2, 0.025)
Back_VaR_GARCH_5 <- BacktestVaR(df_garch$Return, df_garch$VaR_GARCH_5, 0.05)

VaR_VQR(df_garch$Return, df_garch$VaR_GARCH_1, 0.01)
VaR_VQR(df_garch$Return, df_garch$VaR_GARCH_2, 0.025)
VaR_VQR(df_garch$Return, df_garch$VaR_GARCH_5, 0.05)

Back_VaR_QL_GARCH_1 <- Back_VaR_GARCH_1$Loss$Loss
Back_VaR_QL_GARCH_2 <- Back_VaR_GARCH_2$Loss$Loss
Back_VaR_QL_GARCH_5 <- Back_VaR_GARCH_5$Loss$Loss

Back_VaR_FZ_GARCH_1 <- FZLoss(df_garch$Return, df_garch$VaR_GARCH_1, df_garch$ES_GARCH_1, 0.01)
Back_VaR_FZ_GARCH_1 <- mean(Back_VaR_FZ_GARCH_1)
Back_VaR_FZ_GARCH_2 <- FZLoss(df_garch$Return, df_garch$VaR_GARCH_2, df_garch$ES_GARCH_2, 0.025)
Back_VaR_FZ_GARCH_2 <- mean(Back_VaR_FZ_GARCH_2)
Back_VaR_FZ_GARCH_5 <- FZLoss(df_garch$Return, df_garch$VaR_GARCH_5, df_garch$ES_GARCH_5, 0.05)
Back_VaR_FZ_GARCH_5 <- mean(Back_VaR_FZ_GARCH_5)

Back_VaR_NZ_GARCH_1 <- NZ_deprecated(df_garch$VaR_GARCH_1, df_garch$ES_GARCH_1, df_garch$Return, 0.01)
Back_VaR_NZ_GARCH_1 <- mean(Back_VaR_NZ_GARCH_1)
Back_VaR_NZ_GARCH_2 <- NZ_deprecated(df_garch$VaR_GARCH_2, df_garch$ES_GARCH_2, df_garch$Return, 0.025)
Back_VaR_NZ_GARCH_2 <- mean(Back_VaR_NZ_GARCH_2)
Back_VaR_NZ_GARCH_5 <- NZ_deprecated(df_garch$VaR_GARCH_5, df_garch$ES_GARCH_5, df_garch$Return, 0.05)
Back_VaR_NZ_GARCH_5 <- mean(Back_VaR_NZ_GARCH_5)

Back_VaR_AL_GARCH_1 <- AL_deprecated(df_garch$VaR_GARCH_1, df_garch$ES_GARCH_1, df_garch$Return, 0.01)
Back_VaR_AL_GARCH_1 <- mean(Back_VaR_AL_GARCH_1)
Back_VaR_AL_GARCH_2 <- AL_deprecated(df_garch$VaR_GARCH_2, df_garch$ES_GARCH_2, df_garch$Return, 0.025)
Back_VaR_AL_GARCH_2 <- mean(Back_VaR_AL_GARCH_2)
Back_VaR_AL_GARCH_5 <- AL_deprecated(df_garch$VaR_GARCH_5, df_garch$ES_GARCH_5, df_garch$Return, 0.05)
Back_VaR_AL_GARCH_5 <- mean(Back_VaR_AL_GARCH_5)

df_scores_garch <- data.frame(
  Nível = c("1%", "2.5%", "5%"),
  QL = c(Back_VaR_QL_GARCH_1, Back_VaR_QL_GARCH_2, Back_VaR_QL_GARCH_5),
  FZ = c(Back_VaR_FZ_GARCH_1, Back_VaR_FZ_GARCH_2, Back_VaR_FZ_GARCH_5),
  NZ = c(Back_VaR_NZ_GARCH_1, Back_VaR_NZ_GARCH_2, Back_VaR_NZ_GARCH_5)
)


# ES Backtest: CoC, ER, ESR
vol_garch <- sqrt(df_garch$Vol2_GARCH[1:nrow(df_garch)])
Back_ES_CoC_GARCH_1 <- cc_backtest(df_garch$Return, df_garch$VaR_GARCH_1, df_garch$ES_GARCH_1, vol_garch, 0.01)
Back_ES_CoC_GARCH_2 <- cc_backtest(df_garch$Return, df_garch$VaR_GARCH_2, df_garch$ES_GARCH_2, vol_garch, 0.025)
Back_ES_CoC_GARCH_5 <- cc_backtest(df_garch$Return, df_garch$VaR_GARCH_5, df_garch$ES_GARCH_5, vol_garch, 0.05)

Back_ES_ER_GARCH_1 <- er_backtest(df_garch$Return, df_garch$VaR_GARCH_1, df_garch$ES_GARCH_1, vol_garch)
Back_ES_ER_GARCH_2 <- er_backtest(df_garch$Return, df_garch$VaR_GARCH_2, df_garch$ES_GARCH_2, vol_garch)
Back_ES_ER_GARCH_5 <- er_backtest(df_garch$Return, df_garch$VaR_GARCH_5, df_garch$ES_GARCH_5, vol_garch)

Back_ES_ESR_GARCH_1_V1 <- esr_backtest(df_garch$Return, df_garch$VaR_GARCH_1, df_garch$ES_GARCH_1, alpha=0.01, version=1, B=0)
Back_ES_ESR_GARCH_2_V1 <- esr_backtest(df_garch$Return, df_garch$VaR_GARCH_2, df_garch$ES_GARCH_2, alpha=0.025, version=1, B=0)
Back_ES_ESR_GARCH_5_V1 <- esr_backtest(df_garch$Return, df_garch$VaR_GARCH_5, df_garch$ES_GARCH_5, alpha=0.05, version=1, B=0)
# MSGARCH ----------------------------------------------------------------------------
df_msgarch <- read.csv("previsoes_oos_vol_var_es.csv") %>%
  mutate(
    Date = ymd(Date),
    Vol2_MSGARCH = Vol_MSGARCH^2,
    VaR_MSGARCH_2 = VaR_MSGARCH_2_5,
    ES_MSGARCH_2 = ES_MSGARCH_2_5,
    Exceed_VaR_MSGARCH_1 = Return < VaR_MSGARCH_1,
    Exceed_VaR_MSGARCH_2 = Return < VaR_MSGARCH_2,
    Exceed_VaR_MSGARCH_5 = Return < VaR_GARCH_5
  ) %>%
  select(
    Date, Return, Vol2_MSGARCH,
    VaR_MSGARCH_1, ES_MSGARCH_1,
    VaR_MSGARCH_2, ES_MSGARCH_2,
    VaR_MSGARCH_5, ES_MSGARCH_5,
    Exceed_VaR_MSGARCH_1, Exceed_VaR_MSGARCH_2, Exceed_VaR_MSGARCH_5
  )

df_msgarch <- df_msgarch %>%
  filter(!is.na(Return), !is.na(VaR_MSGARCH_1), !is.na(ES_MSGARCH_1), !is.na(Vol2_MSGARCH)) %>%
  filter(Vol2_MSGARCH > 0, ES_MSGARCH_1 < 0)

table(df_msgarch$Exceed_VaR_MSGARCH_1) # hits: 29      espera: 20
table(df_msgarch$Exceed_VaR_MSGARCH_2) # hits: 58     espera: 50
table(df_msgarch$Exceed_VaR_MSGARCH_5) # hits: 83     espera: 100

# VaR Backtest: UC, CC, DQ
Back_VaR_MSGARCH_1 <- BacktestVaR(df_msgarch$Return, df_msgarch$VaR_MSGARCH_1, 0.01)
Back_VaR_MSGARCH_2 <- BacktestVaR(df_msgarch$Return, df_msgarch$VaR_MSGARCH_2, 0.025)
Back_VaR_MSGARCH_5 <- BacktestVaR(df_msgarch$Return, df_msgarch$VaR_MSGARCH_5, 0.05)

VaR_VQR(df_msgarch$Return, df_msgarch$VaR_MSGARCH_1, 0.01)
VaR_VQR(df_msgarch$Return, df_msgarch$VaR_MSGARCH_2, 0.025)
VaR_VQR(df_msgarch$Return, df_msgarch$VaR_MSGARCH_5, 0.05)

Back_VaR_QL_MSGARCH_1 <- Back_VaR_MSGARCH_1$Loss$Loss
Back_VaR_QL_MSGARCH_2 <- Back_VaR_MSGARCH_2$Loss$Loss
Back_VaR_QL_MSGARCH_5 <- Back_VaR_MSGARCH_5$Loss$Loss

Back_VaR_FZ_MSGARCH_1 <- mean(FZLoss(df_msgarch$Return, df_msgarch$VaR_MSGARCH_1, df_msgarch$ES_MSGARCH_1, 0.01))
Back_VaR_FZ_MSGARCH_2 <- mean(FZLoss(df_msgarch$Return, df_msgarch$VaR_MSGARCH_2, df_msgarch$ES_MSGARCH_2, 0.025))
Back_VaR_FZ_MSGARCH_5 <- mean(FZLoss(df_msgarch$Return, df_msgarch$VaR_MSGARCH_5, df_msgarch$ES_MSGARCH_5, 0.05))

Back_VaR_NZ_MSGARCH_1 <- mean(NZ_deprecated(df_msgarch$VaR_MSGARCH_1, df_msgarch$ES_MSGARCH_1, df_msgarch$Return, 0.01))
Back_VaR_NZ_MSGARCH_2 <- mean(NZ_deprecated(df_msgarch$VaR_MSGARCH_2, df_msgarch$ES_MSGARCH_2, df_msgarch$Return, 0.025))
Back_VaR_NZ_MSGARCH_5 <- mean(NZ_deprecated(df_msgarch$VaR_MSGARCH_5, df_msgarch$ES_MSGARCH_5, df_msgarch$Return, 0.05))

Back_VaR_AL_MSGARCH_1 <- AL_deprecated(df_msgarch$VaR_MSGARCH_1, df_msgarch$ES_MSGARCH_1, df_msgarch$Return, 0.01)
Back_VaR_AL_MSGARCH_1 <- mean(Back_VaR_AL_MSGARCH_1)
Back_VaR_AL_MSGARCH_2 <- AL_deprecated(df_msgarch$VaR_MSGARCH_2, df_msgarch$ES_MSGARCH_2, df_msgarch$Return, 0.025)
Back_VaR_AL_MSGARCH_2 <- mean(Back_VaR_AL_MSGARCH_2)
Back_VaR_AL_MSGARCH_5 <- AL_deprecated(df_msgarch$VaR_MSGARCH_5, df_msgarch$ES_MSGARCH_5, df_msgarch$Return, 0.05)
Back_VaR_AL_MSGARCH_5 <- mean(Back_VaR_AL_MSGARCH_5)

df_scores_msgarch <- data.frame(
  Nível = c("1%", "2.5%", "5%"),
  QL = c(Back_VaR_QL_MSGARCH_1, Back_VaR_QL_MSGARCH_2, Back_VaR_QL_MSGARCH_5),
  FZ = c(Back_VaR_FZ_MSGARCH_1, Back_VaR_FZ_MSGARCH_2, Back_VaR_FZ_MSGARCH_5),
  NZ = c(Back_VaR_NZ_MSGARCH_1, Back_VaR_NZ_MSGARCH_2, Back_VaR_NZ_MSGARCH_5)
)

# ES Backtest: CoC, ER, ESR
vol_msgarch <- sqrt(df_msgarch$Vol2_MSGARCH[1:nrow(df_msgarch)])

Back_ES_CoC_MSGARCH_1 <- cc_backtest(df_msgarch$Return, df_msgarch$VaR_MSGARCH_1, df_msgarch$ES_MSGARCH_1, vol_msgarch, 0.01)
Back_ES_CoC_MSGARCH_2 <- cc_backtest(df_msgarch$Return, df_msgarch$VaR_MSGARCH_2, df_msgarch$ES_MSGARCH_2, vol_msgarch, 0.025)
Back_ES_CoC_MSGARCH_5 <- cc_backtest(df_msgarch$Return, df_msgarch$VaR_MSGARCH_5, df_msgarch$ES_MSGARCH_5, vol_msgarch, 0.05)

Back_ES_ER_MSGARCH_1 <- er_backtest(df_msgarch$Return, df_msgarch$VaR_MSGARCH_1, df_msgarch$ES_MSGARCH_1, vol_msgarch)
Back_ES_ER_MSGARCH_2 <- er_backtest(df_msgarch$Return, df_msgarch$VaR_MSGARCH_2, df_msgarch$ES_MSGARCH_2, vol_msgarch)
Back_ES_ER_MSGARCH_5 <- er_backtest(df_msgarch$Return, df_msgarch$VaR_MSGARCH_5, df_msgarch$ES_MSGARCH_5, vol_msgarch)

Back_ES_ESR_MSGARCH_1_V1 <- esr_backtest(df_msgarch$Return, df_msgarch$VaR_MSGARCH_1, df_msgarch$ES_MSGARCH_1, alpha=0.01, version=1, B=0)
Back_ES_ESR_MSGARCH_2_V1 <- esr_backtest(df_msgarch$Return, df_msgarch$VaR_MSGARCH_2, df_msgarch$ES_MSGARCH_2, alpha=0.025, version=1, B=0)
Back_ES_ESR_MSGARCH_5_V1 <- esr_backtest(df_msgarch$Return, df_msgarch$VaR_MSGARCH_5, df_msgarch$ES_MSGARCH_5, alpha=0.05, version=1, B=0)
# GAS ----------------------------------------------------------------------------
df_gas <- read.csv("previsoes_oos_vol_var_es.csv") %>%
  mutate(
    Date = ymd(Date),
    Vol2_GAS = Vol_GAS^2,
    VaR_GAS_2 = VaR_GAS_2_5,
    ES_GAS_2 = ES_GAS_2_5,
    Exceed_VaR_GAS_1 = Return < VaR_GAS_1,
    Exceed_VaR_GAS_2 = Return < VaR_GAS_2,
    Exceed_VaR_GAS_5 = Return < VaR_GARCH_5
  ) %>%
  select(
    Date, Return, Vol2_GAS,
    VaR_GAS_1, ES_GAS_1,
    VaR_GAS_2, ES_GAS_2,
    VaR_GAS_5, ES_GAS_5,
    Exceed_VaR_GAS_1, Exceed_VaR_GAS_2, Exceed_VaR_GAS_5
  )

df_gas <- df_gas %>%
  filter(!is.na(Return), !is.na(VaR_GAS_1), !is.na(ES_GAS_1), !is.na(Vol2_GAS)) %>%
  filter(Vol2_GAS > 0, ES_GAS_1 < 0)

table(df_gas$Exceed_VaR_GAS_1) # hits: 24      espera: 20
table(df_gas$Exceed_VaR_GAS_2) # hits: 51     espera: 50
table(df_gas$Exceed_VaR_GAS_5) # hits: 80     espera: 100

# VaR Backtest: UC, CC, DQ
Back_VaR_GAS_1 <- BacktestVaR(df_gas$Return, df_gas$VaR_GAS_1, 0.01)
Back_VaR_GAS_2 <- BacktestVaR(df_gas$Return, df_gas$VaR_GAS_2, 0.025)
Back_VaR_GAS_5 <- BacktestVaR(df_gas$Return, df_gas$VaR_GAS_5, 0.05)

VaR_VQR(df_gas$Return, df_gas$VaR_GAS_1, 0.01)
VaR_VQR(df_gas$Return, df_gas$VaR_GAS_2, 0.025)
VaR_VQR(df_gas$Return, df_gas$VaR_GAS_5, 0.05)

Back_VaR_QL_GAS_1 <- Back_VaR_GAS_1$Loss$Loss
Back_VaR_QL_GAS_2 <- Back_VaR_GAS_2$Loss$Loss
Back_VaR_QL_GAS_5 <- Back_VaR_GAS_5$Loss$Loss

Back_VaR_FZ_GAS_1 <- mean(FZLoss(df_gas$Return, df_gas$VaR_GAS_1, df_gas$ES_GAS_1, 0.01))
Back_VaR_FZ_GAS_2 <- mean(FZLoss(df_gas$Return, df_gas$VaR_GAS_2, df_gas$ES_GAS_2, 0.025))
Back_VaR_FZ_GAS_5 <- mean(FZLoss(df_gas$Return, df_gas$VaR_GAS_5, df_gas$ES_GAS_5, 0.05))

Back_VaR_NZ_GAS_1 <- mean(NZ_deprecated(df_gas$VaR_GAS_1, df_gas$ES_GAS_1, df_gas$Return, 0.01))
Back_VaR_NZ_GAS_2 <- mean(NZ_deprecated(df_gas$VaR_GAS_2, df_gas$ES_GAS_2, df_gas$Return, 0.025))
Back_VaR_NZ_GAS_5 <- mean(NZ_deprecated(df_gas$VaR_GAS_5, df_gas$ES_GAS_5, df_gas$Return, 0.05))

Back_VaR_AL_GAS_1 <- AL_deprecated(df_gas$VaR_GAS_1, df_gas$ES_GAS_1, df_gas$Return, 0.01)
Back_VaR_AL_GAS_1 <- mean(Back_VaR_AL_GAS_1)
Back_VaR_AL_GAS_2 <- AL_deprecated(df_gas$VaR_GAS_2, df_gas$ES_GAS_2, df_gas$Return, 0.025)
Back_VaR_AL_GAS_2 <- mean(Back_VaR_AL_GAS_2)
Back_VaR_AL_GAS_5 <- AL_deprecated(df_gas$VaR_GAS_5, df_gas$ES_GAS_5, df_gas$Return, 0.05)
Back_VaR_AL_GAS_5 <- mean(Back_VaR_AL_GAS_5)

df_scores_gas <- data.frame(
  Nível = c("1%", "2.5%", "5%"),
  QL = c(Back_VaR_QL_GAS_1, Back_VaR_QL_GAS_2, Back_VaR_QL_GAS_5),
  FZ = c(Back_VaR_FZ_GAS_1, Back_VaR_FZ_GAS_2, Back_VaR_FZ_GAS_5),
  NZ = c(Back_VaR_NZ_GAS_1, Back_VaR_NZ_GAS_2, Back_VaR_NZ_GAS_5)
)

# ES Backtest: CoC, ER, ESR
vol_gas <- sqrt(df_gas$Vol2_GAS[1:nrow(df_gas)])

Back_ES_CoC_GAS_1 <- cc_backtest(df_gas$Return, df_gas$VaR_GAS_1, df_gas$ES_GAS_1, vol_gas, 0.01)
Back_ES_CoC_GAS_2 <- cc_backtest(df_gas$Return, df_gas$VaR_GAS_2, df_gas$ES_GAS_2, vol_gas, 0.025)
Back_ES_CoC_GAS_5 <- cc_backtest(df_gas$Return, df_gas$VaR_GAS_5, df_gas$ES_GAS_5, vol_gas, 0.05)

Back_ES_ER_GAS_1 <- er_backtest(df_gas$Return, df_gas$VaR_GAS_1, df_gas$ES_GAS_1, vol_gas)
Back_ES_ER_GAS_2 <- er_backtest(df_gas$Return, df_gas$VaR_GAS_2, df_gas$ES_GAS_2, vol_gas)
Back_ES_ER_GAS_5 <- er_backtest(df_gas$Return, df_gas$VaR_GAS_5, df_gas$ES_GAS_5, vol_gas)

Back_ES_ESR_GAS_1_V1 <- esr_backtest(df_gas$Return, df_gas$VaR_GAS_1, df_gas$ES_GAS_1, alpha=0.01, version=1, B=0)
Back_ES_ESR_GAS_2_V1 <- esr_backtest(df_gas$Return, df_gas$VaR_GAS_2, df_gas$ES_GAS_2, alpha=0.025, version=1, B=0)
Back_ES_ESR_GAS_5_V1 <- esr_backtest(df_gas$Return, df_gas$VaR_GAS_5, df_gas$ES_GAS_5, alpha=0.05, version=1, B=0)


# GARCH-LSTM ----------------------------------------------------------------------------
sigma2_garch_lstm <- read_csv("DF_PREDS/GARCH_LSTM_T101_tst_carlos_1.csv", show_col_types = FALSE) %>%
  pull(Prediction)

res_lstm <- read_csv("Res/GARCH_LSTM_residuals_in_sample_T101_tst_carlos_1_padronizado_COM_MEAN.csv", show_col_types = FALSE) %>%
  pull(Residual) 

returns <- read_csv("sigma2_ajustado_e_previsto_completo.csv", show_col_types = FALSE) %>%
  pull(Returns)

n_ins <- 1500
n_oos <- 2000

VaR_GARCH_LSTM_1 <- VaR_GARCH_LSTM_2 <- VaR_GARCH_LSTM_5 <- numeric(n_oos)
ES_GARCH_LSTM_1  <- ES_GARCH_LSTM_2  <- ES_GARCH_LSTM_5  <- numeric(n_oos)
r_oos <- numeric(n_oos)

q_1 <- quantile(res_lstm, 0.01, na.rm = TRUE)
q_2 <- quantile(res_lstm, 0.025, na.rm = TRUE)
q_5 <- quantile(res_lstm, 0.05, na.rm = TRUE)

for (i in 1:n_oos) {
  returns_window <- returns[i:(i + n_ins - 1)]
  mu <- mean(returns_window)
  returns_c <- scale(returns_window, scale = FALSE)
  
  sigma_t <- sigma2_garch_lstm[i]
  
  # VaR
  VaR_GARCH_LSTM_1[i] <- mu + sqrt(sigma_t) * q_1
  VaR_GARCH_LSTM_2[i] <- mu + sqrt(sigma_t) * q_2
  VaR_GARCH_LSTM_5[i] <- mu + sqrt(sigma_t) * q_5

  # ES
  ES_GARCH_LSTM_1[i] <- mean(returns_window[returns_window < VaR_GARCH_LSTM_1[i]])
  ES_GARCH_LSTM_2[i] <- mean(returns_window[returns_window < VaR_GARCH_LSTM_2[i]])
  ES_GARCH_LSTM_5[i] <- mean(returns_window[returns_window < VaR_GARCH_LSTM_5[i]])
  
  r_oos[i] <- returns[i + n_ins]
}

df_garch_lstm <- tibble(
  Date = df$Date[(n_ins + 1):(n_ins + n_oos)],
  Return = r_oos,
  VaR_GARCH_LSTM_1 = VaR_GARCH_LSTM_1,
  ES_GARCH_LSTM_1 = ES_GARCH_LSTM_1,
  VaR_GARCH_LSTM_2_5 = VaR_GARCH_LSTM_2,
  ES_GARCH_LSTM_2_5 = ES_GARCH_LSTM_2,
  VaR_GARCH_LSTM_5 = VaR_GARCH_LSTM_5,
  ES_GARCH_LSTM_5 = ES_GARCH_LSTM_5
) %>%
  mutate(
    Exceed_VaR_GARCH_LSTM_1 = Return < VaR_GARCH_LSTM_1,
    Exceed_VaR_GARCH_LSTM_2 = Return < VaR_GARCH_LSTM_2_5,
    Exceed_VaR_GARCH_LSTM_5 = Return < VaR_GARCH_LSTM_5
  ) %>%
  drop_na()

table(df_garch_lstm$Exceed_VaR_GARCH_LSTM_1) # 42    espera: 20
table(df_garch_lstm$Exceed_VaR_GARCH_LSTM_2) # 72    espera: 50
table(df_garch_lstm$Exceed_VaR_GARCH_LSTM_5) # 98    espera: 100

# VaR Backtest: UC, CC, DQ, VQR
Back_VaR_GARCH_LSTM_1 <- BacktestVaR(df_garch_lstm$Return, df_garch_lstm$VaR_GARCH_LSTM_1, 0.01)
Back_VaR_GARCH_LSTM_2 <- BacktestVaR(df_garch_lstm$Return, df_garch_lstm$VaR_GARCH_LSTM_2_5, 0.025)
Back_VaR_GARCH_LSTM_5 <- BacktestVaR(df_garch_lstm$Return, df_garch_lstm$VaR_GARCH_LSTM_5, 0.05)

VaR_VQR(df_garch_lstm$Return, df_garch_lstm$VaR_GARCH_LSTM_1, 0.01)
VaR_VQR(df_garch_lstm$Return, df_garch_lstm$VaR_GARCH_LSTM_2_5, 0.025)
VaR_VQR(df_garch_lstm$Return, df_garch_lstm$VaR_GARCH_LSTM_5, 0.05)

Back_VaR_QL_GARCH_LSTM_1 <- Back_VaR_GARCH_LSTM_1$Loss$Loss
Back_VaR_QL_GARCH_LSTM_2 <- Back_VaR_GARCH_LSTM_2$Loss$Loss
Back_VaR_QL_GARCH_LSTM_5 <- Back_VaR_GARCH_LSTM_5$Loss$Loss

Back_VaR_FZ_GARCH_LSTM_1 <- mean(FZLoss(df_garch_lstm$Return, df_garch_lstm$VaR_GARCH_LSTM_1, df_garch_lstm$ES_GARCH_LSTM_1, 0.01))
Back_VaR_FZ_GARCH_LSTM_2 <- mean(FZLoss(df_garch_lstm$Return, df_garch_lstm$VaR_GARCH_LSTM_2_5, df_garch_lstm$ES_GARCH_LSTM_2_5, 0.025))
Back_VaR_FZ_GARCH_LSTM_5 <- mean(FZLoss(df_garch_lstm$Return, df_garch_lstm$VaR_GARCH_LSTM_5, df_garch_lstm$ES_GARCH_LSTM_5, 0.05))

Back_VaR_NZ_GARCH_LSTM_1 <- mean(NZ_deprecated(df_garch_lstm$VaR_GARCH_LSTM_1, df_garch_lstm$ES_GARCH_LSTM_1, df_garch_lstm$Return, 0.01))
Back_VaR_NZ_GARCH_LSTM_2 <- mean(NZ_deprecated(df_garch_lstm$VaR_GARCH_LSTM_2_5, df_garch_lstm$ES_GARCH_LSTM_2_5, df_garch_lstm$Return, 0.025))
Back_VaR_NZ_GARCH_LSTM_5 <- mean(NZ_deprecated(df_garch_lstm$VaR_GARCH_LSTM_5, df_garch_lstm$ES_GARCH_LSTM_5, df_garch_lstm$Return, 0.05))

Back_VaR_AL_GARCH_LSTM_1 <- AL_deprecated(df_garch_lstm$VaR_GARCH_LSTM_1, df_garch_lstm$ES_GARCH_LSTM_1, df_garch_lstm$Return, 0.01)
Back_VaR_AL_GARCH_LSTM_1 <- mean(Back_VaR_AL_GARCH_LSTM_1)
Back_VaR_AL_GARCH_LSTM_2 <- AL_deprecated(df_garch_lstm$VaR_GARCH_LSTM_2_5, df_garch_lstm$ES_GARCH_LSTM_2_5, df_garch_lstm$Return, 0.025)
Back_VaR_AL_GARCH_LSTM_2 <- mean(Back_VaR_AL_GARCH_LSTM_2)
Back_VaR_AL_GARCH_LSTM_5 <- AL_deprecated(df_garch_lstm$VaR_GARCH_LSTM_5, df_garch_lstm$ES_GARCH_LSTM_5, df_garch_lstm$Return, 0.05)
Back_VaR_AL_GARCH_LSTM_5 <- mean(Back_VaR_AL_GARCH_LSTM_5)


df_scores_garch_lstm <- data.frame(
  Nível = c("1%", "2.5%", "5%"),
  QL = c(Back_VaR_QL_GARCH_LSTM_1, Back_VaR_QL_GARCH_LSTM_2, Back_VaR_QL_GARCH_LSTM_5),
  FZ = c(Back_VaR_FZ_GARCH_LSTM_1, Back_VaR_FZ_GARCH_LSTM_2, Back_VaR_FZ_GARCH_LSTM_5),
  NZ = c(Back_VaR_NZ_GARCH_LSTM_1, Back_VaR_NZ_GARCH_LSTM_2, Back_VaR_NZ_GARCH_LSTM_5)
)

# ES Backtest: CoC, ER, ESR
vol_garch_lstm <- sqrt(sigma2_garch_lstm[1:nrow(df_garch_lstm)])

Back_ES_CoC_GARCH_LSTM_1 <- cc_backtest(df_garch_lstm$Return, df_garch_lstm$VaR_GARCH_LSTM_1, df_garch_lstm$ES_GARCH_LSTM_1, vol_garch_lstm, 0.01)
Back_ES_CoC_GARCH_LSTM_2 <- cc_backtest(df_garch_lstm$Return, df_garch_lstm$VaR_GARCH_LSTM_2_5, df_garch_lstm$ES_GARCH_LSTM_2_5, vol_garch_lstm, 0.025)
Back_ES_CoC_GARCH_LSTM_5 <- cc_backtest(df_garch_lstm$Return, df_garch_lstm$VaR_GARCH_LSTM_5, df_garch_lstm$ES_GARCH_LSTM_5, vol_garch_lstm, 0.05)

Back_ES_ER_GARCH_LSTM_1 <- er_backtest(df_garch_lstm$Return, df_garch_lstm$VaR_GARCH_LSTM_1, df_garch_lstm$ES_GARCH_LSTM_1, sigma2_garch_lstm)
Back_ES_ER_GARCH_LSTM_2 <- er_backtest(df_garch_lstm$Return, df_garch_lstm$VaR_GARCH_LSTM_2_5, df_garch_lstm$ES_GARCH_LSTM_2_5, sigma2_garch_lstm)
Back_ES_ER_GARCH_LSTM_5 <- er_backtest(df_garch_lstm$Return, df_garch_lstm$VaR_GARCH_LSTM_5, df_garch_lstm$ES_GARCH_LSTM_5, sigma2_garch_lstm)

Back_ES_ESR_GARCH_LSTM_1_V1 <- esr_backtest(df_garch_lstm$Return, df_garch_lstm$VaR_GARCH_LSTM_1, df_garch_lstm$ES_GARCH_LSTM_1, alpha=0.01, version=1, B=0)
Back_ES_ESR_GARCH_LSTM_2_V1 <- esr_backtest(df_garch_lstm$Return, df_garch_lstm$VaR_GARCH_LSTM_2_5, df_garch_lstm$ES_GARCH_LSTM_2_5, alpha=0.025, version=1, B=0)
Back_ES_ESR_GARCH_LSTM_5_V1 <- esr_backtest(df_garch_lstm$Return, df_garch_lstm$VaR_GARCH_LSTM_5, df_garch_lstm$ES_GARCH_LSTM_5, alpha=0.05, version=1, B=0)

# 1%
ggplot(df_garch_lstm, aes(x = Date)) +
  geom_line(aes(y = Return), color = "black", alpha = 0.6, linewidth = 0.4) +
  geom_line(aes(y = VaR_GARCH_LSTM_1), color = "red", linewidth = 0.7) +
  geom_point(data = subset(df_garch_lstm, Exceed_VaR_GARCH_LSTM_1),
             aes(y = Return),
             color = "red", size = 1.5, shape = 4) +
  labs(title = "VaR GARCH-LSTM - Nível de 1%",
       x = "Data", y = "Retorno") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# 5%
ggplot(df_garch_lstm, aes(x = Date)) +
  geom_line(aes(y = Return), color = "black", alpha = 0.6, linewidth = 0.4) +
  geom_line(aes(y = VaR_GARCH_LSTM_5), color = "red", linewidth = 0.7) +
  geom_point(data = subset(df_garch_lstm, Exceed_VaR_GARCH_LSTM_5),
             aes(y = Return),
             color = "red", size = 1.5, shape = 4) +
  labs(title = "VaR GARCH-LSTM - Nível de 5%",
       x = "Data", y = "Retorno") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# MSGARCH-LSTM ----------------------------------------------------------------------------
sigma2_msgarch_lstm <- read_csv("DF_PREDS/MSGARCH_LSTM_T101_tst_carlos_1.csv", show_col_types = FALSE) %>%
  pull(Prediction)

res_msgarch_lstm <- read_csv("Res/MSGARCH_LSTM_residuals_in_sample_T101_tst_carlos_1_padronizado_COM_MEAN.csv", show_col_types = FALSE) %>%
  pull(Residual) 

returns <- read_csv("sigma2_ajustado_e_previsto_completo.csv", show_col_types = FALSE) %>%
  pull(Returns)

n_ins <- 1500
n_oos <- 2000

VaR_MSGARCH_LSTM_1 <- VaR_MSGARCH_LSTM_2 <- VaR_MSGARCH_LSTM_5 <- numeric(n_oos)
ES_MSGARCH_LSTM_1  <- ES_MSGARCH_LSTM_2  <- ES_MSGARCH_LSTM_5  <- numeric(n_oos)
r_oos <- numeric(n_oos)

q_1 <- quantile(res_msgarch_lstm, 0.01, na.rm = TRUE)
q_2 <- quantile(res_msgarch_lstm, 0.025, na.rm = TRUE)
q_5 <- quantile(res_msgarch_lstm, 0.05, na.rm = TRUE)

for (i in 1:n_oos) {
  returns_window <- returns[i:(i + n_ins - 1)]
  mu <- mean(returns_window)
  returns_c <- scale(returns_window, scale = FALSE)
  
  sigma_t <- sigma2_msgarch_lstm[i]
  
  # VaR
  VaR_MSGARCH_LSTM_1[i] <- mu + sqrt(sigma_t) * q_1
  VaR_MSGARCH_LSTM_2[i] <- mu + sqrt(sigma_t) * q_2
  VaR_MSGARCH_LSTM_5[i] <- mu + sqrt(sigma_t) * q_5
  
  # ES
  ES_MSGARCH_LSTM_1[i] <- mean(returns_window[returns_window < VaR_MSGARCH_LSTM_1[i]])
  ES_MSGARCH_LSTM_2[i] <- mean(returns_window[returns_window < VaR_MSGARCH_LSTM_2[i]])
  ES_MSGARCH_LSTM_5[i] <- mean(returns_window[returns_window < VaR_MSGARCH_LSTM_5[i]])
  
  r_oos[i] <- returns[i + n_ins]
}

df_msgarch_lstm <- tibble(
  Date = df$Date[(n_ins + 1):(n_ins + n_oos)],
  Return = r_oos,
  VaR_MSGARCH_LSTM_1 = VaR_MSGARCH_LSTM_1,
  ES_MSGARCH_LSTM_1 = ES_MSGARCH_LSTM_1,
  VaR_MSGARCH_LSTM_2_5 = VaR_MSGARCH_LSTM_2,
  ES_MSGARCH_LSTM_2_5 = ES_MSGARCH_LSTM_2,
  VaR_MSGARCH_LSTM_5 = VaR_MSGARCH_LSTM_5,
  ES_MSGARCH_LSTM_5 = ES_MSGARCH_LSTM_5
) %>%
  mutate(
    Exceed_VaR_MSGARCH_LSTM_1 = Return < VaR_MSGARCH_LSTM_1,
    Exceed_VaR_MSGARCH_LSTM_2 = Return < VaR_MSGARCH_LSTM_2_5,
    Exceed_VaR_MSGARCH_LSTM_5 = Return < VaR_MSGARCH_LSTM_5
  ) %>%
  drop_na()

table(df_msgarch_lstm$Exceed_VaR_MSGARCH_LSTM_1) # 32    espera: 20
table(df_msgarch_lstm$Exceed_VaR_MSGARCH_LSTM_2) # 68    espera: 50
table(df_msgarch_lstm$Exceed_VaR_MSGARCH_LSTM_5) # 97    espera: 100

# VaR Backtest: UC, CC, DQ
Back_VaR_MSGARCH_LSTM_1 <- BacktestVaR(df_msgarch_lstm$Return, df_msgarch_lstm$VaR_MSGARCH_LSTM_1, 0.01)
Back_VaR_MSGARCH_LSTM_2 <- BacktestVaR(df_msgarch_lstm$Return, df_msgarch_lstm$VaR_MSGARCH_LSTM_2_5, 0.025)
Back_VaR_MSGARCH_LSTM_5 <- BacktestVaR(df_msgarch_lstm$Return, df_msgarch_lstm$VaR_MSGARCH_LSTM_5, 0.05)

VaR_VQR(df_msgarch_lstm$Return, df_msgarch_lstm$VaR_MSGARCH_LSTM_1, 0.01)
VaR_VQR(df_msgarch_lstm$Return, df_msgarch_lstm$VaR_MSGARCH_LSTM_2_5, 0.025)
VaR_VQR(df_msgarch_lstm$Return, df_msgarch_lstm$VaR_MSGARCH_LSTM_5, 0.05)

Back_VaR_QL_MSGARCH_LSTM_1 <- Back_VaR_MSGARCH_LSTM_1$Loss$Loss
Back_VaR_QL_MSGARCH_LSTM_2 <- Back_VaR_MSGARCH_LSTM_2$Loss$Loss
Back_VaR_QL_MSGARCH_LSTM_5 <- Back_VaR_MSGARCH_LSTM_5$Loss$Loss

Back_VaR_FZ_MSGARCH_LSTM_1 <- mean(FZLoss(df_msgarch_lstm$Return, df_msgarch_lstm$VaR_MSGARCH_LSTM_1, df_msgarch_lstm$ES_MSGARCH_LSTM_1, 0.01))
Back_VaR_FZ_MSGARCH_LSTM_2 <- mean(FZLoss(df_msgarch_lstm$Return, df_msgarch_lstm$VaR_MSGARCH_LSTM_2_5, df_msgarch_lstm$ES_MSGARCH_LSTM_2_5, 0.025))
Back_VaR_FZ_MSGARCH_LSTM_5 <- mean(FZLoss(df_msgarch_lstm$Return, df_msgarch_lstm$VaR_MSGARCH_LSTM_5, df_msgarch_lstm$ES_MSGARCH_LSTM_5, 0.05))

Back_VaR_NZ_MSGARCH_LSTM_1 <- mean(NZ_deprecated(df_msgarch_lstm$VaR_MSGARCH_LSTM_1, df_msgarch_lstm$ES_MSGARCH_LSTM_1, df_msgarch_lstm$Return, 0.01))
Back_VaR_NZ_MSGARCH_LSTM_2 <- mean(NZ_deprecated(df_msgarch_lstm$VaR_MSGARCH_LSTM_2_5, df_msgarch_lstm$ES_MSGARCH_LSTM_2_5, df_msgarch_lstm$Return, 0.025))
Back_VaR_NZ_MSGARCH_LSTM_5 <- mean(NZ_deprecated(df_msgarch_lstm$VaR_MSGARCH_LSTM_5, df_msgarch_lstm$ES_MSGARCH_LSTM_5, df_msgarch_lstm$Return, 0.05))

Back_VaR_AL_MSGARCH_LSTM_1 <- AL_deprecated(df_msgarch_lstm$VaR_MSGARCH_LSTM_1, df_msgarch_lstm$ES_MSGARCH_LSTM_1, df_msgarch_lstm$Return, 0.01)
Back_VaR_AL_MSGARCH_LSTM_1 <- mean(Back_VaR_AL_MSGARCH_LSTM_1)
Back_VaR_AL_MSGARCH_LSTM_2 <- AL_deprecated(df_msgarch_lstm$VaR_MSGARCH_LSTM_2_5, df_msgarch_lstm$ES_MSGARCH_LSTM_2_5, df_msgarch_lstm$Return, 0.025)
Back_VaR_AL_MSGARCH_LSTM_2 <- mean(Back_VaR_AL_MSGARCH_LSTM_2)
Back_VaR_AL_MSGARCH_LSTM_5 <- AL_deprecated(df_msgarch_lstm$VaR_MSGARCH_LSTM_5, df_msgarch_lstm$ES_MSGARCH_LSTM_5, df_msgarch_lstm$Return, 0.05)
Back_VaR_AL_MSGARCH_LSTM_5 <- mean(Back_VaR_AL_MSGARCH_LSTM_5)

df_scores_msgarch_lstm <- data.frame(
  Nível = c("1%", "2.5%", "5%"),
  QL = c(Back_VaR_QL_MSGARCH_LSTM_1, Back_VaR_QL_MSGARCH_LSTM_2, Back_VaR_QL_MSGARCH_LSTM_5),
  FZ = c(Back_VaR_FZ_MSGARCH_LSTM_1, Back_VaR_FZ_MSGARCH_LSTM_2, Back_VaR_FZ_MSGARCH_LSTM_5),
  NZ = c(Back_VaR_NZ_MSGARCH_LSTM_1, Back_VaR_NZ_MSGARCH_LSTM_2, Back_VaR_NZ_MSGARCH_LSTM_5)
)

# ES Backtest: CoC, ER, ESR
vol_msgarch_lstm <- sqrt(sigma2_msgarch_lstm[1:nrow(df_msgarch_lstm)])

Back_ES_CoC_MSGARCH_LSTM_1 <- cc_backtest(df_msgarch_lstm$Return, df_msgarch_lstm$VaR_MSGARCH_LSTM_1, df_msgarch_lstm$ES_MSGARCH_LSTM_1, vol_msgarch_lstm, 0.01)
Back_ES_CoC_MSGARCH_LSTM_2 <- cc_backtest(df_msgarch_lstm$Return, df_msgarch_lstm$VaR_MSGARCH_LSTM_2_5, df_msgarch_lstm$ES_MSGARCH_LSTM_2_5, vol_msgarch_lstm, 0.025)
Back_ES_CoC_MSGARCH_LSTM_5 <- cc_backtest(df_msgarch_lstm$Return, df_msgarch_lstm$VaR_MSGARCH_LSTM_5, df_msgarch_lstm$ES_MSGARCH_LSTM_5, vol_msgarch_lstm, 0.05)

Back_ES_ER_MSGARCH_LSTM_1 <- er_backtest(df_msgarch_lstm$Return, df_msgarch_lstm$VaR_MSGARCH_LSTM_1, df_msgarch_lstm$ES_MSGARCH_LSTM_1, sigma2_msgarch_lstm)
Back_ES_ER_MSGARCH_LSTM_2 <- er_backtest(df_msgarch_lstm$Return, df_msgarch_lstm$VaR_MSGARCH_LSTM_2_5, df_msgarch_lstm$ES_MSGARCH_LSTM_2_5, sigma2_msgarch_lstm)
Back_ES_ER_MSGARCH_LSTM_5 <- er_backtest(df_msgarch_lstm$Return, df_msgarch_lstm$VaR_MSGARCH_LSTM_5, df_msgarch_lstm$ES_MSGARCH_LSTM_5, sigma2_msgarch_lstm)

Back_ES_ESR_MSGARCH_LSTM_1_V1 <- esr_backtest(df_msgarch_lstm$Return, df_msgarch_lstm$VaR_MSGARCH_LSTM_1, df_msgarch_lstm$ES_MSGARCH_LSTM_1, alpha=0.01, version=1, B=0)
Back_ES_ESR_MSGARCH_LSTM_2_V1 <- esr_backtest(df_msgarch_lstm$Return, df_msgarch_lstm$VaR_MSGARCH_LSTM_2_5, df_msgarch_lstm$ES_MSGARCH_LSTM_2_5, alpha=0.025, version=1, B=0)
Back_ES_ESR_MSGARCH_LSTM_5_V1 <- esr_backtest(df_msgarch_lstm$Return, df_msgarch_lstm$VaR_MSGARCH_LSTM_5, df_msgarch_lstm$ES_MSGARCH_LSTM_5, alpha=0.05, version=1, B=0)

# GAS-LSTM ----------------------------------------------------------------------------
sigma2_gas_lstm <- read_csv("DF_PREDS/GAS_LSTM_T101_tst_carlos_1.csv", show_col_types = FALSE) %>%
  pull(Prediction)

res_gas_lstm <- read_csv("Res/GAS_LSTM_residuals_in_sample_T101_tst_carlos_1_padronizado_COM_MEAN.csv", show_col_types = FALSE) %>%
  pull(Residual) 

returns <- read_csv("sigma2_ajustado_e_previsto_completo.csv", show_col_types = FALSE) %>%
  pull(Returns)

n_ins <- 1500
n_oos <- 2000

VaR_GAS_LSTM_1 <- VaR_GAS_LSTM_2 <- VaR_GAS_LSTM_5 <- numeric(n_oos)
ES_GAS_LSTM_1  <- ES_GAS_LSTM_2  <- ES_GAS_LSTM_5  <- numeric(n_oos)
r_oos <- numeric(n_oos)

q_1 <- quantile(res_gas_lstm, 0.01, na.rm = TRUE)
q_2 <- quantile(res_gas_lstm, 0.025, na.rm = TRUE)
q_5 <- quantile(res_gas_lstm, 0.05, na.rm = TRUE)

for (i in 1:n_oos) {
  returns_window <- returns[i:(i + n_ins - 1)]
  mu <- mean(returns_window)
  returns_c <- scale(returns_window, scale = FALSE)
  
  sigma_t <- sigma2_gas_lstm[i]
  
  # VaR
  VaR_GAS_LSTM_1[i] <- mu + sqrt(sigma_t) * q_1
  VaR_GAS_LSTM_2[i] <- mu + sqrt(sigma_t) * q_2
  VaR_GAS_LSTM_5[i] <- mu + sqrt(sigma_t) * q_5
  
  # ES
  ES_GAS_LSTM_1[i] <- mean(returns_window[returns_window < VaR_GAS_LSTM_1[i]])
  ES_GAS_LSTM_2[i] <- mean(returns_window[returns_window < VaR_GAS_LSTM_2[i]])
  ES_GAS_LSTM_5[i] <- mean(returns_window[returns_window < VaR_GAS_LSTM_5[i]])
  
  r_oos[i] <- returns[i + n_ins]
}

df_gas_lstm <- tibble(
  Date = df$Date[(n_ins + 1):(n_ins + n_oos)],
  Return = r_oos,
  VaR_GAS_LSTM_1 = VaR_GAS_LSTM_1,
  ES_GAS_LSTM_1 = ES_GAS_LSTM_1,
  VaR_GAS_LSTM_2_5 = VaR_GAS_LSTM_2,
  ES_GAS_LSTM_2_5 = ES_GAS_LSTM_2,
  VaR_GAS_LSTM_5 = VaR_GAS_LSTM_5,
  ES_GAS_LSTM_5 = ES_GAS_LSTM_5
) %>%
  mutate(
    Exceed_VaR_GAS_LSTM_1 = Return < VaR_GAS_LSTM_1,
    Exceed_VaR_GAS_LSTM_2 = Return < VaR_GAS_LSTM_2_5,
    Exceed_VaR_GAS_LSTM_5 = Return < VaR_GAS_LSTM_5
  ) %>%
  drop_na()

table(df_gas_lstm$Exceed_VaR_GAS_LSTM_1) # 39    espera: 20
table(df_gas_lstm$Exceed_VaR_GAS_LSTM_2) # 76    espera: 50
table(df_gas_lstm$Exceed_VaR_GAS_LSTM_5) # 99    espera: 100

# VaR Backtest: UC, CC, DQ, VQR
Back_VaR_GAS_LSTM_1 <- BacktestVaR(df_gas_lstm$Return, df_gas_lstm$VaR_GAS_LSTM_1, 0.01)
Back_VaR_GAS_LSTM_2 <- BacktestVaR(df_gas_lstm$Return, df_gas_lstm$VaR_GAS_LSTM_2_5, 0.025)
Back_VaR_GAS_LSTM_5 <- BacktestVaR(df_gas_lstm$Return, df_gas_lstm$VaR_GAS_LSTM_5, 0.05)

VaR_VQR(df_gas_lstm$Return, df_gas_lstm$VaR_GAS_LSTM_1, 0.01)
VaR_VQR(df_gas_lstm$Return, df_gas_lstm$VaR_GAS_LSTM_2_5, 0.025)
VaR_VQR(df_gas_lstm$Return, df_gas_lstm$VaR_GAS_LSTM_5, 0.05)

Back_VaR_QL_GAS_LSTM_1 <- Back_VaR_GAS_LSTM_1$Loss$Loss
Back_VaR_QL_GAS_LSTM_2 <- Back_VaR_GAS_LSTM_2$Loss$Loss
Back_VaR_QL_GAS_LSTM_5 <- Back_VaR_GAS_LSTM_5$Loss$Loss

Back_VaR_FZ_GAS_LSTM_1 <- mean(FZLoss(df_gas_lstm$Return, df_gas_lstm$VaR_GAS_LSTM_1, df_gas_lstm$ES_GAS_LSTM_1, 0.01))
Back_VaR_FZ_GAS_LSTM_2 <- mean(FZLoss(df_gas_lstm$Return, df_gas_lstm$VaR_GAS_LSTM_2_5, df_gas_lstm$ES_GAS_LSTM_2_5, 0.025))
Back_VaR_FZ_GAS_LSTM_5 <- mean(FZLoss(df_gas_lstm$Return, df_gas_lstm$VaR_GAS_LSTM_5, df_gas_lstm$ES_GAS_LSTM_5, 0.05))

Back_VaR_NZ_GAS_LSTM_1 <- mean(NZ_deprecated(df_gas_lstm$VaR_GAS_LSTM_1, df_gas_lstm$ES_GAS_LSTM_1, df_gas_lstm$Return, 0.01))
Back_VaR_NZ_GAS_LSTM_2 <- mean(NZ_deprecated(df_gas_lstm$VaR_GAS_LSTM_2_5, df_gas_lstm$ES_GAS_LSTM_2_5, df_gas_lstm$Return, 0.025))
Back_VaR_NZ_GAS_LSTM_5 <- mean(NZ_deprecated(df_gas_lstm$VaR_GAS_LSTM_5, df_gas_lstm$ES_GAS_LSTM_5, df_gas_lstm$Return, 0.05))

Back_VaR_AL_GAS_LSTM_1 <- AL_deprecated(df_gas_lstm$VaR_GAS_LSTM_1, df_gas_lstm$ES_GAS_LSTM_1, df_gas_lstm$Return, 0.01)
Back_VaR_AL_GAS_LSTM_1 <- mean(Back_VaR_AL_GAS_LSTM_1)
Back_VaR_AL_GAS_LSTM_2 <- AL_deprecated(df_gas_lstm$VaR_GAS_LSTM_2_5, df_gas_lstm$ES_GAS_LSTM_2_5, df_gas_lstm$Return, 0.025)
Back_VaR_AL_GAS_LSTM_2 <- mean(Back_VaR_AL_GAS_LSTM_2)
Back_VaR_AL_GAS_LSTM_5 <- AL_deprecated(df_gas_lstm$VaR_GAS_LSTM_5, df_gas_lstm$ES_GAS_LSTM_5, df_gas_lstm$Return, 0.05)
Back_VaR_AL_GAS_LSTM_5 <- mean(Back_VaR_AL_GAS_LSTM_5)

df_scores_gas_lstm <- data.frame(
  Nível = c("1%", "2.5%", "5%"),
  QL = c(Back_VaR_QL_GAS_LSTM_1, Back_VaR_QL_GAS_LSTM_2, Back_VaR_QL_GAS_LSTM_5),
  FZ = c(Back_VaR_FZ_GAS_LSTM_1, Back_VaR_FZ_GAS_LSTM_2, Back_VaR_FZ_GAS_LSTM_5),
  NZ = c(Back_VaR_NZ_GAS_LSTM_1, Back_VaR_NZ_GAS_LSTM_2, Back_VaR_NZ_GAS_LSTM_5)
)

# ES Backtest: CoC, ER, ESR
vol_gas_lstm <- sqrt(sigma2_gas_lstm[1:nrow(df_gas_lstm)])
Back_ES_CoC_GAS_LSTM_1 <- cc_backtest(df_gas_lstm$Return, df_gas_lstm$VaR_GAS_LSTM_1, df_gas_lstm$ES_GAS_LSTM_1, vol_gas_lstm, 0.01)
Back_ES_CoC_GAS_LSTM_2 <- cc_backtest(df_gas_lstm$Return, df_gas_lstm$VaR_GAS_LSTM_2_5, df_gas_lstm$ES_GAS_LSTM_2_5, vol_gas_lstm, 0.025)
Back_ES_CoC_GAS_LSTM_5 <- cc_backtest(df_gas_lstm$Return, df_gas_lstm$VaR_GAS_LSTM_5, df_gas_lstm$ES_GAS_LSTM_5, vol_gas_lstm, 0.05)

Back_ES_ER_GAS_LSTM_1 <- er_backtest(df_gas_lstm$Return, df_gas_lstm$VaR_GAS_LSTM_1, df_gas_lstm$ES_GAS_LSTM_1, vol_gas_lstm)
Back_ES_ER_GAS_LSTM_2 <- er_backtest(df_gas_lstm$Return, df_gas_lstm$VaR_GAS_LSTM_2_5, df_gas_lstm$ES_GAS_LSTM_2_5, vol_gas_lstm)
Back_ES_ER_GAS_LSTM_5 <- er_backtest(df_gas_lstm$Return, df_gas_lstm$VaR_GAS_LSTM_5, df_gas_lstm$ES_GAS_LSTM_5, vol_gas_lstm)

Back_ES_ESR_GAS_LSTM_1_V1 <- esr_backtest(df_gas_lstm$Return, df_gas_lstm$VaR_GAS_LSTM_1, df_gas_lstm$ES_GAS_LSTM_1, alpha=0.01, version=1, B=0)
Back_ES_ESR_GAS_LSTM_2_V1 <- esr_backtest(df_gas_lstm$Return, df_gas_lstm$VaR_GAS_LSTM_2_5, df_gas_lstm$ES_GAS_LSTM_2_5, alpha=0.025, version=1, B=0)
Back_ES_ESR_GAS_LSTM_5_V1 <- esr_backtest(df_gas_lstm$Return, df_gas_lstm$VaR_GAS_LSTM_5, df_gas_lstm$ES_GAS_LSTM_5, alpha=0.05, version=1, B=0)


# 1%
ggplot(df_gas_lstm, aes(x = Date)) +
  geom_line(aes(y = Return), color = "black", alpha = 0.6, linewidth = 0.4) +
  geom_line(aes(y = VaR_GAS_LSTM_1), color = "red", linewidth = 0.7) +
  geom_point(data = subset(df_gas_lstm, Exceed_VaR_GAS_LSTM_1),
             aes(y = Return),
             color = "red", size = 1.5, shape = 4) +
  labs(title = "VaR GAS-LSTM - Nível de 1%",
       x = "Data", y = "Retorno") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# 5%
ggplot(df_gas_lstm, aes(x = Date)) +
  geom_line(aes(y = Return), color = "black", alpha = 0.6, linewidth = 0.4) +
  geom_line(aes(y = VaR_GAS_LSTM_5), color = "red", linewidth = 0.7) +
  geom_point(data = subset(df_gas_lstm, Exceed_VaR_GAS_LSTM_5),
             aes(y = Return),
             color = "red", size = 1.5, shape = 4) +
  labs(title = "VaR GAS-LSTM - Nível de 5%",
       x = "Data", y = "Retorno") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# LSTM PURO -----------------------------------------------------------------------------
sigma2_lstm_puro <- read_csv("DF_PREDS/LSTM_puro_T101_tst_carlos_1.csv", show_col_types = FALSE) %>%
  pull(Prediction)

res_lstm_puro <- read_csv("Res/LSTM_puro_residuals_in_sample_T101_tst_carlos_1_padronizado_COM_MEAN.csv", show_col_types = FALSE) %>%
  pull(Residual) 

returns <- read_csv("sigma2_ajustado_e_previsto_completo.csv", show_col_types = FALSE) %>%
  pull(Returns)

n_ins <- 1500
n_oos <- 2000

VaR_LSTM_PURO_1 <- VaR_LSTM_PURO_2 <- VaR_LSTM_PURO_5 <- numeric(n_oos)
ES_LSTM_PURO_1  <- ES_LSTM_PURO_2  <- ES_LSTM_PURO_5  <- numeric(n_oos)
r_oos <- numeric(n_oos)

q_1 <- quantile(res_lstm_puro, 0.01, na.rm = TRUE)
q_2 <- quantile(res_lstm_puro, 0.025, na.rm = TRUE)
q_5 <- quantile(res_lstm_puro, 0.05, na.rm = TRUE)

for (i in 1:n_oos) {
  returns_window <- returns[i:(i + n_ins - 1)]
  mu <- mean(returns_window)
  returns_c <- scale(returns_window, scale = FALSE)
  
  sigma_t <- sigma2_lstm_puro[i]
  
  # VaR
  VaR_LSTM_PURO_1[i] <- mu + sqrt(sigma_t) * q_1
  VaR_LSTM_PURO_2[i] <- mu + sqrt(sigma_t) * q_2
  VaR_LSTM_PURO_5[i] <- mu + sqrt(sigma_t) * q_5
  
  # ES
  ES_LSTM_PURO_1[i] <- mean(returns_window[returns_window < VaR_LSTM_PURO_1[i]])
  ES_LSTM_PURO_2[i] <- mean(returns_window[returns_window < VaR_LSTM_PURO_2[i]])
  ES_LSTM_PURO_5[i] <- mean(returns_window[returns_window < VaR_LSTM_PURO_5[i]])
  
  r_oos[i] <- returns[i + n_ins]
}

df_lstm_puro <- tibble(
  Date = df$Date[(n_ins + 1):(n_ins + n_oos)],
  Return = r_oos,
  VaR_LSTM_PURO_1 = VaR_LSTM_PURO_1,
  ES_LSTM_PURO_1 = ES_LSTM_PURO_1,
  VaR_LSTM_PURO_2_5 = VaR_LSTM_PURO_2,
  ES_LSTM_PURO_2_5 = ES_LSTM_PURO_2,
  VaR_LSTM_PURO_5 = VaR_LSTM_PURO_5,
  ES_LSTM_PURO_5 = ES_LSTM_PURO_5
) %>%
  mutate(
    Exceed_VaR_LSTM_PURO_1 = Return < VaR_LSTM_PURO_1,
    Exceed_VaR_LSTM_PURO_2 = Return < VaR_LSTM_PURO_2_5,
    Exceed_VaR_LSTM_PURO_5 = Return < VaR_LSTM_PURO_5
  ) %>%
  drop_na()

table(df_lstm_puro$Exceed_VaR_LSTM_PURO_1) # 39    espera: 20
table(df_lstm_puro$Exceed_VaR_LSTM_PURO_2) # 74    espera: 50
table(df_lstm_puro$Exceed_VaR_LSTM_PURO_5) # 99    espera: 100

# VaR Backtest: UC, CC, DQ
Back_VaR_LSTM_PURO_1 <- BacktestVaR(df_lstm_puro$Return, df_lstm_puro$VaR_LSTM_PURO_1, 0.01)
Back_VaR_LSTM_PURO_2 <- BacktestVaR(df_lstm_puro$Return, df_lstm_puro$VaR_LSTM_PURO_2_5, 0.025)
Back_VaR_LSTM_PURO_5 <- BacktestVaR(df_lstm_puro$Return, df_lstm_puro$VaR_LSTM_PURO_5, 0.05)

VaR_VQR(df_lstm_puro$Return, df_lstm_puro$VaR_LSTM_PURO_1, 0.01)
VaR_VQR(df_lstm_puro$Return, df_lstm_puro$VaR_LSTM_PURO_2_5, 0.025)
VaR_VQR(df_lstm_puro$Return, df_lstm_puro$VaR_LSTM_PURO_5, 0.05)

Back_VaR_QL_LSTM_PURO_1 <- Back_VaR_LSTM_PURO_1$Loss$Loss
Back_VaR_QL_LSTM_PURO_2 <- Back_VaR_LSTM_PURO_2$Loss$Loss
Back_VaR_QL_LSTM_PURO_5 <- Back_VaR_LSTM_PURO_5$Loss$Loss

Back_VaR_FZ_LSTM_PURO_1 <- mean(FZLoss(df_lstm_puro$Return, df_lstm_puro$VaR_LSTM_PURO_1, df_lstm_puro$ES_LSTM_PURO_1, 0.01))
Back_VaR_FZ_LSTM_PURO_2 <- mean(FZLoss(df_lstm_puro$Return, df_lstm_puro$VaR_LSTM_PURO_2_5, df_lstm_puro$ES_LSTM_PURO_2_5, 0.025))
Back_VaR_FZ_LSTM_PURO_5 <- mean(FZLoss(df_lstm_puro$Return, df_lstm_puro$VaR_LSTM_PURO_5, df_lstm_puro$ES_LSTM_PURO_5, 0.05))

Back_VaR_NZ_LSTM_PURO_1 <- mean(NZ_deprecated(df_lstm_puro$VaR_LSTM_PURO_1, df_lstm_puro$ES_LSTM_PURO_1, df_lstm_puro$Return, 0.01))
Back_VaR_NZ_LSTM_PURO_2 <- mean(NZ_deprecated(df_lstm_puro$VaR_LSTM_PURO_2_5, df_lstm_puro$ES_LSTM_PURO_2_5, df_lstm_puro$Return, 0.025))
Back_VaR_NZ_LSTM_PURO_5 <- mean(NZ_deprecated(df_lstm_puro$VaR_LSTM_PURO_5, df_lstm_puro$ES_LSTM_PURO_5, df_lstm_puro$Return, 0.05))

Back_VaR_AL_LSTM_PURO_1 <- AL_deprecated(df_lstm_puro$VaR_LSTM_PURO_1, df_lstm_puro$ES_LSTM_PURO_1, df_lstm_puro$Return, 0.01)
Back_VaR_AL_LSTM_PURO_1 <- mean(Back_VaR_AL_LSTM_PURO_1)
Back_VaR_AL_LSTM_PURO_2 <- AL_deprecated(df_lstm_puro$VaR_LSTM_PURO_2_5, df_lstm_puro$ES_LSTM_PURO_2_5, df_lstm_puro$Return, 0.025)
Back_VaR_AL_LSTM_PURO_2 <- mean(Back_VaR_AL_LSTM_PURO_2)
Back_VaR_AL_LSTM_PURO_5 <- AL_deprecated(df_lstm_puro$VaR_LSTM_PURO_5, df_lstm_puro$ES_LSTM_PURO_5, df_lstm_puro$Return, 0.05)
Back_VaR_AL_LSTM_PURO_5 <- mean(Back_VaR_AL_LSTM_PURO_5)

df_scores_lstm_puro <- data.frame(
  Nível = c("1%", "2.5%", "5%"),
  QL = c(Back_VaR_QL_LSTM_PURO_1, Back_VaR_QL_LSTM_PURO_2, Back_VaR_QL_LSTM_PURO_5),
  FZ = c(Back_VaR_FZ_LSTM_PURO_1, Back_VaR_FZ_LSTM_PURO_2, Back_VaR_FZ_LSTM_PURO_5),
  NZ = c(Back_VaR_NZ_LSTM_PURO_1, Back_VaR_NZ_LSTM_PURO_2, Back_VaR_NZ_LSTM_PURO_5)
)

# ES Backtest: CoC, ER, ESR
vol_lstm_puro <- sqrt(sigma2_lstm_puro[1:nrow(df_lstm_puro)])

Back_ES_CoC_LSTM_PURO_1 <- cc_backtest(df_lstm_puro$Return, df_lstm_puro$VaR_LSTM_PURO_1, df_lstm_puro$ES_LSTM_PURO_1, vol_lstm_puro, 0.01)
Back_ES_CoC_LSTM_PURO_2 <- cc_backtest(df_lstm_puro$Return, df_lstm_puro$VaR_LSTM_PURO_2_5, df_lstm_puro$ES_LSTM_PURO_2_5, vol_lstm_puro, 0.025)
Back_ES_CoC_LSTM_PURO_5 <- cc_backtest(df_lstm_puro$Return, df_lstm_puro$VaR_LSTM_PURO_5, df_lstm_puro$ES_LSTM_PURO_5, vol_lstm_puro, 0.05)

Back_ES_ER_LSTM_PURO_1 <- er_backtest(df_lstm_puro$Return, df_lstm_puro$VaR_LSTM_PURO_1, df_lstm_puro$ES_LSTM_PURO_1, vol_lstm_puro)
Back_ES_ER_LSTM_PURO_2 <- er_backtest(df_lstm_puro$Return, df_lstm_puro$VaR_LSTM_PURO_2_5, df_lstm_puro$ES_LSTM_PURO_2_5, vol_lstm_puro)
Back_ES_ER_LSTM_PURO_5 <- er_backtest(df_lstm_puro$Return, df_lstm_puro$VaR_LSTM_PURO_5, df_lstm_puro$ES_LSTM_PURO_5, vol_lstm_puro)

Back_ES_ESR_LSTM_PURO_1_V1 <- esr_backtest(df_lstm_puro$Return, df_lstm_puro$VaR_LSTM_PURO_1, df_lstm_puro$ES_LSTM_PURO_1, alpha=0.01, version=1, B=0)
Back_ES_ESR_LSTM_PURO_2_V1 <- esr_backtest(df_lstm_puro$Return, df_lstm_puro$VaR_LSTM_PURO_2_5, df_lstm_puro$ES_LSTM_PURO_2_5, alpha=0.025, version=1, B=0)
Back_ES_ESR_LSTM_PURO_5_V1 <- esr_backtest(df_lstm_puro$Return, df_lstm_puro$VaR_LSTM_PURO_5, df_lstm_puro$ES_LSTM_PURO_5, alpha=0.05, version=1, B=0)


# Gráficos


# ARFIMA -------------------------------

# usa log-parkinson
sigma2_arfima <- read_csv("df_arfima.csv", show_col_types = FALSE) %>%
  pull(Sigma2_ARFIMA)

#res_arfima <- read_csv("df_arfima_insample.csv", show_col_types = FALSE) %>%
  pull(Residuals) 

res_arfima <- df_arfima_insample_logpks %>%
  select(Residuals)

returns <- read_csv("sigma2_ajustado_e_previsto_completo.csv", show_col_types = FALSE) %>%
  pull(Returns)

returns_ins <- returns[1:1500]
#sigma2_ajustada <- read.csv("df_arfima_insample.csv")$Sigma2_ARFIMA_ajustado
sigma2_ajustada <- df_arfima_insample_logpks %>%
  select(Sigma2_ARFIMA_ajustado_exp)

sigma_ajustada <- sqrt(sigma2_ajustada)
resid_arfima_manual <- returns_ins / sigma_ajustada

q_1 <- quantile(res_arfima, 0.01, na.rm = TRUE)
q_2 <- quantile(res_arfima, 0.025, na.rm = TRUE)
q_5 <- quantile(res_arfima, 0.05, na.rm = TRUE)

VaR_ARFIMA_1 <- VaR_ARFIMA_2 <- VaR_ARFIMA_5 <- numeric(n_oos)
ES_ARFIMA_1  <- ES_ARFIMA_2  <- ES_ARFIMA_5  <- numeric(n_oos)
r_oos <- numeric(n_oos)

for (i in 1:n_oos) {
  returns_window <- returns[i:(i + n_ins - 1)]
  mu <- mean(returns_window)
  
  sigma2_t <- sigma2_arfima[i]
  sigma_t <- sqrt(sigma2_t)
  
  # VaR
  VaR_ARFIMA_1[i] <- mu + sigma_t * q_1
  VaR_ARFIMA_2[i] <- mu + sigma_t * q_2
  VaR_ARFIMA_5[i] <- mu + sigma_t * q_5
  
  # ES
  ES_ARFIMA_1[i] <- mean(returns_window[returns_window < VaR_ARFIMA_1[i]])
  ES_ARFIMA_2[i] <- mean(returns_window[returns_window < VaR_ARFIMA_2[i]])
  ES_ARFIMA_5[i] <- mean(returns_window[returns_window < VaR_ARFIMA_5[i]])
  
  r_oos[i] <- returns[i + n_ins]
}

df_arfima_final <- tibble(
  Date = df$Date[(n_ins + 1):(n_ins + n_oos)],
  Return = r_oos,
  VaR_ARFIMA_1 = VaR_ARFIMA_1,
  ES_ARFIMA_1 = ES_ARFIMA_1,
  VaR_ARFIMA_2_5 = VaR_ARFIMA_2,
  ES_ARFIMA_2_5 = ES_ARFIMA_2,
  VaR_ARFIMA_5 = VaR_ARFIMA_5,
  ES_ARFIMA_5 = ES_ARFIMA_5
) %>%
  mutate(
    Exceed_VaR_ARFIMA_1 = Return < VaR_ARFIMA_1,
    Exceed_VaR_ARFIMA_2 = Return < VaR_ARFIMA_2_5,
    Exceed_VaR_ARFIMA_5 = Return < VaR_ARFIMA_5
  ) 

table(df_arfima_final$Exceed_VaR_ARFIMA_1)
table(df_arfima_final$Exceed_VaR_ARFIMA_2)
table(df_arfima_final$Exceed_VaR_ARFIMA_5)




# sem usar log-parkinson
sigma2_arfima <- read_csv("df_arfima_oos.csv", show_col_types = FALSE) %>%
  pull(Sigma2_ARFIMA)

res_arfima <- read_csv("df_arfima_insample_pkscru.csv", show_col_types = FALSE) %>%
  pull(Residuals) 

returns <- read_csv("sigma2_ajustado_e_previsto_completo.csv", show_col_types = FALSE) %>%
  pull(Returns)

returns_ins <- returns[1:1500]
sigma2_ajustada <- read.csv("df_arfima_insample_pkscru.csv")$Sigma2_ARFIMA_ajustado
sigma_ajustada <- sqrt(sigma2_ajustada)
resid_arfima_manual <- returns_ins / sigma_ajustada


summary(fitted_arfima_pkscru)
sum(is.na(fitted_arfima_pkscru))
sum(fitted_arfima_pkscru <= 0)


q_1 <- quantile(res_arfima, 0.01, na.rm = TRUE)
q_2 <- quantile(res_arfima, 0.025, na.rm = TRUE)
q_5 <- quantile(res_arfima, 0.05, na.rm = TRUE)

VaR_ARFIMA_1 <- VaR_ARFIMA_2 <- VaR_ARFIMA_5 <- numeric(n_oos)
ES_ARFIMA_1  <- ES_ARFIMA_2  <- ES_ARFIMA_5  <- numeric(n_oos)
r_oos <- numeric(n_oos)

for (i in 1:n_oos) {
  returns_window <- returns[i:(i + n_ins - 1)]
  mu <- mean(returns_window)
  
  sigma2_t <- sigma2_arfima[i]
  sigma_t <- sqrt(sigma2_t)
  
  # VaR
  VaR_ARFIMA_1[i] <- mu + sigma_t * q_1
  VaR_ARFIMA_2[i] <- mu + sigma_t * q_2
  VaR_ARFIMA_5[i] <- mu + sigma_t * q_5
  
  # ES
  ES_ARFIMA_1[i] <- mean(returns_window[returns_window < VaR_ARFIMA_1[i]])
  ES_ARFIMA_2[i] <- mean(returns_window[returns_window < VaR_ARFIMA_2[i]])
  ES_ARFIMA_5[i] <- mean(returns_window[returns_window < VaR_ARFIMA_5[i]])
  
  r_oos[i] <- returns[i + n_ins]
}

df_arfima_final <- tibble(
  Date = df$Date[(n_ins + 1):(n_ins + n_oos)],
  Return = r_oos,
  VaR_ARFIMA_1 = VaR_ARFIMA_1,
  ES_ARFIMA_1 = ES_ARFIMA_1,
  VaR_ARFIMA_2_5 = VaR_ARFIMA_2,
  ES_ARFIMA_2_5 = ES_ARFIMA_2,
  VaR_ARFIMA_5 = VaR_ARFIMA_5,
  ES_ARFIMA_5 = ES_ARFIMA_5
) %>%
  mutate(
    Exceed_VaR_ARFIMA_1 = Return < VaR_ARFIMA_1,
    Exceed_VaR_ARFIMA_2 = Return < VaR_ARFIMA_2_5,
    Exceed_VaR_ARFIMA_5 = Return < VaR_ARFIMA_5
  ) 

table(df_arfima_final$Exceed_VaR_ARFIMA_1)
table(df_arfima_final$Exceed_VaR_ARFIMA_2)
table(df_arfima_final$Exceed_VaR_ARFIMA_5)


# Scoring functions ----------------------------------------------------------------
model_names <- c("GARCH", "MSGARCH", "GAS", "LSTM", "GARCH-LSTM", "MSGARCH-LSTM", "GAS-LSTM")

df_list <- list(
  df_scores_garch,
  df_scores_msgarch,
  df_scores_gas,
  df_scores_lstm_puro,
  df_scores_garch_lstm,
  df_scores_msgarch_lstm,
  df_scores_gas_lstm
)

df_scores_all <- bind_rows(
  lapply(seq_along(df_list), function(i) {
    df_list[[i]] %>%
      mutate(Modelo = model_names[i])
  })
)

df_scores_all <- df_scores_all %>%
  select(Nível, Modelo, QL, FZ, NZ)

df_scores_all$Modelo <- factor(df_scores_all$Modelo, levels = model_names)
df_scores_all$Nível <- factor(df_scores_all$Nível, levels = c("1%", "2.5%", "5%"))
df_scores_all <- arrange(df_scores_all, Nível, Modelo)

tabela_nivel <- function(nivel) {
  df_scores_all %>%
    filter(Nível == nivel) %>%
    select(-Nível) %>%
    kbl(format = "latex", booktabs = TRUE, digits = 4,
        col.names = c("Modelo", "QL", "FZ", "NZ"),
        caption = paste("Scoring functions para nível de", nivel)) %>%
    kable_styling(latex_options = c("hold_position"), position = "center")
}

tabela_nivel("1%")
tabela_nivel("2.5%")
tabela_nivel("5%")


# GARCH  -----------------------------------------------------------
# 1%
ggplot(df_garch, aes(x = Date)) +
  geom_line(aes(y = Return), color = "black", alpha = 0.6, linewidth = 0.4) +
  geom_line(aes(y = VaR_GARCH_1), color = "red", linewidth = 0.7) +
  geom_point(data = subset(df_garch, Exceed_VaR_GARCH_1),
             aes(y = Return),
             color = "red", size = 1.5, shape = 4) +
  labs(title = "VaR GARCH - Nível de 1%",
       x = "Data", y = "Retorno") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# 2%
ggplot(df_garch, aes(x = Date)) +
  geom_line(aes(y = Return), color = "black", alpha = 0.6, linewidth = 0.4) +
  geom_line(aes(y = VaR_GARCH_2), color = "red", linewidth = 0.7) +
  geom_point(data = subset(df_garch, Exceed_VaR_GARCH_2),
             aes(y = Return),
             color = "red", size = 1.5, shape = 4) +
  labs(title = "VaR GARCH - Nível de 2.5%",
       x = "Data", y = "Retorno") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# 5%
ggplot(df_garch, aes(x = Date)) +
  geom_line(aes(y = Return), color = "black", alpha = 0.6, linewidth = 0.4) +
  geom_line(aes(y = VaR_GARCH_5), color = "red", linewidth = 0.7) +
  geom_point(data = subset(df_garch, Exceed_VaR_GARCH_5),
             aes(y = Return),
             color = "red", size = 1.5, shape = 4) +
  labs(title = "VaR GARCH - Nível de 5%",
       x = "Data", y = "Retorno") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))
# MSGARCH  -----------------------------------------------------------
# 1%
ggplot(df_msgarch, aes(x = Date)) +
  geom_line(aes(y = Return), color = "black", alpha = 0.6, linewidth = 0.4) +
  geom_line(aes(y = VaR_MSGARCH_1), color = "red", linewidth = 0.7) +
  geom_point(data = subset(df_msgarch, Exceed_VaR_MSGARCH_1),
             aes(y = Return),
             color = "red", size = 1.5, shape = 4) +
  labs(title = "VaR MSGARCH - Nível de 1%",
       x = "Data", y = "Retorno") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# 2%
ggplot(df_msgarch, aes(x = Date)) +
  geom_line(aes(y = Return), color = "black", alpha = 0.6, linewidth = 0.4) +
  geom_line(aes(y = VaR_MSGARCH_2), color = "red", linewidth = 0.7) +
  geom_point(data = subset(df_msgarch, Exceed_VaR_MSGARCH_2),
             aes(y = Return),
             color = "red", size = 1.5, shape = 4) +
  labs(title = "VaR MSGARCH - Nível de 2.5%",
       x = "Data", y = "Retorno") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# 5%
ggplot(df_msgarch, aes(x = Date)) +
  geom_line(aes(y = Return), color = "black", alpha = 0.6, linewidth = 0.4) +
  geom_line(aes(y = VaR_MSGARCH_5), color = "red", linewidth = 0.7) +
  geom_point(data = subset(df_msgarch, Exceed_VaR_MSGARCH_5),
             aes(y = Return),
             color = "red", size = 1.5, shape = 4) +
  labs(title = "VaR MSGARCH - Nível de 5%",
       x = "Data", y = "Retorno") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# GAS  -----------------------------------------------------------
# 1%
ggplot(df_gas, aes(x = Date)) +
  geom_line(aes(y = Return), color = "black", alpha = 0.6, linewidth = 0.4) +
  geom_line(aes(y = VaR_GAS_1), color = "red", linewidth = 0.7) +
  geom_point(data = subset(df_gas, Exceed_VaR_GAS_1),
             aes(y = Return),
             color = "red", size = 1.5, shape = 4) +
  labs(title = "VaR GAS - Nível de 1%",
       x = "Data", y = "Retorno") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# 2%
ggplot(df_gas, aes(x = Date)) +
  geom_line(aes(y = Return), color = "black", alpha = 0.6, linewidth = 0.4) +
  geom_line(aes(y = VaR_GAS_2), color = "red", linewidth = 0.7) +
  geom_point(data = subset(df_gas, Exceed_VaR_GAS_2),
             aes(y = Return),
             color = "red", size = 1.5, shape = 4) +
  labs(title = "VaR GAS - Nível de 2.5%",
       x = "Data", y = "Retorno") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# 5%
ggplot(df_gas, aes(x = Date)) +
  geom_line(aes(y = Return), color = "black", alpha = 0.6, linewidth = 0.4) +
  geom_line(aes(y = VaR_GAS_5), color = "red", linewidth = 0.7) +
  geom_point(data = subset(df_gas, Exceed_VaR_GAS_5),
             aes(y = Return),
             color = "red", size = 1.5, shape = 4) +
  labs(title = "VaR GAS - Nível de 5%",
       x = "Data", y = "Retorno") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# GARCH-LSTM  -----------------------------------------------------------

df_garch_lstm$Date <- as.Date(df_garch_lstm$Date)


# 1%
ggplot(df_garch_lstm, aes(x = Date)) +
  geom_line(aes(y = Return), color = "black", alpha = 0.6, linewidth = 0.4) +
  geom_line(aes(y = VaR_GARCH_LSTM_1), color = "red", linewidth = 0.7) +
  geom_point(data = subset(df_garch_lstm, Exceed_VaR_GARCH_LSTM_1),
             aes(y = Return),
             color = "red", size = 1.5, shape = 4) +
  labs(title = "VaR GARCH-LSTM - Nível de 1%",
       x = "Data", y = "Retorno") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# 2.5%
ggplot(df_garch_lstm, aes(x = Date)) +
  geom_line(aes(y = Return), color = "black", alpha = 0.6, linewidth = 0.4) +
  geom_line(aes(y = VaR_GARCH_LSTM_2), color = "red", linewidth = 0.7) +
  geom_point(data = subset(df_garch_lstm, Exceed_VaR_GARCH_LSTM_2),
             aes(y = Return),
             color = "red", size = 1.5, shape = 4) +
  labs(title = "VaR GARCH-LSTM - Nível de 2.5%",
       x = "Data", y = "Retorno") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# 5%
ggplot(df_garch_lstm, aes(x = Date)) +
  geom_line(aes(y = Return), color = "black", alpha = 0.6, linewidth = 0.4) +
  geom_line(aes(y = VaR_GARCH_LSTM_5), color = "red", linewidth = 0.7) +
  geom_point(data = subset(df_garch_lstm, Exceed_VaR_GARCH_LSTM_5),
             aes(y = Return),
             color = "red", size = 1.5, shape = 4) +
  labs(title = "VaR GARCH-LSTM - Nível de 5%",
       x = "Data", y = "Retorno") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))
# MSGARCH-LSTM  -----------------------------------------------------------

df_msgarch_lstm$Date <- as.Date(df_msgarch_lstm$Date)


# 1%
ggplot(df_msgarch_lstm, aes(x = Date)) +
  geom_line(aes(y = Return), color = "black", alpha = 0.6, linewidth = 0.4) +
  geom_line(aes(y = VaR_MSGARCH_LSTM_1), color = "red", linewidth = 0.7) +
  geom_point(data = subset(df_msgarch_lstm, Exceed_VaR_MSGARCH_LSTM_1),
             aes(y = Return),
             color = "red", size = 1.5, shape = 4) +
  labs(title = "VaR MSGARCH-LSTM - Nível de 1%",
       x = "Data", y = "Retorno") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# 2.5%
ggplot(df_msgarch_lstm, aes(x = Date)) +
  geom_line(aes(y = Return), color = "black", alpha = 0.6, linewidth = 0.4) +
  geom_line(aes(y = VaR_MSGARCH_LSTM_2), color = "red", linewidth = 0.7) +
  geom_point(data = subset(df_msgarch_lstm, Exceed_VaR_MSGARCH_LSTM_2),
             aes(y = Return),
             color = "red", size = 1.5, shape = 4) +
  labs(title = "VaR MSGARCH-LSTM - Nível de 2.5%",
       x = "Data", y = "Retorno") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# 5%
ggplot(df_msgarch_lstm, aes(x = Date)) +
  geom_line(aes(y = Return), color = "black", alpha = 0.6, linewidth = 0.4) +
  geom_line(aes(y = VaR_MSGARCH_LSTM_5), color = "red", linewidth = 0.7) +
  geom_point(data = subset(df_msgarch_lstm, Exceed_VaR_MSGARCH_LSTM_5),
             aes(y = Return),
             color = "red", size = 1.5, shape = 4) +
  labs(title = "VaR MSGARCH-LSTM - Nível de 5%",
       x = "Data", y = "Retorno") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# GAS-LSTM  -----------------------------------------------------------

df_gas_lstm$Date <- as.Date(df_gas_lstm$Date)

# 1%
ggplot(df_gas_lstm, aes(x = Date)) +
  geom_line(aes(y = Return), color = "black", alpha = 0.6, linewidth = 0.4) +
  geom_line(aes(y = VaR_GAS_LSTM_1), color = "red", linewidth = 0.7) +
  geom_point(data = subset(df_gas_lstm, Exceed_VaR_GAS_LSTM_1),
             aes(y = Return),
             color = "red", size = 1.5, shape = 4) +
  labs(title = "VaR GAS-LSTM - Nível de 1%",
       x = "Data", y = "Retorno") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# 2.5%
ggplot(df_gas_lstm, aes(x = Date)) +
  geom_line(aes(y = Return), color = "black", alpha = 0.6, linewidth = 0.4) +
  geom_line(aes(y = VaR_GAS_LSTM_2_5), color = "red", linewidth = 0.7) +
  geom_point(data = subset(df_gas_lstm, Exceed_VaR_GAS_LSTM_2),
             aes(y = Return),
             color = "red", size = 1.5, shape = 4) +
  labs(title = "VaR GAS-LSTM - Nível de 2.5%",
       x = "Data", y = "Retorno") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# 5%
ggplot(df_gas_lstm, aes(x = Date)) +
  geom_line(aes(y = Return), color = "black", alpha = 0.6, linewidth = 0.4) +
  geom_line(aes(y = VaR_GAS_LSTM_5), color = "red", linewidth = 0.7) +
  geom_point(data = subset(df_gas_lstm, Exceed_VaR_GAS_LSTM_5),
             aes(y = Return),
             color = "red", size = 1.5, shape = 4) +
  labs(title = "VaR GAS-LSTM - Nível de 5%",
       x = "Data", y = "Retorno") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))


# LSTM PURO   -----------------------------------------------------------
df_lstm_puro$Date <- as.Date(df_lstm_puro$Date)

# 1%
ggplot(df_lstm_puro, aes(x = Date)) +
  geom_line(aes(y = Return), color = "black", alpha = 0.6, linewidth = 0.4) +
  geom_line(aes(y = VaR_LSTM_PURO_1), color = "red", linewidth = 0.7) +
  geom_point(data = subset(df_lstm_puro, Exceed_VaR_LSTM_PURO_1),
             aes(y = Return),
             color = "red", size = 1.5, shape = 4) +
  labs(title = "VaR LSTM Puro - Nível de 1%",
       x = "Data", y = "Retorno") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# 2.5%
ggplot(df_lstm_puro, aes(x = Date)) +
  geom_line(aes(y = Return), color = "black", alpha = 0.6, linewidth = 0.4) +
  geom_line(aes(y = VaR_LSTM_PURO_2_5), color = "red", linewidth = 0.7) +
  geom_point(data = subset(df_lstm_puro, Exceed_VaR_LSTM_PURO_2),
             aes(y = Return),
             color = "red", size = 1.5, shape = 4) +
  labs(title = "VaR LSTM Puro - Nível de 2.5%",
       x = "Data", y = "Retorno") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# 5%
ggplot(df_lstm_puro, aes(x = Date)) +
  geom_line(aes(y = Return), color = "black", alpha = 0.6, linewidth = 0.4) +
  geom_line(aes(y = VaR_LSTM_PURO_5), color = "red", linewidth = 0.7) +
  geom_point(data = subset(df_lstm_puro, Exceed_VaR_LSTM_PURO_5),
             aes(y = Return),
             color = "red", size = 1.5, shape = 4) +
  labs(title = "VaR LSTM Puro - Nível de %",
       x = "Data", y = "Retorno") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))


# Gráfico GAS, LSTM e GAS-LSTM 1%
df_garch <- df_garch %>% mutate(Date = as.Date(Date))
df_gas <- df_gas %>% mutate(Date = as.Date(Date))
df_lstm_puro <- df_lstm_puro %>% mutate(Date = as.Date(Date))
df_gas_lstm <- df_gas_lstm %>% mutate(Date = as.Date(Date))

df_plot_1 <- df_garch %>%
  select(Date, Return) %>%
  inner_join(df_gas %>% select(Date, VaR_GAS_1), by = "Date") %>%
  inner_join(df_lstm_puro %>% select(Date, VaR_LSTM_PURO_1), by = "Date") %>%
  inner_join(df_gas_lstm %>% select(Date, VaR_GAS_LSTM_1), by = "Date")

df_plot_1_long <- df_plot_1 %>%
  pivot_longer(
    cols = starts_with("VaR"),
    names_to = "Modelo",
    values_to = "VaR"
  )

df_plot_1_long$Modelo <- recode(df_plot_1_long$Modelo,
                                "VaR_GAS_1" = "GAS",
                                "VaR_LSTM_PURO_1" = "LSTM Puro",
                                "VaR_GAS_LSTM_1" = "GAS-LSTM")

ggplot(df_plot_1_long, aes(x = Date)) +
  geom_line(aes(y = Return), color = "black", alpha = 0.6, linewidth = 0.4) +
  geom_line(aes(y = VaR, color = Modelo), linewidth = 0.7) +
  labs(title = "Comparação do VaR - Nível de 1%",
       x = "Data", y = "Retorno / VaR", color = "Modelo") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# Gráfico GAS, LSTM e GAS-LSTM 5%
df_plot_5 <- df_garch %>%
  select(Date, Return) %>%
  inner_join(df_gas %>% select(Date, VaR_GAS_5), by = "Date") %>%
  inner_join(df_lstm_puro %>% select(Date, VaR_LSTM_PURO_5), by = "Date") %>%
  inner_join(df_gas_lstm %>% select(Date, VaR_GAS_LSTM_5), by = "Date")

df_plot_5_long <- df_plot_5 %>%
  pivot_longer(
    cols = starts_with("VaR"),
    names_to = "Modelo",
    values_to = "VaR"
  )

df_plot_5_long$Modelo <- recode(df_plot_5_long$Modelo,
                                "VaR_GAS_5" = "GAS",
                                "VaR_LSTM_PURO_5" = "LSTM Puro",
                                "VaR_GAS_LSTM_5" = "GAS-LSTM")

ggplot(df_plot_5_long, aes(x = Date)) +
  geom_line(aes(y = Return), color = "black", alpha = 0.6, linewidth = 0.4) +
  geom_line(aes(y = VaR, color = Modelo), linewidth = 0.7) +
  labs(title = "Comparação do VaR - Nível de 5%",
       x = "Data", y = "Retorno / VaR", color = "Modelo") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# Gráfico alta volatilidade
limite_vol <- quantile(abs(df_plot_1$Return), 0.90)

df_long <- df_plot_1 %>%
  pivot_longer(cols = starts_with("VaR"), names_to = "Modelo", values_to = "VaR") %>%
  mutate(Modelo = recode(Modelo,
                         "VaR_GAS_1" = "GAS",
                         "VaR_LSTM_PURO_1" = "LSTM Puro",
                         "VaR_GAS_LSTM_1" = "GAS-LSTM"))

df_long <- df_long %>%
  mutate(Alta_Vol = abs(Return) > limite_vol)

plot_ly(data = df_long, x = ~Date) %>%
  add_lines(y = ~Return, name = "Retorno", line = list(color = "black", width = 1), opacity = 0.5) %>%
  add_lines(y = ~VaR, color = ~Modelo, linetype = "Modelo",
            line = list(width = 1.5), name = ~Modelo) %>%
  add_markers(data = subset(df_long, Alta_Vol),
              y = ~Return, name = "Alta volatilidade",
              marker = list(color = "orange", size = 6, symbol = "x")) %>%
  layout(title = "Comparação VaR (1%) com Destaque de Alta Volatilidade",
         xaxis = list(title = "Data"),
         yaxis = list(title = "Retorno / VaR"),
         legend = list(orientation = "h", x = 0.1, y = -0.2))

# Todas as curvas VaR para GAS, LSTM PURO e GAS-LSTM
df_plot_all <- df_garch %>%
  select(Date, Return) %>%
  inner_join(df_gas %>% select(Date, VaR_GAS_1, VaR_GAS_2, VaR_GAS_5), by = "Date") %>%
  inner_join(df_lstm_puro %>% select(Date, VaR_LSTM_PURO_1, VaR_LSTM_PURO_2_5, VaR_LSTM_PURO_5), by = "Date") %>%
  inner_join(df_gas_lstm %>% select(Date, VaR_GAS_LSTM_1, VaR_GAS_LSTM_2_5, VaR_GAS_LSTM_5), by = "Date")

df_plot_all <- df_plot_all %>%
  rename(
    `GAS_1%` = VaR_GAS_1,
    `GAS_2.5%` = VaR_GAS_2,
    `GAS_5%` = VaR_GAS_5,
    `LSTM Puro_1%` = VaR_LSTM_PURO_1,
    `LSTM Puro_2.5%` = VaR_LSTM_PURO_2_5,
    `LSTM Puro_5%` = VaR_LSTM_PURO_5,
    `GAS-LSTM_1%` = VaR_GAS_LSTM_1,
    `GAS-LSTM_2.5%` = VaR_GAS_LSTM_2_5,
    `GAS-LSTM_5%` = VaR_GAS_LSTM_5
  )

df_long <- df_plot_all %>%
  pivot_longer(
    cols = -c(Date, Return),
    names_to = c("Modelo", "Nível"),
    names_sep = "_",
    values_to = "VaR"
  )

ggplot(df_long, aes(x = Date)) +
  geom_line(aes(y = Return), color = "black", alpha = 0.5, linewidth = 0.4) +
  geom_line(aes(y = VaR, color = Modelo, linetype = Nível), linewidth = 0.8) +
  labs(title = "Curvas de VaR (1%, 2.5%, 5%) - Modelos GAS, LSTM Puro e GAS-LSTM",
       x = "Data", y = "Retorno / VaR",
       color = "Modelo", linetype = "Nível") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))


ggsave("Resultados/grafico_var_todos_modelos.pdf",
       width = 10, height = 6, units = "in")
ggsave("Resultados/grafico_var_todos_modelos.jpeg",
       width = 10, height = 6, units = "in", dpi = 300)







####


library(ggplot2)

# Define manualmente cores e linetypes
cores_modelos <- c("GAS" = "firebrick",
                   "LSTM Puro" = "steelblue",
                   "GAS-LSTM" = "darkgreen")

linetypes_niveis <- c("1%" = "solid", "2.5%" = "dashed", "5%" = "dotdash")

# Criar gráfico com refinamentos visuais
ggplot(df_long, aes(x = Date)) +
  # Linha dos retornos
  geom_line(aes(y = Return), color = "black", alpha = 0.4, linewidth = 0.4) +
  
  # VaR: separando visualmente o LSTM puro com alpha menor
  geom_line(data = df_long %>% filter(Modelo == "LSTM Puro"),
            aes(y = VaR, color = Modelo, linetype = Nível),
            linewidth = 0.9, alpha = 0.8) +
  
  # VaR: GAS e GAS-LSTM com alpha padrão
  geom_line(data = df_long %>% filter(Modelo != "LSTM Puro"),
            aes(y = VaR, color = Modelo, linetype = Nível),
            linewidth = 0.9, alpha = 0.8) +
  
  scale_color_manual(values = cores_modelos) +
  scale_linetype_manual(values = linetypes_niveis) +
  
  labs(title = "Curvas de VaR (1%, 2.5%, 5%) - Modelos GAS, LSTM Puro e GAS-LSTM",
       x = "Data", y = "Retorno / VaR",
       color = "Modelo", linetype = "Nível") +
  
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = "right")
