# ==============================================
# VaR & ES – Modelos GARCH e GARCH-LSTM
# ==============================================

library(dplyr)
library(readr)
library(ggplot2)
library(lubridate)
library(patchwork)
library(PerformanceAnalytics)
library(esback)
library(GAS) # VER TB função FZLoss
library(knitr)
library(kableExtra)
source("Function_VaR_VQR.R")

# Parâmetros gerais
initial_train <- 1500
alphas <- c(0.01, 0.025, 0.05)
n_oos <- 2000

# GARCH
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

# VaR Backtest: UC, CC, DQ
Back_VaR_GARCH_1 <- BacktestVaR(df_garch$Return, df_garch$VaR_GARCH_1, 0.01)
Back_VaR_GARCH_2 <- BacktestVaR(df_garch$Return, df_garch$VaR_GARCH_2, 0.025)
Back_VaR_GARCH_5 <- BacktestVaR(df_garch$Return, df_garch$VaR_GARCH_5, 0.05)

# ES Backtest: CoC, ER, ESR
Back_ES_CoC_GARCH_1 <- cc_backtest(df_garch$Return, df_garch$VaR_GARCH_1, df_garch$ES_GARCH_1, df_garch$Vol2_GARCH, 0.01)
Back_ES_CoC_GARCH_2 <- cc_backtest(df_garch$Return, df_garch$VaR_GARCH_2, df_garch$ES_GARCH_2, df_garch$Vol2_GARCH, 0.025)
Back_ES_CoC_GARCH_5 <- cc_backtest(df_garch$Return, df_garch$VaR_GARCH_5, df_garch$ES_GARCH_5, df_garch$Vol2_GARCH, 0.05)

Back_ES_ER_GARCH_1 <- er_backtest(df_garch$Return, df_garch$VaR_GARCH_1, df_garch$ES_GARCH_1, df_garch$Vol2_GARCH)
Back_ES_ER_GARCH_2 <- er_backtest(df_garch$Return, df_garch$VaR_GARCH_2, df_garch$ES_GARCH_2, df_garch$Vol2_GARCH)
Back_ES_ER_GARCH_5 <- er_backtest(df_garch$Return, df_garch$VaR_GARCH_5, df_garch$ES_GARCH_5, df_garch$Vol2_GARCH)

Back_ES_ESR_GARCH_1_V1 <- esr_backtest(df_garch$Return, df_garch$VaR_GARCH_1, df_garch$ES_GARCH_1, alpha=0.01, version=1, B=0)
Back_ES_ESR_GARCH_2_V1 <- esr_backtest(df_garch$Return, df_garch$VaR_GARCH_2, df_garch$ES_GARCH_2, alpha=0.025, version=1, B=0)
Back_ES_ESR_GARCH_5_V1 <- esr_backtest(df_garch$Return, df_garch$VaR_GARCH_5, df_garch$ES_GARCH_5, alpha=0.05, version=1, B=0)

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

# GARCH-LSTM
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

# 2.5%
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


# GARCH-LSTM
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


# GRÁFICOS GARCH

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

# 2.5%
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


# MSGARCH-LSTM
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

# GAS-LSTM
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
