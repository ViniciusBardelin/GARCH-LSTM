library(dplyr)
library(ggplot2)
library(lubridate)

# Dados e parâmetros
df <- read.csv("volatilidades_previstas_completo_corrigido_GARCH_1_1.csv")
t6_lstm <- read.csv("T6.csv")
initial_train <- 1500
alpha <- 0.01
n_oos <- 999

# Resíduos padronizados e quantil
df <- df %>%
  mutate(residuals = Returns / Sigma_GARCH)
q_alpha <- quantile(df$residuals[1:initial_train], probs = alpha, na.rm = TRUE)

# --- VaR GARCH Univariado ---
df_oos_garch <- df %>%
  slice((initial_train + 1):n()) %>%
  slice_head(n = n_oos) %>%
  transmute(
    Date   = ymd(Date),
    Return = Returns,
    VaR    = Sigma_GARCH * q_alpha
  ) %>%
  mutate(Exceed = Return < VaR)

ggplot(df_oos_garch, aes(Date)) +
  geom_line(aes(y = VaR),   color = "red",   linetype = "dashed", size = 0.8) +
  geom_line(aes(y = Return), color = "black", size = 0.6) +
  geom_point(data = filter(df_oos_garch, Exceed),
             aes(y = Return),
             color = "blue", size = 1.5) +
  labs(
    title    = "VaR 99% – GARCH Univariado",
    subtitle = sprintf("Período OOS: %d pontos", n_oos),
    x        = "Data",
    y        = "Retorno / VaR"
  ) +
  scale_x_date(date_labels = "%Y-%m", date_breaks = "3 months") +
  theme_minimal() +
  theme(
    plot.title    = element_text(face = "bold"),
    plot.subtitle = element_text(size = 10),
    axis.text.x   = element_text(angle = 45, hjust = 1)
  )

print(table(df_oos_garch$Exceed))

# --- VaR Híbrido (GARCH-LSTM) ---
t6_lstm <- t6_lstm %>%
  slice_head(n = n_oos) %>%
  mutate(VaR = Prediction * q_alpha)

df_oos_lstm <- df %>%
  slice((initial_train + 1):n()) %>%
  slice_head(n = n_oos) %>%
  transmute(
    Date   = ymd(Date),
    Return = Returns,
    VaR    = t6_lstm$VaR
  ) %>%
  mutate(Exceed = Return < VaR)

ggplot(df_oos_lstm, aes(Date)) +
  geom_line(aes(y = VaR),   color = "red",   linetype = "dashed", size = 0.8) +
  geom_line(aes(y = Return), color = "black", size = 0.6) +
  geom_point(data = filter(df_oos_lstm, Exceed),
             aes(y = Return),
             color = "blue", size = 1.5) +
  labs(
    title    = "VaR 99% – GARCH-LSTM",
    subtitle = sprintf("Período OOS: %d pontos", n_oos),
    x        = "Data",
    y        = "Retorno / VaR"
  ) +
  scale_x_date(date_labels = "%Y-%m", date_breaks = "3 months") +
  theme_minimal() +
  theme(
    plot.title    = element_text(face = "bold"),
    plot.subtitle = element_text(size = 10),
    axis.text.x   = element_text(angle = 45, hjust = 1)
  )

print(table(df_oos_lstm$Exceed))