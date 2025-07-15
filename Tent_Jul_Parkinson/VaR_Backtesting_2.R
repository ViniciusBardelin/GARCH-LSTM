# ==============================================
# VaR & ES – GARCH vs GARCH-LSTM
# ==============================================
library(dplyr)
library(ggplot2)
library(lubridate)
library(patchwork)

# --- Parâmetros ---
initial_train <- 1500
alpha         <- 0.01
n_oos         <- 999

# --- GARCH ---
df_garch <- read.csv("vol_GARCH_1_1.csv", stringsAsFactors = FALSE) %>%
  mutate(
    Date             = ymd(Date),
    Return           = Returns,
    Residuals_garch  = Return / Sigma_GARCH
  )

# --- Médias dos retornos (janela móvel) ---
means_df <- read.csv("means_1500_garch_1_1.csv", stringsAsFactors = FALSE) %>%
  mutate(Date = ymd(Date)) %>%
  slice_tail(n = n_oos)   # médias do período OOS

# --- Previsões LSTM-GARCH OOS ---
preds_lstm <- read.csv("DF_PREDS/T70.csv", stringsAsFactors = FALSE)$Prediction

# --- Quantil GARCH (resíduos 1:1500) ---
q_alpha <- quantile(
  df_garch$Residuals_garch[1:initial_train],
  probs = alpha, na.rm = TRUE
)

# ====================================
# 1) VaR & ES – GARCH Univariado
# ====================================
df_oos_garch <- df_garch %>%
  slice((initial_train + 1):n()) %>%
  slice_head(n = n_oos) %>%
  left_join(means_df, by = "Date") %>%
  transmute(
    Date   = Date,
    Return = Return,
    Mu     = Mu_Window,
    Sigma  = Sigma_GARCH,
    VaR    = Mu + Sigma * q_alpha,
    Exceed = Return < VaR
  )

ES_garch <- df_oos_garch %>%
  filter(Exceed) %>%
  summarise(ES = mean(Return, na.rm = TRUE)) %>%
  pull(ES)

p1 <- ggplot(df_oos_garch, aes(Date)) +
  geom_line(aes(y = VaR),    color = "red",   linetype = "dashed", size = 0.8) +
  geom_line(aes(y = Return), color = "black", size = 0.6) +
  geom_point(data = filter(df_oos_garch, Exceed),
             aes(y = Return), color = "blue", size = 1.5) +
  labs(
    title    = "VaR 99% – GARCH Univariado",
    subtitle = sprintf("OOS: %d pontos | ES = %.4f", n_oos, ES_garch),
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

p1
cat("GARCH Univariado – violações:", sum(df_oos_garch$Exceed), "\n")

# ====================================
# 2) VaR & ES – GARCH-LSTM Híbrido
# ====================================
# resíduos in-sample do LSTM para recalibrar quantil
residuals_lstm_insample <- read.csv("Res/LSTM_residuals_in_sample_71.csv",
                                    stringsAsFactors = FALSE)$Residual
q_alpha_lstm <- quantile(residuals_lstm_insample,
                         probs = alpha, na.rm = TRUE)

df_oos_lstm <- df_garch %>%
  slice((initial_train + 1):n()) %>%
  slice_head(n = n_oos) %>%
  left_join(means_df, by = "Date") %>%
  mutate(
    Sigma_hat    = preds_lstm[1:n_oos],
    VaR_LSTM     = Mu_Window + Sigma_hat * q_alpha_lstm,
    Exceed_LSTM  = Return < VaR_LSTM
  ) %>%
  transmute(Date, Return, Mu = Mu_Window, Sigma_hat, VaR_LSTM, Exceed_LSTM)

ES_lstm <- df_oos_lstm %>%
  filter(Exceed_LSTM) %>%
  summarise(ES = mean(Return, na.rm = TRUE)) %>%
  pull(ES)

p2 <- ggplot(df_oos_lstm, aes(Date)) +
  geom_line(aes(y = VaR_LSTM),   color = "red",   linetype = "dashed", size = 0.8) +
  geom_line(aes(y = Return),     color = "black", size = 0.6) +
  geom_point(data = filter(df_oos_lstm, Exceed_LSTM),
             aes(y = Return), color = "blue", size = 1.5) +
  labs(
    title    = "VaR 99% – GARCH-LSTM Híbrido",
    subtitle = sprintf("OOS: %d pontos | ES = %.4f", n_oos, ES_lstm),
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

p2
cat("GARCH-LSTM Híbrido – violações:", sum(df_oos_lstm$Exceed_LSTM), "\n")

# --- Gráficos combinados ---
print(p1 / p2)
