# ==============================================
# VaR & ES – Modelos GARCH e GARCH-LSTM
# ==============================================

# Pacotes
library(dplyr)
library(ggplot2)
library(lubridate)
library(patchwork)
library(PerformanceAnalytics)
library(GAS)

# -----------------------------
# Parâmetros gerais
# -----------------------------
initial_train <- 1500
alphas        <- c(0.01, 0.025, 0.05)
n_oos         <- 999

# -----------------------------
# Carregamento dos dados
# -----------------------------
# Dados GARCH
df_garch <- read.csv("vol_GARCH_1_1.csv", stringsAsFactors = FALSE) %>%
  mutate(
    Date            = ymd(Date),
    Return          = Returns,
    Residuals_garch = Return / Sigma_GARCH
  )

# Médias móveis (Mu) e previsões LSTM
means_df   <- read.csv("means_1500_garch_1_1.csv", stringsAsFactors = FALSE) %>%
  mutate(Date = ymd(Date)) %>%
  slice_tail(n = n_oos)
preds_lstm <- read.csv("DF_PREDS/T101.csv", stringsAsFactors = FALSE)$Prediction
resid_lstm <- read.csv("Res/LSTM_residuals_in_sample_T101.csv", stringsAsFactors = FALSE)$Residual

# -----------------------------
# Função para gerar gráfico e ES
# -----------------------------
gerar_var_plot <- function(df_base, mu_col, sigma_col, return_col, preds = NULL,
                           label = "GARCH", alphas = c(0.01, 0.025, 0.05)) {
  plots <- list()
  results <- list()
  for (a in alphas) {
    q_a <- quantile(if (is.null(preds)) df_base$Residuals_garch[1:initial_train] else resid_lstm,
                    probs = a, na.rm = TRUE)
    
    df_oos <- df_base %>%
      slice((initial_train + 1):(initial_train + n_oos)) %>%
      left_join(means_df, by = "Date") %>%
      mutate(
        Sigma = if (is.null(preds)) .data[[sigma_col]] else preds[1:n_oos],
        VaR   = Mu_Window + Sigma * q_a,
        Exceed = Return < VaR
      ) %>%
      transmute(Date, Return, Mu = Mu_Window, Sigma, VaR, Exceed)
    
    ES <- mean(df_oos$Return[df_oos$Exceed], na.rm = TRUE)
    n_viols <- sum(df_oos$Exceed, na.rm = TRUE)
    
    p <- ggplot(df_oos, aes(Date)) +
      geom_line(aes(y = VaR), color = "red", linetype = "dashed", size = 0.8) +
      geom_line(aes(y = Return), color = "black", size = 0.6) +
      geom_point(data = filter(df_oos, Exceed), aes(y = Return), color = "blue", size = 1.5) +
      labs(
        title = sprintf("VaR %.1f%% – %s", 100 * (1 - a), label),
        subtitle = sprintf("OOS: %d pontos | ES = %.4f | Violações = %d", n_oos, ES, n_viols),
        x = "Data", y = "Retorno / VaR"
      ) +
      scale_x_date(date_labels = "%Y-%m", date_breaks = "3 months") +
      theme_minimal() +
      theme(
        plot.title    = element_text(face = "bold"),
        plot.subtitle = element_text(size = 10),
        axis.text.x   = element_text(angle = 45, hjust = 1)
      )
    
    plots[[paste0("VaR_", a)]] <- p
    results[[paste0("Backtest_", a)]] <- list(
      df = df_oos,
      ES = ES,
      Violations = n_viols
    )
  }
  
  return(list(plots = plots, results = results))
}

# -----------------------------
# Geração dos gráficos e testes
# -----------------------------
# GARCH
garch_out <- gerar_var_plot(df_garch, "Mu_Window", "Sigma_GARCH", "Return",
                            preds = NULL, label = "GARCH", alphas = alphas)

# LSTM-GARCH
lstm_out <- gerar_var_plot(df_garch, "Mu_Window", "Sigma_GARCH", "Return",
                           preds = preds_lstm, label = "GARCH-LSTM Híbrido", alphas = alphas)

# -----------------------------
# Gráficos separados por nível
# -----------------------------
for (a in alphas) {
  df_g <- garch_out$results[[paste0("Backtest_", a)]]$df
  df_l <- lstm_out$results[[paste0("Backtest_", a)]]$df
  
  p1 <- ggplot(df_g, aes(Date)) +
    geom_line(aes(y = VaR), color = "red", linetype = "dashed", size = 0.8) +
    geom_line(aes(y = Return), color = "black", size = 0.6) +
    geom_point(data = filter(df_g, Exceed), aes(y = Return), color = "blue", size = 1.5) +
    labs(
      title = sprintf("VaR %.1f%% – GARCH Univariado", 100 * (1 - a)),
      subtitle = sprintf("OOS: %d pontos | ES = %.4f | Violações = %d",
                         n_oos,
                         garch_out$results[[paste0("Backtest_", a)]]$ES,
                         garch_out$results[[paste0("Backtest_", a)]]$Violations),
      x = "Data", y = "Retorno / VaR"
    ) +
    scale_x_date(date_labels = "%Y-%m", date_breaks = "3 months") +
    theme_minimal() +
    theme(
      plot.title    = element_text(face = "bold"),
      plot.subtitle = element_text(size = 10),
      axis.text.x   = element_text(angle = 45, hjust = 1)
    )
  
  p2 <- ggplot(df_l, aes(Date)) +
    geom_line(aes(y = VaR), color = "red", linetype = "dashed", size = 0.8) +
    geom_line(aes(y = Return), color = "black", size = 0.6) +
    geom_point(data = filter(df_l, Exceed), aes(y = Return), color = "blue", size = 1.5) +
    labs(
      title = sprintf("VaR %.1f%% – GARCH-LSTM Híbrido", 100 * (1 - a)),
      subtitle = sprintf("OOS: %d pontos | ES = %.4f | Violações = %d",
                         n_oos,
                         lstm_out$results[[paste0("Backtest_", a)]]$ES,
                         lstm_out$results[[paste0("Backtest_", a)]]$Violations),
      x = "Data", y = "Retorno / VaR"
    ) +
    scale_x_date(date_labels = "%Y-%m", date_breaks = "3 months") +
    theme_minimal() +
    theme(
      plot.title    = element_text(face = "bold"),
      plot.subtitle = element_text(size = 10),
      axis.text.x   = element_text(angle = 45, hjust = 1)
    )
  
  print(p1 / p2)
}


# -----------------------------
# Backtesting para cada nível
# -----------------------------
for (a in alphas) {
  cat(sprintf("\n===== Resultados para VaR %.1f%% =====\n", 100 * (1 - a)))
  
  df_g  <- garch_out$results[[paste0("Backtest_", a)]]$df
  df_l  <- lstm_out$results[[paste0("Backtest_", a)]]$df
  VaR_g <- df_g$VaR
  VaR_l <- df_l$VaR
  ret   <- df_g$Return
  
  cat("→ GARCH:\n")
  print(BacktestVaR(ret, VaR_g, a))
  
  cat("\n→ LSTM-GARCH:\n")
  print(BacktestVaR(ret, VaR_l, a))
}
