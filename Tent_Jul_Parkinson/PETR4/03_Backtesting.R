# ==============================================
# VaR & ES – Modelos GARCH e GARCH-LSTM
# ==============================================

# Pacotes
library(dplyr)
library(ggplot2)
library(lubridate)
library(patchwork)
library(PerformanceAnalytics)
library(esback)
library(GAS) # VER TB função FZLoss
library(knitr)
library(kableExtra)
library(Rcpp)
source("Function_VaR_VQR.R")
sourceCpp("scoring_functions.cpp")

# -----------------------------
# Parâmetros gerais
# -----------------------------
initial_train <- 1500
alphas <- c(0.01, 0.025, 0.05)
n_oos <- 999

# -----------------------------
# Carregamento dos dados
# -----------------------------

# GARCH
df_garch <- read.csv("vol_GARCH_1_1_new.csv", stringsAsFactors = FALSE) %>%
  mutate(
    Date = ymd(Date),
    Return = Returns,
    Residuals_garch = Return / Sigma_Adjusted
  )

# MSGARCH
df_msgarch <- read.csv("vol_MSGARCH_1_1_new.csv", stringsAsFactors = FALSE) %>%
  mutate(
    Date = ymd(Date),
    Return = Returns,
    Residuals_msgarch = Return / Sigma_Adjusted
  )

# GAS
df_gas <- read.csv("vol_GAS_1_1_new.csv", stringsAsFactors = FALSE) %>%
  mutate(
    Date = ymd(Date),
    Return = Returns,
    Residuals_gas = Return / Sigma_Adjusted
  )

# Médias (Mu) e previsões LSTM
means_df   <- read.csv("means_1500_garch_1_1.csv", stringsAsFactors = FALSE) %>%
  mutate(Date = ymd(Date)) %>%
  slice_tail(n = n_oos)

# LSTM Puro – previsões e resíduos
preds_lstm_puro <- read.csv("DF_PREDS/lstm_puro_T101_new.csv", stringsAsFactors = FALSE)$Prediction
resid_lstm_puro <- read.csv("Res/LSTM_PURO_residuals_in_sample_T101_new.csv", stringsAsFactors = FALSE)$Residual

# GARCH-LSTM
preds_lstm <- read.csv("DF_PREDS/GARCH_LSTM_T101_new.csv", stringsAsFactors = FALSE)$Prediction
resid_lstm <- read.csv("Res/GARCH_LSTM_residuals_in_sample_T101_new.csv", stringsAsFactors = FALSE)$Residual

# MSGARCH-LSTM
preds_msgarch_lstm <- read.csv("DF_PREDS/MSGARCH_LSTM_T101_new.csv", stringsAsFactors = FALSE)$Prediction
resid_msgarch_lstm <- read.csv("Res/MSGARCH_LSTM_residuals_in_sample_T101_new.csv", stringsAsFactors = FALSE)$Residual

# GAS-LSTM
preds_gas_lstm <- read.csv("DF_PREDS/GAS_LSTM_T101_new.csv", stringsAsFactors = FALSE)$Prediction
resid_gas_lstm <- read.csv("Res/GAS_LSTM_residuals_in_sample_T101_new.csv", stringsAsFactors = FALSE)$Residual

# -----------------------------
# Função para gerar gráfico e ES
# -----------------------------

gerar_var_plot <- function(df_base, mu_col, sigma_col, return_col,
                           resid_col = NULL, preds = NULL,
                           label = "GARCH", alphas = c(0.01, 0.025, 0.05)) {
  plots <- list()
  results <- list()
  
  for (a in alphas) {
    # === Escolha correta do vetor de resíduos para o cálculo do quantil ===
    q_a <- if (!is.null(resid_col)) {
      # Se informado explicitamente, usa o vetor indicado de resíduos
      quantile(resid_col[1:initial_train], probs = a, na.rm = TRUE)
      #quantile(df_base[[resid_col]][1:initial_train], probs = a, na.rm = TRUE)
      
    } else if (!is.null(preds)) {
      # fallback se não tiver um vetor externo de resíduos mas há previsões (caso de LSTM)
      warning("resid_col não fornecido. Quantil pode estar sendo calculado incorretamente.")
      qnorm(a) # valor padrão apenas para evitar crash
    } else {
      # caso dos modelos econométricos puros (GARCH, MSGARCH, GAS)
      quantile(df_base[[paste0("Residuals_", tolower(label))]][1:initial_train],
               probs = a, na.rm = TRUE)
    }
    
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
                            resid_col = "Residuals_garch", label = "GARCH", alphas = alphas)

garch_out

# MSGARCH
msgarch_out <- gerar_var_plot(df_msgarch, "Mu_Window", "Sigma_MSGARCH", "Return",
                              resid_col = "Residuals_msgarch", label = "MSGARCH", alphas = alphas)
msgarch_out

# GAS
gas_out <- gerar_var_plot(df_gas, "Mu_Window", "Sigma_GAS", "Return",
                          resid_col = "Residuals_gas", label = "GAS", alphas = alphas)
gas_out

# LSTM Puro
lstm_puro_out <- gerar_var_plot(
  df_garch, "Mu_Window", "Sigma_GARCH", "Return",
  resid_col = resid_lstm_puro,
  preds = preds_lstm_puro,
  label = "LSTM",
  alphas = alphas
)

lstm_puro_out

# GARCH-LSTM
lstm_out <- gerar_var_plot(
  df_garch, "Mu_Window", "Sigma_GARCH", "Return",
  resid_col = resid_lstm,
  preds = preds_lstm,
  label = "GARCH-LSTM",
  alphas = alphas
)

lstm_out

# MSGARCH-LSTM
msgarch_lstm_out <- gerar_var_plot(
  df_msgarch, "Mu_Window", "Sigma_MSGARCH", "Return",
  resid_col = resid_msgarch_lstm,
  preds = preds_msgarch_lstm,
  label = "MSGARCH-LSTM",
  alphas = alphas
)

msgarch_lstm_out

# GAS-LSTM
gas_lstm_out <- gerar_var_plot(
  df_gas, "Mu_Window", "Sigma_GAS", "Return",
  resid_col = resid_gas_lstm,
  preds = preds_gas_lstm,
  label = "GAS-LSTM",
  alphas = alphas
)

gas_lstm_out

# -----------------------------
# Gráficos separados por nível
# -----------------------------
for (a in alphas) {
  df_g   <- garch_out$results[[paste0("Backtest_", a)]]$df
  df_l   <- lstm_out$results[[paste0("Backtest_", a)]]$df
  df_p   <- lstm_puro_out$results[[paste0("Backtest_", a)]]$df
  df_m   <- msgarch_out$results[[paste0("Backtest_", a)]]$df
  df_mh  <- msgarch_lstm_out$results[[paste0("Backtest_", a)]]$df
  df_s   <- gas_out$results[[paste0("Backtest_", a)]]$df
  df_sh  <- gas_lstm_out$results[[paste0("Backtest_", a)]]$df
  
  plots <- list(
    # GARCH vs GARCH-LSTM
    ggplot() +
      geom_line(data = df_g, aes(x = Date, y = VaR), color = "red", linetype = "dashed", size = 0.8) +
      geom_line(data = df_l, aes(x = Date, y = VaR), color = "blue", linetype = "dashed", size = 0.8) +
      geom_line(data = df_g, aes(x = Date, y = Return), color = "black", size = 0.6) +
      labs(title = sprintf("VaR %.1f%% – GARCH vs GARCH-LSTM", 100 * (1 - a)),
           subtitle = sprintf("GARCH: %d viol. | GARCH-LSTM: %d viol.",
                              sum(df_g$Exceed), sum(df_l$Exceed)),
           x = "Data", y = "Retorno / VaR") +
      theme_minimal(),
    
    # MSGARCH vs MSGARCH-LSTM
    ggplot() +
      geom_line(data = df_m, aes(x = Date, y = VaR), color = "red", linetype = "dashed", size = 0.8) +
      geom_line(data = df_mh, aes(x = Date, y = VaR), color = "blue", linetype = "dashed", size = 0.8) +
      geom_line(data = df_m, aes(x = Date, y = Return), color = "black", size = 0.6) +
      labs(title = sprintf("VaR %.1f%% – MSGARCH vs MSGARCH-LSTM", 100 * (1 - a)),
           subtitle = sprintf("MSGARCH: %d viol. | MSGARCH-LSTM: %d viol.",
                              sum(df_m$Exceed), sum(df_mh$Exceed)),
           x = "Data", y = "Retorno / VaR") +
      theme_minimal(),
    
    # GAS vs GAS-LSTM
    ggplot() +
      geom_line(data = df_s, aes(x = Date, y = VaR), color = "red", linetype = "dashed", size = 0.8) +
      geom_line(data = df_sh, aes(x = Date, y = VaR), color = "blue", linetype = "dashed", size = 0.8) +
      geom_line(data = df_s, aes(x = Date, y = Return), color = "black", size = 0.6) +
      labs(title = sprintf("VaR %.1f%% – GAS vs GAS-LSTM", 100 * (1 - a)),
           subtitle = sprintf("GAS: %d viol. | GAS-LSTM: %d viol.",
                              sum(df_s$Exceed), sum(df_sh$Exceed)),
           x = "Data", y = "Retorno / VaR") +
      theme_minimal(),
    
    # GARCH vs LSTM puro
    ggplot() +
      geom_line(data = df_g, aes(x = Date, y = VaR), color = "red", linetype = "dashed", size = 0.8) +
      geom_line(data = df_p, aes(x = Date, y = VaR), color = "blue", linetype = "dashed", size = 0.8) +
      geom_line(data = df_g, aes(x = Date, y = Return), color = "black", size = 0.6) +
      labs(title = sprintf("VaR %.1f%% – GARCH vs LSTM Puro", 100 * (1 - a)),
           subtitle = sprintf("GARCH: %d viol. | LSTM Puro: %d viol.",
                              sum(df_g$Exceed), sum(df_p$Exceed)),
           x = "Data", y = "Retorno / VaR") +
      theme_minimal()
  )
  
  print(wrap_plots(plots, ncol = 1))
}

# -----------------------------
# VQR Trucios
# -----------------------------
library(quantreg)
#VaR_VQR = function(r,VaR, alpha){
  
  fit1 = suppressWarnings(summary(rq(r ~ VaR, tau = alpha, method = "fn"), method="fn" , se="nid" , cov=TRUE))
  
  a1 = fit1$coefficients[1]
  a2 = fit1$coefficients[2]
  
  M = matrix(nrow = 2 , ncol=1)
  M[1,1] = a1
  M[2,1] = (a2-1)
  
  icov = matrix(nrow = 2 , ncol = 2)
  aa = fit1$cov[1,1]
  bb = fit1$cov[1,2]
  cc = fit1$cov[2,1]
  dd = fit1$cov[2,2]
  icov[2,1] = 1/(bb-aa*dd/cc)
  icov[2,2] = 1/(dd-cc*bb/aa)
  icov[1,1] = -icov[2,1]*dd/cc
  icov[1,2] = -icov[2,2]*bb/aa
  
  statistic = (t(M)) %*% icov %*% M 
  # Added by myself, only in case of computational problems
  if(is.na(statistic)){
    fit1 = suppressWarnings(summary(rq(r ~ VaR, tau = alpha, method = "fn"), method="fn" , se="boot" , cov=TRUE))
    
    a1 = fit1$coefficients[1]
    a2 = fit1$coefficients[2]
    
    M = matrix(nrow = 2 , ncol=1)
    M[1,1] = a1
    M[2,1] = (a2-1)
    
    icov = matrix(nrow = 2 , ncol = 2)
    aa = fit1$cov[1,1]
    bb = fit1$cov[1,2]
    cc = fit1$cov[2,1]
    dd = fit1$cov[2,2]
    icov[2,1] = 1/(bb-aa*dd/cc)
    icov[2,2] = 1/(dd-cc*bb/aa)
    icov[1,1] = -icov[2,1]*dd/cc
    icov[1,2] = -icov[2,2]*bb/aa
    
    statistic = (t(M)) %*% icov %*% M 
  }
  
  p.value = 1-pchisq(statistic[1,1], df=2)
  
  return(p.value)
}

# GARCH
r_garch <- df_garch$Returns[(initial_train + 1):nrow(df_garch)]
VaR_1_garch <- garch_out$plots$VaR_0.01$data$VaR  
VQR_1_GARCH <- VaR_VQR(r_garch, VaR_1_garch, 0.01) # 0.2361

VaR_2_5_garch <- garch_out$plots$VaR_0.025$data$VaR  
VQR_2_5_GARCH <- VaR_VQR(r_garch, VaR_2_5_garch, 0.025) # 0.000237

VaR_5_garch <- garch_out$plots$VaR_0.05$data$VaR  
VQR_5_GARCH <- VaR_VQR(r_garch, VaR_5_garch, 0.05) # 0.00162583

# MSGARCH
r_msgarch <- df_msgarch$Returns[(initial_train + 1):nrow(df_msgarch)]
VaR_1_msgarch <- msgarch_out$plots$VaR_0.01$data$VaR  
VQR_1_MSGARCH <- VaR_VQR(r_msgarch, VaR_1_msgarch, 0.01) # 0.009534

VaR_2_5_msgarch <- msgarch_out$plots$VaR_0.025$data$VaR  
VQR_2_5_MSGARCH <- VaR_VQR(r_msgarch, VaR_2_5_msgarch, 0.025) # 0

VaR_5_msgarch <- msgarch_out$plots$VaR_0.05$data$VaR  
VQR_5_MSGARCH <- VaR_VQR(r_msgarch, VaR_5_msgarch, 0.05) # 0.00085

# GAS

# -----------------------------
# Calculating ES 
# -----------------------------





# -----------------------------
# Backtesting ES 
# -----------------------------

r <- df_garch$Returns[(initial_train + 1):nrow(df_garch)]

# cc_backtest() # CoC - Conditional Calibration Backtest (Nolde & Ziegel (2007))
# er_backtest() # ER - Exceedance Residuals Backtest (McNeil & Frey (2000))
# esr_backtest() # ESR - Expected Shortfall Regression Backtest (Bayer & Dimitriadis (2020))


# -----------------------------
# Backtesting para cada nível
# -----------------------------

for (a in alphas) {
  cat(sprintf("\n===== Resultados para VaR %.1f%% =====\n", 100 * (1 - a)))
  
  df_g   <- garch_out$results[[paste0("Backtest_", a)]]$df
  df_m   <- msgarch_out$results[[paste0("Backtest_", a)]]$df
  df_s   <- gas_out$results[[paste0("Backtest_", a)]]$df
  df_p   <- lstm_puro_out$results[[paste0("Backtest_", a)]]$df
  df_gl  <- lstm_out$results[[paste0("Backtest_", a)]]$df
  df_ml  <- msgarch_lstm_out$results[[paste0("Backtest_", a)]]$df
  df_sl  <- gas_lstm_out$results[[paste0("Backtest_", a)]]$df
  
  ret <- df_g$Return  # mesmo vetor para todos
  
  cat("→ GARCH:\n")
  print(BacktestVaR(ret, df_g$VaR, a))
  
  cat("\n→ MSGARCH:\n")
  print(BacktestVaR(ret, df_m$VaR, a))
  
  cat("\n→ GAS:\n")
  print(BacktestVaR(ret, df_s$VaR, a))
  
  cat("\n→ LSTM Puro:\n")
  print(BacktestVaR(ret, df_p$VaR, a))
  
  cat("\n→ GARCH-LSTM:\n")
  print(BacktestVaR(ret, df_gl$VaR, a))
  
  cat("\n→ MSGARCH-LSTM:\n")
  print(BacktestVaR(ret, df_ml$VaR, a))
  
  cat("\n→ GAS-LSTM:\n")
  print(BacktestVaR(ret, df_sl$VaR, a))
}

# -----------------------------
# Scoring Functions
# -----------------------------
QL_deprecated = function(VaR, r, alpha){
  D = dim(VaR)
  if (is.null(D)){
    k = 1
    val_ =  (alpha - ifelse(r<=VaR,1,0))*(r-VaR)
  } else {
    k = D[2]
    n = D[1]
    val_ = matrix(0,ncol = k, nrow = n)
    for (i in 1:k){
      val_[,i] = (alpha - ifelse(r<=VaR[,i],1,0))*(r-VaR[,i])
    }
  }
  return(val_)
}

AL_deprecated = function(VaR, ES, r, alpha){
  D = dim(VaR)
  if (is.null(D)){
    k = 1
    val_ = ((-1/ES)*(ES - VaR + ifelse(r<=VaR,1,0)*(VaR - r)/alpha) - (-log(-ES)) + (1-log(1-alpha)))
  } else {
    k = D[2]
    n = D[1]
    val_ = matrix(0,ncol = k, nrow = n)
    for (i in 1:k){
      val_[,i] = ((-1/ES[,i])*(ES[,i] - VaR[,i] + ifelse(r<=VaR[,i],1,0)*(VaR[,i] - r)/alpha) - (-log(-ES[,i])) + (1-log(1-alpha)))
    }
  }
  return(val_)
}

FZ0_deprecated = function(VaR, ES, r, alpha){
  D = dim(VaR)
  if (is.null(D)){
    k = 1
    val_ = ((-1/ES)*(ES - VaR + ifelse(r<=VaR,1,0)*(VaR - r)/alpha) - (-log(-ES)))
  } else {
    k = D[2]
    n = D[1]
    val_ = matrix(0,ncol = k, nrow = n)
    for (i in 1:k){
      val_[,i] = ((-1/ES[,i])*(ES[,i] - VaR[,i] + ifelse(r<=VaR[,i],1,0)*(VaR[,i] - r)/alpha) - (-log(-ES[,i])))
    }
  }
  return(val_)
}

NZ_deprecated = function(VaR, ES, r, alpha){
  D = dim(VaR)
  if (is.null(D)){
    k = 1
    val_ = ((1/(2*sqrt(-ES)))*(ES - VaR + ifelse(r<=VaR,1,0)*(VaR - r)/alpha) + sqrt(-ES))
  } else {
    k = D[2]
    n = D[1]
    val_ = matrix(0,ncol = k, nrow = n)
    for (i in 1:k){
      val_[,i] = ((1/(2*sqrt(-ES[,i])))*(ES[,i] - VaR[,i] + ifelse(r<=VaR[,i],1,0)*(VaR[,i] - r)/alpha) + sqrt(-ES[,i]))
    }
  }
  return(val_)
}

FZG_deprecated = function(VaR, ES, r, alpha){
  D = dim(VaR)
  if (is.null(D)){
    k = 1
    val_ = (ifelse(r<= VaR,1,0) - alpha)*VaR - ifelse(r<=VaR,1,0)*r+ (exp(ES)/(1+exp(ES)))*(ES - VaR + ifelse(r<=VaR,1,0)*(VaR - r)/alpha) - log(1+exp(ES)) + log(2)
  } else {
    k = D[2]
    n = D[1]
    val_ = matrix(0,ncol = k, nrow = n)
    for (i in 1:k){
      val_[,i] = (ifelse(r<= VaR[,i],1,0) - alpha)*VaR[,i] - ifelse(r<=VaR[,i],1,0)*r+ (exp(ES[,i])/(1+exp(ES[,i])))*(ES[,i] - VaR[,i] + ifelse(r<=VaR[,i],1,0)*(VaR[,i] - r)/alpha) - log(1+exp(ES[,i])) + log(2)
    }
  }
  return(val_)
}

# -----------------------------
# Tabela para cada nível
# -----------------------------
library(knitr)
library(kableExtra)
library(dplyr)

# Define os modelos e a ordem desejada
modelos <- c("GARCH", "MSGARCH", "GAS", "LSTM", "GARCH-LSTM", "MSGARCH-LSTM", "GAS-LSTM")
resultados <- list(
  "GARCH"         = garch_out,
  "MSGARCH"       = msgarch_out,
  "GAS"           = gas_out,
  "LSTM"          = lstm_puro_out,
  "GARCH-LSTM"    = lstm_out,
  "MSGARCH-LSTM"  = msgarch_lstm_out,
  "GAS-LSTM"      = gas_lstm_out
)

# Inicializa a tabela
tabela_final <- data.frame()

# Loop pelos modelos e níveis
for (modelo in modelos) {
  for (a in alphas) {
    nivel_char <- paste0(100 * a, "%")
    
    df_ret <- resultados[[modelo]]$results[[paste0("Backtest_", a)]]$df
    VaR    <- df_ret$VaR
    ret    <- df_ret$Return
    bt     <- BacktestVaR(ret, VaR, a)
    
    viol_pct <- mean(ret < VaR) * 100
    
    linha <- data.frame(
      Modelo    = modelo,
      Nivel     = nivel_char,
      Violacoes = sprintf("%.2f%%", viol_pct),
      UC        = sprintf("%.3f", bt$LRuc[2]),
      CC        = sprintf("%.3f", bt$LRcc[2]),
      DQ        = sprintf("%.3f", bt$DQ$pvalue[1])
    )
    
    tabela_final <- bind_rows(tabela_final, linha)
  }
}

# Gera a tabela com multirow
kable(tabela_final, "latex", booktabs = TRUE, linesep = "", align = "lcccc",
      caption = "Resultados dos testes de backtesting para os modelos avaliados.",
      label = "tab:backtest") %>%
  kable_styling(latex_options = c("hold_position", "repeat_header")) %>%
  pack_rows(index = table(tabela_final$Modelo)) %>%
  column_spec(1, bold = TRUE)

