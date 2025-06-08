library(keras)
library(caret)
library(abind)
library(tibble)
library(readr)
library(dplyr)
library(purrr)
library(tidyr)
library(Metrics)
library(ggplot2)

## Corrigi a questão de desnormalização do antigo script 02_1; ainda não funciona N_INS = 1500.

# === PARÂMETROS GLOBAIS === #
WINDOW_SIZE <- 15 # dias considerados pela rede para prever o próximo ponto
NUM_RUNS <- 1 # uma execução para teste
MODELS <- c("garch")
TARGET <- "Returns_sq" # retornos ao quadrado
USE_FEATURES <- c("GARCH") # usando só as previsões como input
N_INS <- 1000 # com 1500 não funciona
EPOCHS <- 50
BATCH_SIZE <- 32

# === FUNÇÃO PARA CRIAR JANELAS === #
create_windows <- function(features, target, window_size) {
  X <- list()
  y <- list()
  n <- nrow(features)
  
  for (i in 1:(n - window_size)) {
    X[[i]] <- features[i:(i + window_size - 1), , drop = FALSE]
    y[[i]] <- target[i + window_size]
  }
  
  X_array <- array(unlist(X), dim = c(length(X), window_size, ncol(features)))
  y_array <- array(unlist(y), dim = c(length(y), 1))
  return(list(X = X_array, y = y_array))
}

# === DEFINIÇÃO DO MODELO ===
build_lstm_model <- function(input_shape) {
  model <- keras_model_sequential() %>%
    layer_lstm(units = 32, input_shape = input_shape) %>%
    layer_dense(units = 1, activation = "linear")
  
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = 0.001),
    loss = "mse"
  )
  return(model)
}

# === TREINAMENTO EM UMA JANELA === #
train_lstm_in_window <- function(df_window, model_type, run) {
  feature_cols <- USE_FEATURES
  
  X_raw <- df_window[, feature_cols, drop = FALSE]
  colnames(X_raw) <- feature_cols
  y_raw <- df_window[[TARGET]]
  
  split_idx <- floor(nrow(df_window) * 0.8)
  X_train_raw <- X_raw[1:split_idx, , drop = FALSE]
  X_val_raw <- X_raw[(split_idx + 1):nrow(df_window), , drop = FALSE]
  y_train_raw <- y_raw[1:split_idx]
  y_val_raw <- y_raw[(split_idx + 1):length(y_raw)]
  
  preProc_X <- preProcess(X_train_raw, method = "range")
  
  X_train_scaled <- predict(preProc_X, X_train_raw)
  X_val_scaled <- predict(preProc_X, X_val_raw)
  
  # Normalização manual do target (estava tendo problemas com a "desnormalização")
  y_min <- min(y_train_raw)
  y_max <- max(y_train_raw)
  y_train_scaled <- (y_train_raw - y_min) / (y_max - y_min)
  y_val_scaled <- (y_val_raw - y_min) / (y_max - y_min)
  
  windowed_train <- create_windows(X_train_scaled, y_train_scaled, WINDOW_SIZE)
  windowed_val <- create_windows(X_val_scaled, y_val_scaled, WINDOW_SIZE)
  
  input_shape <- c(WINDOW_SIZE, ncol(X_train_scaled))
  
  model <- build_lstm_model(input_shape)
  
  history <- model %>% fit(
    x = windowed_train$X,
    y = windowed_train$y,
    validation_data = list(windowed_val$X, windowed_val$y),
    epochs = EPOCHS,
    batch_size = BATCH_SIZE,
    verbose = 0
  )
  
  return(list(model = model, preProc_X = preProc_X, y_min = y_min, y_max = y_max))
}

# === RODAR PREVISAO COM ROLLING === #
run_rolling_forecast <- function() {
  df <- read.csv("volatilidades_previstas_completo_corrigido.csv")
  total_linhas <- nrow(df)
  cat("Total de linhas no dataset:", total_linhas, "\n")
  
  for (model_type in MODELS) {
    cat("\n=== MODELO", toupper(model_type), "===\n")
    for (run in 1:NUM_RUNS) {
      cat(" → Execução", run, "/", NUM_RUNS, "\n")
      preds <- list()
      
      valid_windows <- 0
      skipped_windows <- 0
      
      total_janelas <- total_linhas - N_INS
      cat("Rodando", total_janelas, "janelas móveis (com N_INS =", N_INS, ")\n")
      
      for (start in 1:total_janelas) {
        tryCatch({
          df_window <- df[start:(start + N_INS - 1), ]
          df_next <- df[start + N_INS, ]
          
          # checagem
          if (any(is.na(df_window[, USE_FEATURES])) || is.na(df_next[[TARGET]])) {
            skipped_windows <- skipped_windows + 1
            next
          }
          
          result <- train_lstm_in_window(df_window, model_type, run)
          model <- result$model
          preProc_X <- result$preProc_X
          y_min <- result$y_min
          y_max <- result$y_max
          
          # Previsão
          X_next <- df[(start + N_INS - WINDOW_SIZE + 1):(start + N_INS), USE_FEATURES, drop = FALSE]
          colnames(X_next) <- USE_FEATURES
          X_next_scaled <- predict(preProc_X, X_next)
          colnames(X_next_scaled) <- USE_FEATURES
          
          X_next_array <- array(as.numeric(unlist(X_next_scaled)), dim = c(1, WINDOW_SIZE, length(USE_FEATURES)))
          
          y_pred_scaled <- model %>% predict(X_next_array)
          y_pred <- as.numeric(y_pred_scaled) * (y_max - y_min) + y_min
          y_real <- df_next[[TARGET]]
          
          preds[[length(preds) + 1]] <- data.frame(
            run = run,
            window_start = start,
            predicted = y_pred,
            real = y_real
          )
          
          valid_windows <- valid_windows + 1
          
        }, error = function(e) {
          message(paste("Erro na janela", start, ":", e$message))
          skipped_windows <<- skipped_windows + 1
        })
      }
      
      cat("Janelas válidas:", valid_windows, "| Janelas puladas:", skipped_windows, "\n")
      
      if (length(preds) > 0) {
        all_preds_df <- bind_rows(preds)
        dir.create(file.path("rolling_preds", model_type), recursive = TRUE, showWarnings = FALSE)
        write.csv(all_preds_df, file.path("rolling_preds", model_type, paste0("run_", run, "_preds.csv")), row.names = FALSE)
      } else {
        cat("\nNenhuma previsão gerada.\n")
      }
    }
  }
}

run_rolling_forecast()

# === PÓS treinamento === #

# Arquivo com previsões
preds <- read.csv("rolling_preds/garch/run_1_preds.csv")

# Calcular o MSE
mse_val <- mse(preds$real, preds$predicted)
cat("MSE:", mse_val, "\n")

# Gráfico
ggplot(preds, aes(x = window_start)) +
  geom_line(aes(y = real), color = "black", linewidth = 1, alpha = 0.8) +
  geom_line(aes(y = predicted), color = "blue", linewidth = 1, alpha = 0.7) +
  labs(
    title = "GARCH-LSTM vs Proxy",
    x = "Janela",
    y = "Returns_sq",
    color = "Série"
  ) +
  theme_minimal()

head(preds$predicted, 10)
head(preds$real, 10)