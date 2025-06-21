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

# === PARÂMETROS GLOBAIS === #
WINDOW_SIZE <- 30 # tamanho da janela dentro da LSTM
NUM_RUNS <- 1
MODELS <- c("garch")
TARGET <- "Returns_sq"
USE_FEATURES <- c("Sigma_GARCH")
N_INS <- 1500
EPOCHS <- 100
BATCH_SIZE <- 32
save_every <- 100  # Salvar a cada 100 janelas

# === JANELAS PARA LSTM === #
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

# === CONFIGURAÇÃO DO MODELO === #
build_lstm_model <- function(input_shape) {
  model <- keras_model_sequential() %>%
    layer_lstm(units = 32, 
               return_sequences = TRUE, 
               input_shape = input_shape) %>%
    layer_dropout(rate = 0.2) %>%
    layer_lstm(units = 16) %>%
    layer_dropout(rate = 0.2) %>%
    layer_dense(units = 1, activation = "softplus")
  
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = 0.001),
    loss      = "mse"
  )
  model
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
    verbose = 0,
    callbacks = list(
      callback_early_stopping(
        monitor = "val_loss",
        patience = 5,
        restore_best_weights = TRUE
      )
    )
  )
  rm(history)
  gc()
  
  return(list(model = model, preProc_X = preProc_X, y_min = y_min, y_max = y_max))
}

# === ROLLING FORECAST (salvando a cada 100 previsões) === #
run_rolling_forecast <- function() {
  df <- read.csv("volatilidades_previstas_completo_corrigido_GARCH_1_1.csv") # atualizei o CSV, GARCH(1,1) COM armaOrder(0,0)!
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
      
      start_time <- Sys.time()
      
      for (start in 1:total_janelas) {
        cat("Processando janela", start, "de", total_janelas, "\r")
        flush.console()
        tryCatch({
          df_window <- df[start:(start + N_INS - 1), ]
          df_next <- df[start + N_INS, ]
          
          if (any(is.na(df_window[, USE_FEATURES])) || is.na(df_next[[TARGET]])) {
            skipped_windows <- skipped_windows + 1
            next
          }
          
          result <- train_lstm_in_window(df_window, model_type, run)
          model <- result$model
          preProc_X <- result$preProc_X
          y_min <- result$y_min
          y_max <- result$y_max
          
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
          
          if (start %% save_every == 0 || start == total_janelas) {
            cat("\nSalvando até a janela", start, "\n")
            partial_df <- bind_rows(preds)
            dir.create(file.path("rolling_preds", model_type), recursive = TRUE, showWarnings = FALSE)
            write.csv(partial_df, file.path("rolling_preds", model_type, paste0("run_", run, "_partial_up_to_", start, ".csv")), row.names = FALSE)
            rm(partial_df)
            gc()
          }
          
          rm(result, model, preProc_X, X_next_array, X_next_scaled, df_window)
          gc()
          
        }, error = function(e) {
          message(paste("Erro na janela", start, ":", e$message))
          skipped_windows <<- skipped_windows + 1
        })
      }
      
      end_time <- Sys.time()
      elapsed_time <- end_time - start_time
      print(elapsed_time)
      cat("Janelas válidas:", valid_windows, "| Janelas puladas:", skipped_windows, "\n")
    }
  }
}

run_rolling_forecast()
