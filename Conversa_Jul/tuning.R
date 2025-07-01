library(keras)
library(kerastuneR)
library(tensorflow)
library(caret)
library(tidyverse)


df <- read.csv("volatilidades_previstas_completo_corrigido_GARCH_1_1.csv")

N_INS <- 500
WINDOW_SIZE <- 15
TARGET <- "Returns_sq"
USE_FEATURES <- c("Sigma_GARCH")

plot(df$Sigma_GARCH, type = "l")

plot(df$Returns_sq, type = "l")
plot(df$Returns)

# Fazendo a primeira janela de treinamento para o tuning
df_window <- df[1:N_INS, ]
X_raw <- df_window[, USE_FEATURES]
y_raw <- df_window[[TARGET]]

# Split train/val 
split_idx <- floor(nrow(df_window) * 0.8)
X_train_raw <- X_raw[1:split_idx, , drop = FALSE]
X_val_raw   <- X_raw[(split_idx + 1):nrow(df_window), , drop = FALSE]
y_train_raw <- y_raw[1:split_idx]
y_val_raw   <- y_raw[(split_idx + 1):length(y_raw)]

# Normalização
preProc_X <- preProcess(X_train_raw, method = "range")
X_train_scaled <- predict(preProc_X, X_train_raw)
X_val_scaled   <- predict(preProc_X, X_val_raw)

y_min <- min(y_train_raw)
y_max <- max(y_train_raw)
y_train_scaled <- (y_train_raw - y_min) / (y_max - y_min)
y_val_scaled   <- (y_val_raw - y_min) / (y_max - y_min)

# Criar janelas dentro do LSTM
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

windowed_train <- create_windows(X_train_scaled, y_train_scaled, WINDOW_SIZE)
windowed_val   <- create_windows(X_val_scaled, y_val_scaled, WINDOW_SIZE)

X_train_array <- windowed_train$X
y_train_array <- windowed_train$y
X_val_array   <- windowed_val$X
y_val_array   <- windowed_val$y

# === Modelo parametrizável
build_model <- function(hp) {
  
  units1 <- as.integer(hp$Choice('units1', values = c(32L, 64L, 128L, 256L)))
  units2 <- as.integer(hp$Choice('units2', values = c(32L, 64L, 128L, 256L)))
  batch_size <- as.integer(hp$Choice('batch_size', values = c(32, 64)))
  epochs <- as.integer(hp$Choice('epochs', values = c(50, 100)))
  
  dropout1 <- hp$Choice('dropout1', values = c(0.05, 0.1))
  dropout2 <- hp$Choice('dropout2', values = c(0.05, 0.1))
  learning_rate <- hp$Choice('learning_rate', values = c(0.0005, 0.001))
  activation <- hp$Choice('activation', values = c('relu', 'softplus'))
  
  batch_size <- hp$Choice("batch_size", values = c(32L, 64L)) 
  epochs <- hp$Choice("epochs", values = c(50L, 100L))
  
  model <- keras_model_sequential() %>%
    layer_lstm(units = units1, return_sequences = TRUE, activation = "tanh", input_shape = c(WINDOW_SIZE, dim(X_train_array)[3])
) %>%
    layer_dropout(rate = dropout1) %>%
    layer_lstm(units = units2, activation = "tanh") %>%
    layer_dropout(rate = dropout2) %>%
    layer_dense(units = 1, activation = activation)
  
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = learning_rate),
    loss = "mse"
  )
  
  return(model)
}

# Early stopping 
early_stop <- callback_early_stopping(
  monitor = "val_loss",
  patience = 5,
  restore_best_weights = TRUE
)

# Define o tuning
tuner <- RandomSearch(
  build_model,
  objective = "val_loss",
  max_trials = 30,          # número de configurações diferentes a testar
  executions_per_trial = 3, # quantas repetições por configuração
  directory = "lstm_tuning",
  project_name = "btc_lstm"
)

# Roda o tuning
tuner$search(
  x = X_train_array,
  y = y_train_array,
  epochs = as.integer(50), # isso é ignorado
  batch_size = as.integer(32), # isso é ignorado
  validation_data = list(X_val_array, y_val_array),
  callbacks = list(early_stop)
)


# Melhores configurações
best_hyperparams <- tuner$get_best_hyperparameters(num_trials = as.integer(1))[[1]]
print(best_hyperparams)

best_hyperparams$get_config()


