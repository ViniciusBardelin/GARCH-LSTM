################################################################################
######                       LSTM (tuning)                                ######
################################################################################



FLAGS <- flags(
  flag_integer("start_index", 1), 
  flag_integer("window_size", 1500),
  flag_integer("lag", 30),
  flag_integer("units", 50),
  flag_numeric("dropout", 0.2),
  flag_integer("patience", 10),
  flag_integer("batch_size", 32),
  flag_integer("epochs", 50)
)

series <- read.csv('./BTC_novos_testes/BTC_new/retornos_btc.csv')$Returns^2

end_index <- FLAGS$start_index + FLAGS$window_size - 1
window_series <- series[FLAGS$start_index:end_index]

# Função para criar janelas de defasagem
create_lagged_data <- function(series, lag = 10) {
  X <- t(sapply(1:(length(series) - lag), function(i) series[i:(i + lag - 1)]))
  y <- series[(lag + 1):length(series)]
  list(X = X, y = y)
}

data <- create_lagged_data(series, FLAGS$lag)
X <- array(data$X, dim = c(nrow(data$X), ncol(data$X), 1))
y <- data$y

# Modelo LSTM
model <- keras_model_sequential() %>%
  layer_lstm(units = FLAGS$units, input_shape = c(FLAGS$lag, 1)) %>%
  layer_dropout(rate = FLAGS$dropout) %>%
  layer_dense(units = 1)

model %>% compile(
  loss = "mse",
  optimizer = "adam"
)

# Treinamento
history <- model %>% fit(
  X, y,
  batch_size = FLAGS$batch_size,
  epochs = FLAGS$epochs,
  validation_split = 0.2,
  verbose = 0
)
