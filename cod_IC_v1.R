library(quantmod)
library(rugarch)
library(MSGARCH)
library(keras)
library(tensorflow)

###########
## Dados ##
###########

# Obter dados do Bitcoin de 2015 até 2016 (poucos dados por enquanto)
get_btc_data <- function() {
  getSymbols("BTC-USD", src = "yahoo", from = "2015-01-01", to = "2016-12-31", auto.assign = FALSE)
}

btc_data <- get_btc_data()

# Log-retornos usando o preço de fechamento
log_returns <- diff(log(Cl(btc_data))) * 100 # Lu et al. 2023
log_returns <- na.omit(log_returns)

plot(log_returns, type = "l", main = "Log-retornos do Bitcoin (2015-2016)")


## Volatilidade realizada
window_size <- 30

# Vetor para armazenar a volatilidade realizada
realized_volatility <- numeric(length(log_returns) - window_size + 1)

# Loop para calcular a volatilidade realizada em uma janela deslizante
for (i in 1:(length(log_returns) - window_size + 1)) {
  # Seleciona a janela de log-retornos
  window_data <- log_returns[i:(i + window_size - 1)]
  
  # Calcula a média dos log-retornos dentro da janela
  mean_return <- mean(window_data)
  
  # Calcula a volatilidade realizada para a janela atual
  realized_volatility[i] <- sqrt(mean((window_data - mean_return)^2))
}

# Verifique os primeiros valores de realized_volatility
head(realized_volatility)

# Salvar o vetor realized_volatility em um arquivo CSV
write.csv(realized_volatility, file = "realized_volatility.csv", row.names = FALSE)

###########
## GARCH ##
###########

# Especificar o modelo GARCH(1,1)-t
spec <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
                   mean.model = list(armaOrder = c(0, 0), include.mean = FALSE),
                   distribution.model = "std")  # Distribuição t-Student

# Parâmetros para a janela deslizante
window_size <- 30 
n <- length(log_returns)
n_windows <- n - window_size

# Vetor para armazenar as previsões de volatilidade e parâmetros do modelo
vol_forecasts <- rep(NA, n_windows)
garch_params <- data.frame(omega = rep(NA, n_windows),
                           alpha1 = rep(NA, n_windows),
                           beta1 = rep(NA, n_windows))

# Loop janelas deslizante
for (i in 1:n_windows) {
  # Mensagem de progresso a cada 50 janelas
  if (i %% 50 == 0) {
    message("Processando janela ", i, " de ", n_windows)
  }
  
  # Definir a janela de dados
  window_data <- log_returns[i:(i + window_size - 1)]
  
  # Ajustar o modelo GARCH(1,1) na janela
  fit <- tryCatch(
    {
      ugarchfit(spec, data = window_data)
    },
    error = function(e) {
      message("Erro ao ajustar o modelo na janela ", i)
      return(NA)
    }
  )
  
  # Verificar se o ajuste foi bem-sucedido
  if (!is.na(fit)) {
    # Prever a volatilidade para o próximo dia
    vol_pred <- tryCatch(
      {
        pred <- ugarchforecast(fit, n.ahead = 1)
        sigma(pred)[1]  # 'sigma' retorna a previsão de volatilidade
      },
      error = function(e) {
        message("Erro na previsão de volatilidade na janela ", i)
        return(NA)
      }
    )
    
    # Armazenar a previsão de volatilidade
    vol_forecasts[i] <- vol_pred
    
    # Extrair e armazenar os parâmetros do modelo ajustado
    params <- coef(fit)
    garch_params$omega[i] <- params["omega"]
    garch_params$alpha1[i] <- params["alpha1"]
    garch_params$beta1[i] <- params["beta1"]
  }
}

# Exibir os primeiros valores dos parâmetros
head(garch_params) # deve ter dimensão (nº de janelas x nº parametros do modelo)

# Criar uma série temporal para as previsões
vol_forecasts_ts <- xts(vol_forecasts, order.by = index(log_returns)[(window_size + 1):n])

# Plotar as previsões de volatilidade
plot(vol_forecasts_ts, type = 'l', main = "Previsões de Volatilidade (GARCH(1,1)-t) com Janela Deslizante de 60 Dias")

'''
#############
## MSGARCH ##      ## não consegui implementar até agora, teoricamente fiz a mesma coisa que para o GARCH(1,1)
#############

library(MSGARCH)

# Especificar modelo MSGARCH com dois regimes e distribuição t
msgarch_spec <- CreateSpec(
  variance.spec = list(model = "sGARCH"), 
  distribution.spec = list(distribution = "std"),
  switch.spec = list(K = 2)
)

# Parâmetros para a janela deslizante
window_size <- 60
n <- length(log_returns)
n_windows <- n - window_size

# Vetores para armazenar as previsões de volatilidade e parâmetros do modelo
vol_forecasts_msgarch <- rep(NA, n_windows)
msgarch_params <- data.frame(omega1 = rep(NA, n_windows),
                             alpha1 = rep(NA, n_windows),
                             beta1 = rep(NA, n_windows),
                             omega2 = rep(NA, n_windows),
                             alpha2 = rep(NA, n_windows),
                             beta2 = rep(NA, n_windows))

# Loop através das janelas para ajustar o modelo e salvar parâmetros
for (i in 1:n_windows) {
  # Exibir mensagem de progresso a cada 50 janelas
  if (i %% 50 == 0) {
    message("Processando janela ", i, " de ", n_windows)
  }
  
  # Definir a janela de dados
  window_data <- log_returns[i:(i + window_size - 1)]
  
  # Ajustar o modelo MSGARCH na janela
  fit <- tryCatch(
    {
      FitML(msgarch_spec, window_data)
    },
    error = function(e) {
      message("Erro ao ajustar o modelo MSGARCH na janela ", i)
      return(NA)
    }
  )
  
  # Verificar se o ajuste foi bem-sucedido e extrair parâmetros
  if (inherits(fit, "MSGARCH_ML_Fit")) {
    # Prever a volatilidade para o próximo dia
    vol_forecasts_msgarch[i] <- tryCatch(
      {
        pred <- predict(fit, nahead = 1)
        pred$vol
      },
      error = function(e) {
        message("Erro na previsão de volatilidade MSGARCH na janela ", i)
        return(NA)
      }
    )
    
    # Extrair e armazenar os parâmetros do modelo ajustado
    params <- coef(fit)
    msgarch_params$omega1[i] <- params["Regime_1_Omega"]
    msgarch_params$alpha1[i] <- params["Regime_1_Alpha"]
    msgarch_params$beta1[i] <- params["Regime_1_Beta"]
    msgarch_params$omega2[i] <- params["Regime_2_Omega"]
    msgarch_params$alpha2[i] <- params["Regime_2_Alpha"]
    msgarch_params$beta2[i] <- params["Regime_2_Beta"]
  }
}

# Exibir os primeiros valores dos parâmetros para verificação
head(msgarch_params)

# Criar uma série temporal para as previsões
vol_forecasts_msgarch_ts <- xts(vol_forecasts_msgarch, order.by = index(log_returns)[(window_size + 1):n])

# Plotar as previsões de volatilidade
plot(vol_forecasts_msgarch_ts, type = 'l', main = "Previsões de Volatilidade (MSGARCH) com Janela Deslizante de 60 Dias")
'''

#################
## GARCH-LSTM ##
################

# Os parâmetros do GARCH estão em `garch_params`;
# As previsões de volatilidade estão em `vol_forecasts`;
# O volume de transação é obtido em btc_data();

# Ajustar o volume de transação para o mesmo número de linhas que `vol_forecasts`
volume_data <- tail(btc_data$`BTC-USD.Volume`, length(vol_forecasts))

# Criar o dataframe com as previsões de volatilidade + parâmetros GARCH +
# volume de transação que será inputado no LSTM

lstm_data <- data.frame(
  vol_forecast = vol_forecasts,
  omega = garch_params$omega,
  alpha1 = garch_params$alpha1,
  beta1 = garch_params$beta1,
  trans_vol = volume_data
)

# Função de normalização
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

# Não vou normalizar os parâmetros `omega`, `alpha1`, `beta1` do GARCH - de acordo
# com Medina e Moreno (2023); obs: em geral os parâmetros já estão em [0,1]

# Renomear a coluna do volume de transação
names(lstm_data)[names(lstm_data) == "BTC.USD.Volume"] <- "Volume"

# Normalizar as colunas que não são parâmetros GARCH
lstm_data$vol_forecast <- normalize(lstm_data$vol_forecast)
lstm_data$Volume <- normalize(lstm_data$Volume)

# Obter um CSV desses dados
#write.csv(lstm_data, file = "C:/Users/vinic/OneDrive/Área de Trabalho/Estudos/IC/Codes/lstm_data.csv", row.names = FALSE)

## Abaixo eu tinha tentado implementar o LSTM mas não consegui de jeito nenhum :( 

'''
# Definir o número de timesteps (janela) e número de features
n_timesteps <- 30
n_features <- ncol(lstm_data)
n_samples <- nrow(lstm_data) - n_timesteps

# Preparar os dados em formato tridimensional para o LSTM
X <- array(NA, dim = c(n_samples, n_timesteps, n_features))
y <- array(NA, dim = c(n_samples))

for (i in 1:n_samples) {
  X[i,,] <- as.matrix(lstm_data[i:(i + n_timesteps - 1), ])
  y[i] <- lstm_data$vol_forecast[i + n_timesteps]
}

# Dividir os dados em treino e teste
train_size <- round(0.8 * n_samples)
X_train <- X[1:train_size,,]
y_train <- y[1:train_size]
X_test <- X[(train_size + 1):n_samples,,]
y_test <- y[(train_size + 1):n_samples]

# Definir a estrutura da rede LSTM
model <- keras_model_sequential() %>%
  layer_lstm(units = 50, input_shape = c(n_timesteps, n_features), return_sequences = FALSE) %>%
  layer_dense(units = 1)

# Compilar o modelo
model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam()
)

# Treinar o modelo
history <- model %>% fit(
  X_train, y_train,
  epochs = 100,
  batch_size = 32,
  validation_split = 0.2,
  verbose = 1
)

# Avaliar o modelo nos dados de teste
model %>% evaluate(X_test, y_test)

# Fazer previsões
predictions <- model %>% predict(X_test)

# Exibir as previsões e valores reais
plot(y_test, type = "l", col = "blue", main = "Previsão de Volatilidade com LSTM", ylab = "Volatilidade")
lines(predictions, col = "red")
legend("topright", legend = c("Real", "Previsto"), col = c("blue", "red"), lty = 1)
'''

