library(readxl)
library(dplyr)
library(tidyr)


# OHLC Ibovespa
data <- read_excel("ohlc_ibovespa_daily.xlsx", skip = 2)

# --- BBAS3 --- #

# Selecionar colunas que contenham "BBAS3" no nome
data_bbas3 <- data %>% select(1, contains("BBAS3"))

# Retirar os NaNs
data_bbas3_clean <- data_bbas3 %>%
  mutate(across(where(is.character), ~na_if(.x, "-"))) %>%
  mutate(across(-Data, as.numeric)) %>%  # converter colunas (exceto Data) para numérico
  drop_na()

# Selecionar últimos 2500 pontos não nulos
data_bbas3_clean <- tail(data_bbas3_clean, 2500)
colnames(data_bbas3_clean) <- c("Data", "Open", "Close", "High", "Low")
library(dplyr)

data_bbas3_clean <- data_bbas3_clean %>%
  arrange(Data) %>% 
  mutate(log_return = log(Close / lag(Close)))

data_bbas3_clean <- na.omit(data_bbas3_clean)  

write.csv(data_bbas3_clean, "bbas3_returns.csv", row.names = FALSE)

# Estimativas Parkinson
func_parkinson <- function(H, L, log_price = FALSE) {
  if (log_price == FALSE) {
    P <- ((log(H / L))^2) / (4 * log(2))
  } else {
    P <- ((H - L)^2) / (4 * log(2))
  }
  return(P)
  # P = Variancia
}

bbas3_parkinson <- func_parkinson(data_bbas3_clean$High, data_bbas3_clean$Low,
                                  log_price = TRUE)
write.csv(bbas3_parkinson, "bbas3_parkinson.csv", row.names = FALSE)

# -- Obtendo as médias dos retornos -- #
df <- read.csv("bbas3_returns.csv")
df$Data <- as.Date(df$Data)
returns <- df$log_return
N <- length(returns)
window_size <- 1500

# Salva as médias para calcular o VaR depois
mu_vec <- rollapply(
  returns,
  width     = window_size,
  FUN       = mean,
  align     = "right",
  fill      = NA
)

mu_windows <- mu_vec[window_size:N]

out <- data.frame(
  Date      = df$Data[window_size:N],
  Mu_Window = mu_windows
)

write.csv(out, "bbas3_means_1500.csv", row.names = FALSE)

