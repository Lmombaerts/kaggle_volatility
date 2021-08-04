library(data.table)
library(magrittr)
library(Hmisc)


# File load ---------------------------------------------------------------


# # Read the .csv file
# features <- fread(paste0(getwd(), "/df_features_train.csv")) # 20.5 sec loading time
# 
# # set the name to the index column
# colnames(features)[1] <- "index"
# 
# # output classes of features in the .csv file
# fwrite(features[, lapply(.SD, class)], "~/Desktop/temp.csv")
# 
# saveRDS(features, file=paste0(getwd(), "/features.RDS"))

# or read .RDS file
features <- readRDS(paste0(getwd(), "/features.RDS")) # 5.6 sec loading time


# feature analysis --------------------------------------------------------

# recover stock_id and time_id
features[, `:=`(
  stock_id = as.integer(gsub(pattern = "(\\d+)-(\\d+)", replacement = "\\1", x = row_id)),
  time_id = as.integer(gsub(pattern = "(\\d+)-(\\d+)", replacement = "\\2", x = row_id))
)]
features[, row_id := NULL]

all_stock_ids <- features[, unique(stock_id)]
all_time_ids <- features[, unique(time_id)]

# there are some time_id that are not present for all stocks
features[, .N, by=.(time_id)][N < 112, ]

# and some stocks have less than 3830 time_id's
features[, .N, by = .(stock_id)][, describe(N)]


# descriptive stats for each column across all stocks and time id's
row_mean <- features[, lapply(.SD, mean)]
row_std <- features[, lapply(.SD, sd)]
row_p05 <- features[, lapply(.SD, function(x) quantile(x, prob=0.05))]
row_p25 <- features[, lapply(.SD, function(x) quantile(x, prob=0.25))]
row_p50 <- features[, lapply(.SD, function(x) quantile(x, prob=0.50))]
row_p75 <- features[, lapply(.SD, function(x) quantile(x, prob=0.75))]
row_p95 <- features[, lapply(.SD, function(x) quantile(x, prob=0.95))]
row_min <- features[, lapply(.SD, min)]
row_max <- features[, lapply(.SD, max)]

stats_tab <- cbind(data.table(stats = c("mean", "std", "min", "p05", "p25", "p50", "p75", "p95", "max")), rbind(row_mean, row_std, row_min, row_p05, row_p25, row_p50, row_p75, row_p95, row_max))

stats_tab_t <- transpose(stats_tab)
colnames(stats_tab_t) <- c("mean", "std", "min", "p05", "p25", "p50", "p75", "p95", "max")
stats_tab_t <- cbind(names(stats_tab), stats_tab_t)
colnames(stats_tab_t)[1] <- "feature"
stats_tab <- stats_tab_t[c(-1,-2, -279, -280),]
rm(stats_tab_t, row_max, row_mean, row_min, row_p05, row_p25, row_p50, row_p75, row_p95, row_std)

stats_tab

features[, describe(trade_roll_measure)]
features[, histogram(trade_roll_measure)]
features[trade_roll_measure < 0.002, histogram(trade_roll_measure)]




