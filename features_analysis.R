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

# there are some time_id that are not present for all stocks
features[, .N, by=.(time_id)][N < 112, ]

# and some stocks have less than 3830 time_id's
features[, .N, by = .(stock_id)][, describe(N)]


features[, describe(trade_roll_measure)]
features[, histogram(trade_roll_measure)]
features[trade_roll_measure < 0.002, histogram(trade_roll_measure)]




