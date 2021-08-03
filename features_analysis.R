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

features[, describe(trade_roll_measure)]
features[, histogram(trade_roll_measure)]
features[trade_roll_measure < 0.002, histogram(trade_roll_measure)]




