library(data.table)
library(magrittr)
library(Hmisc)

# Read the .csv file
features <- fread(paste0(getwd(), "/df_features_train.csv"))

# set the name to the index column
colnames(features)[1] <- "index"

# # output classes of features in the .csv file
# fwrite(features[, lapply(.SD, class)], "~/Desktop/temp.csv")

# analyze the features
features[, describe(trade_roll_measure)]
features[, histogram(trade_roll_measure)]
features[trade_roll_measure < 0.002, histogram(trade_roll_measure)]



