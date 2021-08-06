library(data.table)
library(magrittr)
library(Hmisc)
library(ggplot2)
library(cluster)

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



# stats on features -------------------------------------------------------

# # recover stock_id and time_id
# features[, `:=`(
#   stock_id = as.integer(gsub(pattern = "(\\d+)-(\\d+)", replacement = "\\1", x = row_id)),
#   time_id = as.integer(gsub(pattern = "(\\d+)-(\\d+)", replacement = "\\2", x = row_id))
# )]
# features[, row_id := NULL]

all_stock_ids <- features[, unique(stock_id)]
all_time_ids <- features[, unique(time_id)]

# there are some time_id that are not present for all stocks
features[, .N, by=.(time_id)][N < 112, ]

# and some stocks have less than 3830 time_id's
features[, .N, by=.(stock_id)][N < 3830, ]


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
# fwrite(stats_tab, file = "~/Desktop/stats.csv")

# as we are interested in finding clusters in stock and time lets first
# grout the data into aggregated tables

stock_means <- features[, lapply(.SD, mean), by = .(stock_id)]
# stock_sds <- features[, lapply(.SD, sd), by = .(stock_id)]

time_means <- features[, lapply(.SD, mean), by = .(time_id)]
# time_sds <- features[, lapply(.SD, sd), by = .(time_id)]

# scatter plots -----------------------------------------------------------

# (STOCKS) realized volatility vs some other features 
stock_means[, plot(log_return1_std, log_return1_realized_volatility)] # sd of returns and realized volatility is highly linked
stock_means[, plot(price_spread_mean, log_return1_realized_volatility)] # same for the price spread
stock_means[, plot(spread_sum, log_return1_realized_volatility)] # same result for time-weighted quoted spread
stock_means[, plot(total_volume_mean %>% log, log_return1_realized_volatility)] # high total depth ~ lower realized volatility
stock_means[, plot(total_volume_mean %>% log, log_returnMidprice_realized_volatility)] # not the same with midprice realized volatility
stock_means[, plot(trade_amihud %>% log, trade_log_return_realized_volatility)] # higher illiquidity ~ higher volatility
stock_means[, plot(trade_roll_measure, trade_log_return_realized_volatility)] # higher autocorrelation in returns ~ higher volatility
stock_means[, plot(trade_mkt_impact %>% log, trade_log_return_realized_volatility)] # 
stock_means[, plot(trade_avg_trade_size %>% log, trade_log_return_realized_volatility)] # no statistical effect
stock_means[, plot(depth_imbalance_sum, log_return1_realized_volatility)] # higher depth imbalance ~ higher volatility
stock_means[, plot(volume_imbalance_mean %>% log, log_return1_realized_volatility)] # unweighted volume imbalance negatively correlates with realized volatility

# (STOCKS) other graphs
stock_means[, plot(price_spread_mean, total_volume_mean %>% log)] # unusual: higher spread ~ lower depth
stock_means[, plot(price_spread_mean, volume_imbalance_mean %>% log)] # unusual: higher buy/sell imbalance ~ lower spread
stock_means[, plot(spread_sum, depth_imbalance_sum)] # for the weighted measures: higher spread ~ higher depth imbalance
stock_means[, plot(total_volume_mean %>% log, volume_imbalance_mean %>% log)] # almost linear relationship btw total depth and depth imbalance => one side is always dominating the market

stock_means[, plot(price_spread_mean, trade_amihud)] # greater spread ~ more illiquidity
stock_means[, plot(bid_spread_mean, ask_spread_mean %>% abs)] # almost linear relationship => there is a minimum tick size between bid1 and bid2; ask1 and ask2
stock_means[, plot(ask_spread_mean %>% abs, price_spread_mean)] # higher spread within buy/sell side ~ higher spread on the market => different tick sizes for different stocks
stock_means[, plot(bid_spread_mean, price_spread_mean)] # higher spread within buy/sell side ~ higher spread on the market

stock_means[, plot(trade_avg_trade_size %>% log, price_spread_mean)] # interesting: higher trade size ~ lower spread
stock_means[, plot(trade_avg_trade_size %>% log, total_volume_mean %>% log)] # more depth ~ more aggressive trading
stock_means[, plot(trade_size_sum %>% log, total_volume_mean %>% log)] # more depth ~ more trading

# VERY SIMILAR OUTCOMES WHEN LOOKING AT stocks_sds TABLE




# (TIMES) realized volatility vs some other features 
time_means[, plot(log_return1_std, log_return1_realized_volatility)] # sd of returns and realized volatility is highly linked
time_means[, plot(price_spread_mean, log_return1_realized_volatility)] # same for the price spread
time_means[, plot(spread_sum, log_return1_realized_volatility)] # same result for time-weighted quoted spread
time_means[, plot(total_volume_mean %>% log, log_return1_realized_volatility)] # high total depth ~ lower realized volatility
time_means[, plot(total_volume_mean %>% log, log_returnMidprice_realized_volatility)] # SIMILAR with midprice realized volatility
time_means[, plot(trade_amihud %>% log, trade_log_return_realized_volatility)] # LESS OBVIOUS: higher illiquidity ~ higher volatility (more heterogeneous)
time_means[, plot(trade_roll_measure, trade_log_return_realized_volatility)] # LESS OBVIOUS: higher autocorrelation in returns ~ higher volatility
time_means[, plot(trade_mkt_impact %>% log, trade_log_return_realized_volatility)] # not obvious relationship
time_means[, plot(trade_avg_trade_size %>% log, trade_log_return_realized_volatility)] # 
time_means[, plot(depth_imbalance_sum, log_return1_realized_volatility)] # no visible relationship
time_means[, plot(volume_imbalance_mean %>% log, log_return1_realized_volatility)] # unweighted volume imbalance negatively correlates with realized volatility


# (TIMES) other graphs
time_means[, plot(price_spread_mean, total_volume_mean %>% log)] # unusual: higher spread ~ lower depth
time_means[, plot(price_spread_mean, volume_imbalance_mean %>% log)] # unusual: higher buy/sell imbalance ~ lower spread
time_means[, plot(spread_sum, depth_imbalance_sum)] # for the weighted measures: no distinct relationship
time_means[, plot(total_volume_mean %>% log, volume_imbalance_mean %>% log)] # almost linear relationship btw total depth and depth imbalance => one side is always dominating the market

time_means[, plot(price_spread_mean, trade_amihud)] # NOT OBVIOUS: greater spread ~ more illiquidity
time_means[, plot(bid_spread_mean, ask_spread_mean %>% abs)] # almost linear relationship
time_means[, plot(ask_spread_mean %>% abs, price_spread_mean)] # higher spread within buy/sell side ~ higher spread on the market
time_means[, plot(bid_spread_mean, price_spread_mean)] # higher spread within buy/sell side ~ higher spread on the market

time_means[, plot(trade_avg_trade_size %>% log, price_spread_mean)] # not obvious relationship
time_means[, plot(trade_avg_trade_size, total_volume_mean)] # no relationship
time_means[, plot(trade_size_sum %>% log, total_volume_mean %>% log)] # no obvious relationship

# same results for time_sds, but weaker statistically imo



# Normalization -----------------------------------------------------------

stock_means_norm <- as.data.table(scale(stock_means))
# stock_sds_norm <- as.data.table(scale(stock_sds))

time_means_norm <- as.data.table(scale(time_means))
# time_sds_norm <- as.data.table(scale(time_sds))

# drop stock_id, time_id and index
stock_means_norm[, `:=`(stock_id = NULL, index=NULL, time_id=NULL)]
time_means_norm[, `:=`(stock_id = NULL, index=NULL, time_id=NULL)]



# Euclidean distance ------------------------------------------------------

stock_means_dist <- dist(stock_means_norm)
# stock_sds_dist <- dist(stock_sds_norm)

time_means_dist <- dist(time_means_norm)


# Cluster Dendrogram with Complete Linkage --------------------------------

stock_means_hc_c <- hclust(stock_means_dist)
plot(stock_means_hc_c) # around 2 main clusters + stock #29 at height ~50

# stock_sds_hc_c <- hclust(stock_sds_dist)
# plot(stock_sds_hc_c) # around 3 main clusters + stock #29

time_means_hc_c <- hclust(time_means_dist)
plot(time_means_hc_c, labels = FALSE) # around 3 clusters

# Cluster Dendrogram with Average Linkage ---------------------------------

stock_means_hc_a <- hclust(stock_means_dist, method = "average")
plot(stock_means_hc_a) # just one main cluster with stock #29 as an outlier

# stock_sds_hc_a <- hclust(stock_sds_dist, method = "average")
# plot(stock_sds_hc_a) # around 2 main clusters + stocks #29 and #71

time_means_hc_a <- hclust(time_means_dist, method = "average")
plot(time_means_hc_a, labels = FALSE) # 

# Cluster membership ------------------------------------------------------

stock_means_memb_c <- cutree(stock_means_hc_c, k = 4)
stock_means_memb_a <- cutree(stock_means_hc_a, k = 4)

table(stock_means_memb_c, stock_means_memb_a) # identifies stocks #29 and #67 as an outliers (cluster#3#4), but has some discrepancy in cluster#2.

which(stock_means_memb_c == 3) # stock#29
which(stock_means_memb_c == 4) # stock#67



time_means_memb_c <- cutree(time_means_hc_c, k = 3)
time_means_memb_a <- cutree(time_means_hc_a, k = 3)

table(time_means_memb_c, time_means_memb_a) # 


# Cluster Means -----------------------------------------------------------

# stocks #29 and #67 are very much different
as.data.table(aggregate(stock_means_norm, list(stock_means_memb_c), mean)) %>% View

#' #29: low price, low returns, low wap_imbalance, high price_spread, high total_volume,
#'      high volume_imbalance, high trade_size, order_count, avg_trade_size
#' #67: high volatility, high returns, high spread_sum
#' 



as.data.table(aggregate(time_means_norm, list(time_means_memb_c), mean)) %>% View


# Silhouette plot ---------------------------------------------------------

silh1 <- silhouette(cutree(stock_means_hc_c, k = 4), stock_means_dist)
plot(silh1)

# silh2 <- silhouette(cutree(time_means_hc_c, k = 4), time_means_dist)
# plot(silh2)



# Scree plot --------------------------------------------------------------

# STOCKS:
wss <- (nrow(stock_means_norm)-1)*sum(apply(stock_means_norm, 2, var))
for (i in 2:10) {
  wss[i] <- sum(kmeans(stock_means_norm, centers = i)$withinss)
}
plot(1:10, wss, type='b', xlab = "Number of clusters", ylab = "Within group SS") # up to 5 clusters may be reasonable

# TIMES
wss <- (nrow(time_means_norm)-1)*sum(apply(time_means_norm, 2, var))
for (i in 2:10) {
  wss[i] <- sum(kmeans(time_means_norm, centers = i)$withinss)
}
plot(1:10, wss, type='b', xlab = "Number of clusters", ylab = "Within group SS") # up to 5 clusters may be reasonable


# K-Means Clustering ------------------------------------------------------

stock_means_kc <- kmeans(stock_means_norm, 4)
stock_means_kc

stock_means[, plot(total_volume_mean %>% log, log_return1_realized_volatility, col=stock_means_kc$cluster)]



time_means_kc <- kmeans(stock_means_norm, 3)
time_means_kc

time_means[, plot(total_volume_mean %>% log, log_return1_realized_volatility, col=time_means_kc$cluster)]
time_means[, plot(trade_roll_measure, trade_log_return_realized_volatility, col=time_means_kc$cluster)]
