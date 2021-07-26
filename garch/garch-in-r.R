library(rugarch)
library(data.table)
library(magrittr)
library(rmgarch)
library(forecast)

df <- fread("/Users/vladimir.levin/Yandex.Disk.localized/01_Documents/01_Study/0_LUX/01_PhD/Drafts/Project3/kaggle_volatility/test-data.csv")

df[time_id == 5, plot(second, wap, type='l')]


df_5 <- copy(df[time_id == 16774])


# ACF for absolute returns
tsdisplay(df_5$log_return %>% abs)

df_5[, sum(log_return)]

# fit GARCH(1,1)
spec = ugarchspec(variance.model = list(model="sGARCH",garchOrder = c(1, 1))) # empty function  = default model
def.fit = ugarchfit(spec = spec, data = df_5$log_return)
# print(def.fit)

# predict
# n.ahead = 300
bootp <- ugarchboot(def.fit, method=c("Partial","Full")[1], n.ahead = 600, n.bootpred=1,n.bootfit=1)
garch_pred <- c(bootp@fseries)

plot(garch_pred, type='h')
plot(c(abs(df_5$log_return), abs(garch_pred)), type = 'l')

# realized volatility of the predicted returns
sum(garch_pred^2) %>% sqrt




# df[, .(rv = sqrt(sum(log_return^2))), by = .(time_id)][, plot(time_id, rv)]
df[, .(rv = sqrt(sum(log_return^2))), by = .(time_id)][rv > 0.02]




