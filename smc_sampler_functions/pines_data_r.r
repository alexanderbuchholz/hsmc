# pines data extraction
library(spatstat)
data(finpines)
data_x <- (finpines$x + 5)/10 # normalize data to unit square
data_y <- (finpines$y + 8)/10
plot(x = data_x, y = data_y, type = "p", xlab = "x coordinate", ylab = "y coordinate")

df_pines = data.frame(data_x, data_y)
setwd("~/Dropbox/smc_hmc/python_smchmc/smc_sampler_functions")
write.csv(df_pines, file = "df_pines.csv")
