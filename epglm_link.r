# install r package james

# Download package tarball from CRAN archive

url <- "https://cran.r-project.org/src/contrib/Archive/EPGLM/EPGLM_1.1.2.tar.gz"
pkgFile <- "EPGLM_1.1.2.tar.gz"
download.file(url = url, destfile = pkgFile)

# Install dependencies

install.packages(c("RcppArmadillo", "BH"))

# Install package
install.packages(pkgs=pkgFile, type="source", repos=NULL)

# Delete package tarball
unlink(pkgFile)
library(EPGLM)

path_to_data =  c('/home/alex/Dropbox/smc_hmc/python_smchmc/smc_sampler_functions/data/simulated_dim_5', 
  '/home/alex/Dropbox/smc_hmc/python_smchmc/smc_sampler_functions/data/simulated_dim_10', 
  '/home/alex/Dropbox/smc_hmc/python_smchmc/smc_sampler_functions/data/breast_cancer', 
  '/home/alex/Dropbox/smc_hmc/python_smchmc/smc_sampler_functions/data/musk',
  '/home/alex/Dropbox/smc_hmc/python_smchmc/smc_sampler_functions/data/sonar',
  '/home/alex/Dropbox/smc_hmc/python_smchmc/smc_sampler_functions/data/german_credit', 
  '/home/alex/Dropbox/smc_hmc/python_smchmc/smc_sampler_functions/data/german_credit_interactions'
)

for (i in 1:5){
  file_name = paste(path_to_data[i], '.csv', sep='')
  data = read.csv(file_name)
  p = length(data)
  X = as.matrix(data[,1:p-1])
  y = as.matrix(data[,p])
  
  res_logit = EPlogit(X, y, 1)
  res_probit = EPprobit(X, y, 1)
  
  covar_matrix_logit = res_logit$V
  mean_matirx_logit = res_logit$m
  log_Z_logit = res_logit$Z
  
  covar_matrix_probit = res_probit$V
  mean_matirx_probit = res_probit$m
  log_Z_probit = res_probit$Z
  
  write.csv(covar_matrix_logit, file = paste(path_to_data[i], '_covar_logit.csv', sep=''))
  write.csv(mean_matirx_logit, file = paste(path_to_data[i], '_mean_logit.csv', sep=''))
  write.csv(log_Z_logit, file = paste(path_to_data[i], '_log_Z_logit.csv', sep=''))
            
  write.csv(covar_matrix_probit, file = paste(path_to_data[i], '_covar_probit.csv', sep=''))
  write.csv(mean_matirx_probit, file = paste(path_to_data[i], '_mean_probit.csv', sep=''))  
  write.csv(log_Z_probit, file = paste(path_to_data[i], '_log_Z_probit.csv', sep=''))
}
