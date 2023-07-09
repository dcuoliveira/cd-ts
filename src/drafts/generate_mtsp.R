rm(list = ls())
library("dplyr")
library("data.table")
library("reticulate")
library("vars")
library("BigVAR")
library("tsDyn")

np <- import("numpy")

source_code <- file.path(getwd(), "src")
source(file.path(source_code, "utils", "Rutils.R"))

data_files <- file.path(source_code, "data")
stocks_files <- file.path(data_files, "world_stock_indexes")
groups <- c("americas", "asia_and_pacific", "europe", "mea")

# multivariate time-series process characteristics
functional_forms <- c("linear", "nonlinear")
error_term_dists <- c("gaussian", "nongaussian")
sampling_freq <- c("daily", "monthly")
Ts <- c(100, 500, 1000, 2000, 3000, 4000, 5000)

# check if dir exists
dir.create(file.path(data_files, "simulations"), showWarnings = FALSE)
dir.create(file.path(data_files, "DGP"), showWarnings = FALSE)

for (g in groups){
  # load data
  target_file <- file.path(stocks_files, paste0(g, "_stock_indexes.npz"))
  target_data <- np$load(target_file)
  target_ts <- target_data["X_np"] %>% as.data.table()
  size <- dim(target_ts)[1]
  
  # select AR component of VAR model
  var_select_model <- VARselect(y = target_ts, lag.max = 10)
  p <- mode(var_select_model$selection)
  
  # define lasso VAR model
  var_lasso_model <- constructModel(Y = target_ts %>% as.matrix(),
                                    p = p,
                                    struct = "BasicEN",
                                    gran = c(150, 10),
                                    h = 1,
                                    cv = "Rolling",
                                    verbose = FALSE,
                                    IC = TRUE,
                                    model.controls = list(intercept = TRUE))
  
  # use cross-validation to find the penalty parameters
  cv_results <- cv.BigVAR(var_lasso_model)
  # plot(cv_results)
  # SparsityPlot.BigVAR.results(cv_results)
  optimal_lambda <- cv_results@OptimalLambda
  
  # lasso VAR model
  B <- BigVAR.fit(Y = target_ts %>% as.matrix(),
                  p = p,
                  struct = "BasicEN",
                  lambda = optimal_lambda,
                  intercept = FALSE)[, , 1]
  B <- B[, 2:(p * dim(target_ts)[2] + 1)]
  
  for (f in functional_forms){
    for (ed in error_term_dists){
      for (sf in sampling_freq){
        for (size in Ts){
          file_name <- paste(g, size, f, ed, sf, sep = "_")
          
          if (!file.exists(file.path(data_files, "simulations", paste0(file_name, ".npz")))){
            
            # functional form
            if (f == "linear"){
              var_sim <- VAR.sim(B = B,
                                 n = size,
                                 lag = p,
                                 include = "none")
            }else if (f == "nonlinear"){
              var_sim <- VAR.sim(B = B,
                                 n = size,
                                 lag = p,
                                 include = "none")
              var_sim[var_sim > 0] <- var_sim[var_sim > 0] * 0.5
              var_sim[var_sim < 0] <- var_sim[var_sim < 0] * 1.5
            }

            # error term dist
            if (ed == "gaussian"){
              var_sim <- var_sim
            }else if (ed == "nongaussian"){
              var_sim <- var_sim + runif(n = dim(var_sim)[1], min = 0, max = 1)
            }

            # sampling freq
            if (sf == "daily"){
              var_sim <- var_sim
            }else if (sf == "monthly"){
              var_sim <- var_sim[seq(from = 1, to = dim(var_sim)[1], by = 20), ]
            }
            
            np$savez(file.path(data_files, "simulations", file_name),
                     simulation=var_sim, coefficients=B)
            
          }
        }
      }
    }
  }
  
}
