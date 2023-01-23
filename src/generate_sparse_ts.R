rm(list = ls())
library("dplyr")
library("data.table")
library("reticulate")
library("vars")
library("BigVAR")
library("tsDyn")

np <- import("numpy")

source_code <- file.path(getwd(), "src")
source(file.path(source_code, "Rutils.R"))

data_files <- file.path(source_code, "data")
stocks_files <- file.path(data_files, "world_stock_indexes")
groups <- c("americas", "asia_and_pacific", "europe", "mea")

for (g in groups){
    # load data
    target_file <- file.path(stocks_files, paste0(g, "_stock_indexes.npz"))
    target_data <- np$load(target_file)
    target_ts <- target_data["X_np"] %>% as.data.table()

    # select AR component of VAR model
    var_select_model <- VARselect(y = target_ts, lag.max = 10)
    p <- mode(var_select_model$selection)

    # define lasso VAR model
    var_lasso_model <- constructModel(Y = target_ts %>% as.matrix(),
                                    p = p,
                                    struct = "Basic",
                                    gran = c(150, 10),
                                    h = 1,
                                    cv = "Rolling",
                                    verbose = FALSE,
                                    IC = TRUE,
                                    model.controls = list(intercept = TRUE))

    # use cross-validation to find the penalty parameters
    cv_results <- cv.BigVAR(var_lasso_model)
    # plot(cv_results) # nolint
    # SparsityPlot.BigVAR.results(cv_results) # nolint
    optimal_lambda <- cv_results@OptimalLambda

    # lasso VAR model
    B <- BigVAR.fit(Y = target_ts %>% as.matrix(),
                    p = p,
                    struct = "Basic",
                    lambda = optimal_lambda,
                    intercept = FALSE)[, , 1]
    B <- B[, 2:(p * dim(target_ts)[2] + 1)]

    # check if dir exists
    dir.create(file.path(data_files, "simulation"), showWarnings = FALSE)

    # simulate from fitted VAR model
    var_sim <- VAR.sim(B = B, n = 2000, lag = p, include = "none")
    file_name <- paste0(g, "_var_simulation.csv")
    write.csv2(x = var_sim,
            file = file.path(data_files, "simulation", file_name))

    # 1. add noise to VAR simulation

    ## a) gaussian noise
    var_sim_gaussian_noise <- var_sim + rnorm(n = dim(var_sim)[1], mean = 0, sd = 1) # nolint
    file_name <- paste0(g, "_var_simulation_gaussian.csv")
    write.csv2(x = var_sim_gaussian_noise,
            file = file.path(data_files, "simulation", file_name))

    ## b) non-gaussian noise
    var_sim_uniform_noise <- var_sim + runif(n = dim(var_sim)[1], min = 0, max = 1) # nolint
    file_name <- paste0(g, "_var_simulation_nongaussian.csv")
    write.csv2(x = var_sim_uniform_noise,
            file = file.path(data_files, "simulation", file_name))

    # 2. resampled series

    ## a) weekly
    var_sim_weekly <- var_sim[seq(from = 1, to = dim(var_sim)[1], by = 5), ]
    file_name <- paste0(g, "_var_simulation_weekly.csv")
    write.csv2(x = var_sim_weekly,
            file = file.path(data_files, "simulation", file_name))

    ## b) monthly
    var_sim_monthly <- var_sim[seq(from = 1, to = dim(var_sim)[1], by = 20), ]
    file_name <- paste0(g, "_var_simulation_monthly.csv")
    write.csv2(x = var_sim_monthly,
            file = file.path(data_files, "simulation", file_name))

    # 3. non-linear (in parameter) relationships - TVAR model
    # setar_sim <- TVAR.sim(B = B,
    #                       n = 2000,
    #                       lag = p,
    #                       include = "none",
    #                       nthresh = 1,
    #                       Thresh = 2)
    # file_name <- paste0(g, "_setar_simulation.csv") # nolint
    # write.csv2(x = var_sim,
    #           file = file.path(data_files, "simulation", file_name))

    # 4. instantaneous effects
}
