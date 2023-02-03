rm(list = ls())
library("tsDyn")
library("reticulate")
library("dplyr")
library("data.table")

np <- import("numpy")

source_code <- file.path(getwd(), "src")
data_files <- file.path(source_code, "data")
stocks_files <- file.path(data_files, "world_stock_indexes")
g <- "americas"

target_file <- file.path(stocks_files, paste0(g, "_stock_indexes.npz"))
target_data <- np$load(target_file)
target_ts <- target_data["X_np"] %>% as.data.table()

# estimate TVAR
tvar_fit <- TVAR(target_ts, lag = 7, dummyToBothRegimes = TRUE)
B <- cbind(tvar_fit$coefficients[["Bdown"]], tvar_fit$coefficients[["Bup"]])

sim <- TVAR.sim(B = B,
                n = 500,
                lag = 7,
                mTh = 1,
                nthresh = 1,
                Thresh = tvar_fit$model.specific$Thresh)
