import os

from training.runners import run_training_procedure
from models.GNNs import NRIMLPWrapper

## hyperparameters ##
model_wrapper = NRIMLPWrapper
verbose = False
model_name = "nrimlp"
batch_size = 100

## dataset path ##
source_code = os.path.join(os.getcwd(), "src")
data_files = os.path.join(source_code, "data")
output_files = os.path.join(data_files, "outputs")

## multivariate time-series process characteristics ##
dgp_simulations = ["americas", "asia_and_pacific", "europe", "mea"]
Ts = [100, 500, 1000, 2000, 3000, 4000, 5000]
functional_forms = ["linear", "nonlinear"]
error_term_dists = ["gaussian", "nongaussian"]
sampling_freq = ["daily", "monthly"]

if __name__ == "__main__":
     files = os.listdir(os.path.join(data_files, "simulations"))
     results = run_training_procedure(files=files,
                                      input_path=os.path.join(data_files, "simulations"),
                                      output_path=output_files,
                                      model_wrapper=model_wrapper,
                                      batch_size=batch_size,
                                      model_name=model_name,
                                      verbose=verbose)