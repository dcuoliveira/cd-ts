from src.training.optimization import train_and_evaluate_link_prediction

def gnn_link_prediction_objective(data, model_wrapper, criterion, verbose, trial):
         
     loss = train_and_evaluate_link_prediction(data=data,
                                               model_wrapper=model_wrapper,
                                               criterion=criterion,
                                               verbose=verbose,
                                               trial=trial)

     return loss