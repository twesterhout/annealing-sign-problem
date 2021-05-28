import annealing_sign_problem.square_4x4
from ray import tune

# Default parameters: (10, 10000, 10000, 40960, 1e-3, 5e-5, 25, 100, 32)

def objective(beta0, beta1, sweep_sa, sign_batch, lr, weight_decay, instances, epochs, learning_batch):

    return annealing_sign_problem.square_4x4.main(beta0, beta1, sweep_sa, sign_batch, lr, weight_decay, instances, epochs, learning_batch)

def training_function(config):
    # Hyperparameters

    beta0 = 10
    beta1 = 10000
    sweep_sa = 10000
    sign_batch = 10000
    weight_decay = 5e-5
    instances = 5
    epochs = 100
    learning_batch = 32

#    lr, learning_batch = config["lr"], config["learning_batch"]
    lr = config["lr"]

    # Iterative training function - can be any arbitrary training procedure.
    intermediate_score = objective(beta0, beta1, sweep_sa, sign_batch, lr, weight_decay, instances, epochs, learning_batch)
    # Feed the score back back to Tune.
    tune.report(mean_loss=intermediate_score)

analysis = tune.run(
    training_function,
    config={
        "lr": tune.loguniform(1e-4, 1e-2),
        "learning_batch": tune.grid_search([2, 4, 8, 16, 32, 64])
    },
    num_samples=7)

print("Best config: ", analysis.get_best_config(
    metric="mean_loss", mode="min"))

# Get a dataframe for analyzing trial results.
df = analysis.results_df
