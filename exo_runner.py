import annealing_sign_problem.square_4x4
import nevergrad as ng
from hyperopt import tpe, hp, fmin
import ray
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest.nevergrad import NevergradSearch
import sys

def objective(beta0, beta1, sweep_sa, sign_batch, lr, weight_decay, instances, epochs, learning_batch, features1, features2, window):

    return annealing_sign_problem.square_4x4.main(beta0, beta1, sweep_sa, sign_batch, lr, weight_decay, instances, epochs, learning_batch, features1, features2, window)

#objective(10, 10000, 10000, 5000, 1e-3, 5e-5, 10, 100, 1024, 16, 32, 5)

def toy_function(config):

    lr = config["lr"]
    wd = config["wd"]
    score = (lr-2)**2 + (wd+3.28)**4
    tune.report(mean_loss=score)

def training_function(config):
    # Hyperparameters

    beta0 = 10
    beta1 = 10000
    sweep_sa = 10000
    sign_batch = 5000
    lr = 1e-3
    weight_decay = 5e-5
    instances = 10
    epochs = 100
    learning_batch = 32

#    lr, weight_decay = config["lr"], config["weight_decay"]
#    lr = config["lr"]

    features1, features2, window = config["features1"], config["features2"], config["window"]

    intermediate_score = objective(beta0, beta1, sweep_sa, sign_batch, lr, weight_decay, instances, epochs, learning_batch, features1, features2, window)
    tune.report(mean_loss=intermediate_score)

sizes = [i for i in range(8,65,4)]

config={
        "features1": tune.grid_search(sizes),
        "features2": tune.grid_search(sizes),
        "window": tune.grid_search([3, 5])
#        "learning_batch": tune.choice([2, 4, 8, 16, 32, 64])
    }

#config={
#        "lr": tune.uniform(-4, 4),
#        "wd": tune.choice([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
#    }

#current_best_params = [{
#    "lr": 0,
#    "wd": 0
#}]

ng_search = NevergradSearch(
    optimizer=ng.optimizers.OnePlusOne,
    metric="mean_loss",
    mode="min")

nevergradopt = tune.run(training_function, config=config, search_alg=ng_search, num_samples=100, resources_per_trial={"cpu": 20})
print("Best config: ", nevergradopt.get_best_config(
    metric="mean_loss", mode="min"))

sys.exit()

bayesopt = BayesOptSearch(metric="mean_loss", mode="min")
analysis = tune.run(training_function, config=config, search_alg=bayesopt, num_samples=1, resources_per_trial={"cpu": 20})
#analysis = tune.run(toy_function, config=config, search_alg=bayesopt, num_samples=300)

print("Best config: ", analysis.get_best_config(
    metric="mean_loss", mode="min"))

hyperopt_search = HyperOptSearch(space, metric="mean_loss", mode="min")#, points_to_evaluate=current_best_params)

analysis = tune.run(
    toy_function,
    #training_function,
    #config=config,    
    search_alg=hyperopt_search,
    stop={"training_iteration": 200})

#analysis = tune.run(
#    training_function,
#    config={
#        "lr": tune.loguniform(1e-4, 1e-2)#,
    #    "learning_batch": tune.grid_search([2, 4, 8, 16, 32, 64])
#    },
#    num_samples=40)

print("Best config: ", analysis.get_best_config(
    metric="mean_loss", mode="min"))
