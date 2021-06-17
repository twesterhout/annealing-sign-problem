import annealing_sign_problem.square_deep
import nevergrad as ng
from ray.tune.integration.torch import DistributedTrainableCreator
from hyperopt import tpe, hp, fmin
import ray
from ray import tune
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.nevergrad import NevergradSearch
import sys
import time

#ray.init(address='auto')

def objective(beta0, beta1, sweep_sa, sign_batch, lr, weight_decay, instances, epochs, learning_batch, features1, features2, features3, window):

    return annealing_sign_problem.square_deep.main(beta0, beta1, sweep_sa, sign_batch, lr, weight_decay, instances, epochs, learning_batch, features1, features2, features3, window)

start = time.time()
objective(6, 10000, 10000, 20000, 0.000152338, 5e-5, 40, 50, 256, 28, 28, 20, 5)
print("time ", time.time() - start)
sys.exit()

def training_function(config):
    # Hardcoded hyperparameters

    beta0 = 6
    beta1 = 10000
    sweep_sa = 10000
    sign_batch = 10000
    features1 = 28
    features2 = 28
    features3 = 20
    window = 5
    weight_decay = 5e-5
    instances = 20

    # Variable hyperparameters

    lr, epochs, learning_batch = config["lr"], config["epochs"], config["learning_batch"]

    start = time.time()
    intermediate_score = objective(beta0, beta1, sweep_sa, sign_batch, lr, weight_decay, instances, epochs, learning_batch, features1, features2, features3, window)
    print("time ", time.time() - start)
    tune.report(mean_loss=intermediate_score)
    print("time+ ", time.time() - start)

config={
        "lr": tune.loguniform(1e-4, 1e-2),
        "epochs": tune.choice([30, 50, 70, 100, 110, 130]),
        "learning_batch": tune.choice([128, 256, 512, 1024])
    }

ng_search = NevergradSearch(
    optimizer=ng.optimizers.OnePlusOne,
    metric="mean_loss",
    mode="min")

nevergradopt = tune.run(training_function, config=config, search_alg=ng_search, num_samples=200, resources_per_trial={"cpu": 20})
print("Best config: ", nevergradopt.get_best_config(
    metric="mean_loss", mode="min"))
