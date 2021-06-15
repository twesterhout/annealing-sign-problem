import annealing_sign_problem.square_deep4
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

def objective(beta0, beta1, sweep_sa, sign_batch, lr, weight_decay, instances, epochs, learning_batch, features1, features2, features3, features4, window):

    return annealing_sign_problem.square_deep4.main(beta0, beta1, sweep_sa, sign_batch, lr, weight_decay, instances, epochs, learning_batch, features1, features2, features3, features4, window)

#start = time.time()
#objective(10, 10000, 10000, 5000, 1e-3, 5e-5, 20, 50, 1024, 28, 28, 20, 5)
#print("time ", time.time() - start)
#sys.exit()

def toy_function(config):

    lr = config["lr"]
    wd = config["wd"]
    score = (lr-2)**2 + (wd+3.28)**4
    tune.report(mean_loss=score)

def training_function(config):
    # Hardcoded hyperparameters

    beta0 = 10
    beta1 = 20000
    sweep_sa = 10000
    sign_batch = 5000
    lr = 1e-3
    weight_decay = 5e-5
    instances = 20
    epochs = 100
    learning_batch = 1024

    # Variable hyperparameters

    features1, features2, features3, features4, window = config["features1"], config["features2"], config["features3"], config["features4"], config["window"]

    start = time.time()
    intermediate_score = objective(beta0, beta1, sweep_sa, sign_batch, lr, weight_decay, instances, epochs, learning_batch, features1, features2, features3, features4, window)
    print("time ", time.time() - start)
    tune.report(mean_loss=intermediate_score)
    print("time+ ", time.time() - start)

sizes = [i for i in range(4,32,4)]

config={
        "features1": tune.choice(sizes),
        "features2": tune.choice(sizes),
        "features3": tune.choice(sizes),
        "features4": tune.choice(sizes),
        "window": tune.choice([3])
    }

ng_search = NevergradSearch(
    optimizer=ng.optimizers.OnePlusOne,
    metric="mean_loss",
    mode="min")

#ng_search = ConcurrencyLimiter(ng_search, max_concurrent=2)
#training_function = DistributedTrainableCreator(training_function, num_workers=2)

nevergradopt = tune.run(training_function, config=config, search_alg=ng_search, num_samples=150, resources_per_trial={"cpu": 20})
print("Best config: ", nevergradopt.get_best_config(
    metric="mean_loss", mode="min"))
