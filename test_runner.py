import annealing_sign_problem.square_deep
import numpy as np
#import nevergrad as ng
#from ray.tune.integration.torch import DistributedTrainableCreator
#from hyperopt import tpe, hp, fmin
#import ray
#from ray import tune
#from ray.tune.suggest import ConcurrencyLimiter
#from ray.tune.suggest.nevergrad import NevergradSearch
import sys
import time
import matplotlib.pyplot as plt

#ray.init(address='auto')

def objective(beta0, beta1, sweep_sa, sign_batch, lr, weight_decay, instances, epochs, learning_batch, features1, features2, features3, window):

    return annealing_sign_problem.square_deep.main(beta0, beta1, sweep_sa, sign_batch, lr, weight_decay, instances, epochs, learning_batch, features1, features2, features3, window)

start = time.time()

ising_energies = np.array([])
betas = np.array([])
sweeps = np.array([])

for sweep in [350000]:
    print("beta1 = ", 0.02)
    isen = objective(1e-4, 0.02, sweep, 5000, 1e-3, 5e-5, 1, 1, 1024, 28, 28, 20, 5)[1]
    ising_energies = np.append(ising_energies, isen)
#    betas = np.append(betas, beta1/100)
    sweeps = np.append(sweeps, sweep)

#    objective(1e-4, 1e-4, sweep, 5000, 1e-3, 5e-5, 1, 1, 1024, 28, 28, 20, 5)[1]

    print(ising_energies)

    print("time ", time.time() - start)

#plt.plot(betas, ising_energies, 'ro')
plt.plot(sweeps, ising_energies, 'ro')
plt.savefig('ising_energies.pdf')

print(np.mean(ising_energies), np.std(ising_energies)/np.mean(ising_energies))

sys.exit()

def toy_function(config):

    lr = config["lr"]
    wd = config["wd"]
    score = (lr-2)**2 + (wd+3.28)**4
    tune.report(mean_loss=score)

def training_function(config):
    # Hardcoded hyperparameters

    beta0 = 10
    beta1 = 10000
    sweep_sa = 10000
    sign_batch = 5000
    lr = 1e-3
    weight_decay = 5e-5
    instances = 20
    epochs = 100
    learning_batch = 1024

    # Variable hyperparameters

    features1, features2, features3, window = config["features1"], config["features2"], config["features3"], config["window"]

    start = time.time()
    intermediate_score = objective(beta0, beta1, sweep_sa, sign_batch, lr, weight_decay, instances, epochs, learning_batch, features1, features2, features3, window)
    print("time ", time.time() - start)
    tune.report(mean_loss=intermediate_score)
    print("time+ ", time.time() - start)

sizes = [i for i in range(4,32,4)]

config={
        "features1": tune.choice([28]),
        "features2": tune.choice([28]),
        "features3": tune.choice([20]),
        "window": tune.choice([5])
    }

ng_search = NevergradSearch(
    optimizer=ng.optimizers.OnePlusOne,
    metric="mean_loss",
    mode="min")

#ng_search = ConcurrencyLimiter(ng_search, max_concurrent=2)
#training_function = DistributedTrainableCreator(training_function, num_workers=2)

nevergradopt = tune.run(training_function, config=config, search_alg=ng_search, num_samples=1, resources_per_trial={"cpu": 20})
print("Best config: ", nevergradopt.get_best_config(
    metric="mean_loss", mode="min"))
