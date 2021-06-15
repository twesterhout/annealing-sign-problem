import annealing_sign_problem.square_4x4
import sys

def objective(beta0, beta1, sweep_sa, sign_batch, lr, weight_decay, instances, epochs, learning_batch):

    learning_batch = int(learning_batch)
    return annealing_sign_problem.square_4x4.main(beta0, beta1, sweep_sa, sign_batch, lr, weight_decay, instances, epochs, learning_batch)

objective(10, 10000, 10000, 10000, 1e-3, 5e-5, 10, 100, 1024)
