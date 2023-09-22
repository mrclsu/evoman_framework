from deap import base, creator, tools, algorithms
from evoman.environment import Environment
import random
import numpy as np
import time
import os
<<<<<<< HEAD
from demo_controller import player_controller
=======
import sys
from evoman.controller import Controller
from deap.tools import HallOfFame
>>>>>>> e66c16c (wip)

experiment_name = 'deap_specialist'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

original_stdout = sys.stdout
log_file = open(f"{experiment_name}/evoman_logs_1.txt", "a")
sys.stdout = log_file

n_hidden_neurons = 10


# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[1],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)

<<<<<<< HEAD
# default environment fitness is assumed for experiment

env.state_to_log()  # checks environment state

####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###

ini = time.time()  # sets time marker

# genetic algorithm params

run_mode = 'train'  # train or test

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

dom_u = 1
dom_l = -1
npop = 100
gens = 30
mutation = 0.2
last_best = 0


# runs simulation
def simulation(env, x):
    f, p, e, t = env.play(np.array(x))
    print(f)
    return f


# normalizes
def norm(x, pfit_pop):
    if (max(pfit_pop) - min(pfit_pop)) > 0:
        x_norm = (x - min(pfit_pop)) / (max(pfit_pop) - min(pfit_pop))
    else:
        x_norm = 0

    if x_norm <= 0:
        x_norm = 0.0000000001
    return x_norm


=======
# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

>>>>>>> e66c16c (wip)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


def evaluate(individual):
    return simulation(env, individual),


toolbox = base.Toolbox()
IND_SIZE = 10
<<<<<<< HEAD
toolbox.register("attribute", random.uniform, dom_l, dom_u)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=n_vars)
=======
toolbox.register("attribute", random.uniform, -1, 1)  
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=n_vars)  
>>>>>>> e66c16c (wip)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
stats.register("mean", np.mean)
stats.register("max", np.max)
stats.register("std", np.std)

<<<<<<< HEAD
=======
def doomsday_worst_elimination(population):
    sorted_pop = sorted(population, key=lambda x: x.fitness.values[0])
    worst = sorted_pop[:int(0.1 * len(sorted_pop))]  # Eliminate the worst 10%
    
    for ind in worst:
        for i in range(len(ind)):
            ind[i] = random.uniform(-1, 1)
    
>>>>>>> e66c16c (wip)
if __name__ == "__main__":
    # Parameters
    MU = 100  # Number of initial individuals
    LAMBDA = 100  # Number of children to produce at each generation
    CXPB = 0.7
    MUTPB = 0.2
<<<<<<< HEAD
    NGEN = 100
=======
    NGEN = 30
>>>>>>> e66c16c (wip)

    # Initialize population
    pop = toolbox.population(n=MU)

<<<<<<< HEAD
=======
    for gen in range(NGEN):
        offspring = algorithms.varOr(pop, toolbox, lambda_=LAMBDA, cxpb=CXPB, mutpb=MUTPB)
        fits = toolbox.map(toolbox.evaluate, offspring)
        
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        
        pop = toolbox.select(offspring + pop, k=MU)
        
        # Doomsday event every 20 generations
        if gen % 20 == 0:
            doomsday_worst_elimination(pop)
    
    
>>>>>>> e66c16c (wip)
    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
<<<<<<< HEAD

    # Run the algorithm
    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN,
                              stats=stats, halloffame=None, verbose=True)
=======
    
    hof = HallOfFame(5)
    
    # Run the algorithm
    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN,
                              stats=stats, halloffame=hof, verbose=True)

    # algorithms.eaSimple(pop, toolbox, CXPB, MUTPB, NGEN, stats=stats, halloffame=hof, verbose=True)



sys.stdout = original_stdout
log_file.close()
>>>>>>> e66c16c (wip)
