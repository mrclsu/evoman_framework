from deap import base, creator, tools, algorithms
from evoman.environment import Environment
import random
import numpy as np
import time
import os
from demo_controller import player_controller
import csv
import pickle
import argparse

experiment_name = 'deap_specialist'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Parameters
n_hidden_neurons = 10
mu_size = 100  # Number of initial individuals
lambda_size = 100  # Number of children to produce at each generation
crossover_prob = 0.7
mutation_prob = 0.2
gen_count = 400
dom_u = 1
dom_l = -1


# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[3, 5, 6, 8],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  multiplemode="yes",
                  speed="fastest",
                  visuals=False)

# default environment fitness is assumed for experiment

env.state_to_log()  # checks environment state

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5


# runs simulation
def simulation(env, x):
    f, p, e, t = env.play(np.array(x))
    # print(f)
    return (f, p, e)

def evaluate(individual):
    return simulation(env, individual)

creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attribute", random.uniform, dom_l, dom_u)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=n_vars)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
stats.register("mean", np.mean)
stats.register("max", np.max)
stats.register("std", np.std)

def train(pop = toolbox.population(n=mu_size)): 
    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit


    env.enemies = [3, 5, 6, 8]
    hof = tools.HallOfFame(100)
    _, logbook1 = algorithms.eaMuCommaLambda(pop, toolbox, mu_size, lambda_size, crossover_prob, mutation_prob, gen_count,
                                stats=stats, halloffame=hof, verbose=True)

    env.enemies = [2, 3, 4]
    hof2 = tools.HallOfFame(100)
    _, logbook2 = algorithms.eaMuCommaLambda(list(hof), toolbox, mu_size, lambda_size, crossover_prob, mutation_prob, gen_count,
                                stats=stats, halloffame=hof2, verbose=True)


    env.enemies = [2, 3, 4, 5, 6, 8]
    hof10 = tools.HallOfFame(10)
    _, logbook2 = algorithms.eaMuCommaLambda(list(hof), toolbox, mu_size, lambda_size, crossover_prob, mutation_prob, gen_count,
                                stats=stats, halloffame=hof10, verbose=True)

    with open(f'{experiment_name}/hof1.pkl', 'wb') as f:
        pickle.dump(hof, f)
    
    with open(f'{experiment_name}/hof2.pkl', 'wb') as f:
        pickle.dump(hof2, f)

    with open(f'{experiment_name}/hof10.pkl', 'wb') as f:
        pickle.dump(hof10, f)

    with open(f"{experiment_name}/statistics_run1.csv", "w", newline="") as csvfile:
        fieldnames = ["gen", "mean", "max", "std"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for entry in logbook1:
            writer.writerow({
                'gen': entry['gen'],
                'mean': entry['mean'],
                'max': entry['max'],
                'std': entry['std']
            })

    with open(f"{experiment_name}/statistics_run2.csv", "w", newline="") as csvfile:
        fieldnames = ["gen", "mean", "max", "std"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for entry in logbook2:
            writer.writerow({
                'gen': entry['gen'],
                'mean': entry['mean'],
                'max': entry['max'],
                'std': entry['std']
            })

    with open(f"{experiment_name}/statistics_run3.csv", "w", newline="") as csvfile:
        fieldnames = ["gen", "mean", "max", "std"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for entry in logbook2:
            writer.writerow({
                'gen': entry['gen'],
                'mean': entry['mean'],
                'max': entry['max'],
                'std': entry['std']
            })

def run_pop():
    # Load pickled population
    with open(f'{experiment_name}/hof10.pkl', 'rb') as f:
        hof = pickle.load(f)

        # print(hof)

        global env
        env = Environment(experiment_name=experiment_name,
                    enemies=[2, 3, 4,],
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons),
                    enemymode="static",
                    level=2,
                    multiplemode="yes",
                    speed="fastest",
                    visuals=True)

        res = map(lambda p: (toolbox.evaluate(p), p), hof)
        fit = map(lambda r: r[0], res)
        for f in fit:
            print(f)
        # Run individual with highest fitness
        # fitness = simulation(env, pop[np.argmax(fit)]) 
        # print(f'Fitness: {fitness}')


def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run DEAP generalist algorithm')
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--run', action='store_true', help='Run population')

    args = parser.parse_args()

    if args.train:
        train()
    elif args.run:
        run_pop()

if __name__ == "__main__":
    main()