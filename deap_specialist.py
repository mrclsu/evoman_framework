from deap import base, creator, tools, algorithms
from evoman.environment import Environment
import random 
import numpy as np
import time
import os
from evoman.controller import Controller

experiment_name = 'DEAP_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

class player_controller(Controller):
    def __init__(self, n_hidden):
        self.n_hidden = n_hidden 

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[1],
                  playermode="ai",
                  player_controller=player_controller(n_hidden=n_hidden_neurons),
                  enemymode="static",
                  level=1,
                  speed="fastest",
                  visuals=True)

# default environment fitness is assumed for experiment
env.state_to_log() # checks environment state

# genetic algorithm params
run_mode = 'train' # train or test
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

upper_bound = 1
lower_bound = -1
npop = 100
gens = 40
mutation_rate = 0.2
last_best = 0

def simulation(env,x):
    fitness, player_energy, enemy_energy, game_time = env.play(pcont=x)
    return fitness,

def normalize(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)

# Create individual 
creator.create("MaximizeFitness", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.MaximizeFitness)

# Evaluate individual
def evaluate_individual(individual):
    return simulation(env, individual)

toolbox = base.Toolbox()
individual_size = n_vars

toolbox.register("gene_initializer", random.uniform, lower_bound, upper_bound)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.gene_initializer, individual_size)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)  
toolbox.register("evaluate", evaluate_individual)
toolbox.register("mate", tools.cxBlend, alpha=0.2)  
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)  

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

if __name__ == "__main__":
    MU = 50  # Number of initial individuals
    LAMBDA = 50  # Number of children to produce at each generation
    CXPB = 0.7
    MUTPB = 0.2
    NGEN = 50
    
    pop = toolbox.population(n=MU)
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN,
                              stats=stats, halloffame=None, verbose=True)

# Hyperparameter optimization using GridSearch
#Initilize grids

#Grid for population size
npop_grid = [50,70,100]
mutation_rates = [0.1, 0.2, 0.3]
crossover_probs = [0.6, 0.7, 0.8]
tournament_sizes = [2, 3, 4]
generations = [10,20,30]


results = []

for pop_size in npop_grid:
    for mutation_rate in mutation_rates:
        for crossover_prob in crossover_probs:
            for tournament_size in tournament_sizes:
                for generation in generations:
                    #Set hyperparameters
                    MUTPB = mutation_rate
                    CXPB = crossover_prob
                    NGEN = generation
                    toolbox.unregister("select")
                    toolbox.register("select", tools.selTournament, tournsize=tournament_size)

                    #Algorithm
                    pop = toolbox.population(n=pop_size)  # Use pop_size from the loop
                    fitnesses = map(toolbox.evaluate, pop)
                    for ind, fit in zip(pop, fitnesses):
                        ind.fitness.values = fit

                    hof = tools.HallOfFame(1) # Keeping track of the winning solution  
                    algorithms.eaMuPlusLambda(pop, toolbox, pop_size, LAMBDA, CXPB, MUTPB, NGEN, stats=stats, halloffame=hof, verbose=True)
                    
                    performance = hof[0].fitness.values[0]  # Get the performance of the best individual

                    # Append results
                    results.append({
                        'population_size': pop_size,
                        'mutation_rate': mutation_rate,
                        'crossover_prob': crossover_prob,
                        'tournament_size': tournament_size,
                        'performance': performance,
                        'winning_weights': hof[0]  # The weights of the best solution found
                    })

                    # Print the results for this combination of hyperparameters
                    print("Hyperparameters for current iteration:")
                    print("Population Size:", pop_size)
                    print("Mutation Rate:", mutation_rate)
                    print("Crossover Probability:", crossover_prob)
                    print("Tournament Size:", tournament_size)
                    print("Number of Generations:", generation)
                    print("Performance:", performance)
                    print("-------------------------------")

# Find the best result across all parameter combinations
best_result = max(results, key=lambda x: x['performance'])

print("\n\nBest Overall Hyperparameters:")
print("Population Size:", best_result['population_size'])
print("Mutation Rate:", best_result['mutation_rate'])
print("Crossover Probability:", best_result['crossover_prob'])
print("Tournament Size:", best_result['tournament_size'])
print("Performance:", best_result['performance'])