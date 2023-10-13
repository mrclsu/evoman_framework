from deap import base, creator, tools, algorithms
from evoman.environment import Environment
import random
import numpy as np
import time
import os
from demo_controller import player_controller

experiment_name = 'deap_specialist'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

best_genome_overall = None
best_genome_score = float('-inf')

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[1],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)

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
    global best_genome_score, best_genome_overall
    f, p, e, t = env.play(np.array(x))
    if f > best_genome_score:
        if f > best_genome_score:
            best_genome_score = f
            best_genome_overall = x
    #print(f)
    return f

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


def evaluate(individual):
    return simulation(env, individual),

def test_champion_on_all_enemies(champion):
    results = {}
    for enemy in range(1, 9):
        env.update_parameter('enemies', [enemy])
        fit_scores = [simulation(env, champion) for _ in range(5)] # Test 5 times for better average
        avg_score = np.mean(fit_scores)
        max_score = np.max(fit_scores)
        std_dev = np.std(fit_scores)

        results[enemy] = {
            'avg': avg_score,
            'max': max_score,
            'std_dev': std_dev
        }

    return results

def display_results(results):
    print("\nResults:")
    print("Enemy #\tAvg Fitness\tMax Fitness\tStandard Deviation")
    for enemy, res in results.items():
        print(f"{enemy}\t{res['avg']:.2f}\t\t{res['max']:.2f}\t\t{res['std_dev']:.2f}")



toolbox = base.Toolbox()
IND_SIZE = 10
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


if __name__ == "__main__":
    # Parameters
    MU = 100  # Number of initial individuals
    LAMBDA = 100  # Number of children to produce at each generation
    CXPB = 0.7
    MUTPB = 0.2
    NGEN = 100

    # Initialize population
    pop = toolbox.population(n=MU)

    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Run the algorithm
    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN,
                              stats=stats, halloffame=None, verbose=True)

        
    # Test best candidate
    results = test_champion_on_all_enemies(best_genome_overall)
    display_results(results)
