from deap import base, creator, tools, algorithms
from evoman.environment import Environment
import random
import numpy as np
import time
import os
from demo_controller import player_controller

experiment_name = 'deap_generalist'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10
best_genome_overall = None
best_genome_score = float('-inf')

ENEMIES = [1, 2, 3, 4, 5, 6, 7, 8]
ENEMY_TRAIN_SET = [1,4,5,7]

# initializes simulation in individual evolution mode, for single static enemy.
envs = {enemy: Environment(experiment_name=experiment_name,
                           enemies=[enemy],
                           playermode="ai",
                           player_controller=player_controller(n_hidden_neurons),
                           enemymode="static",
                           level=2,
                           speed="fastest",
                           visuals=False) for enemy in ENEMY_TRAIN_SET}

# default environment fitness is assumed for experiment

#env.state_to_log()  # checks environment state

####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###

ini = time.time()  # sets time marker

# genetic algorithm params

run_mode = 'train'  # train or test

# number of weights for multilayer with 10 hidden neurons
n_vars = (list(envs.values())[0].get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

dom_u = 1
dom_l = -1
npop = 100
gens = 100
mutation = 0.2
last_best = 0

# runs simulation
def simulation(x):
    global best_genome_score, best_genome_overall, envs
    current_env = envs[ENEMY_TRAIN_SET[0]]
    f, p, e, t = current_env.play(np.array(x))
    if f > best_genome_score:
        best_genome_score = f
        best_genome_overall = x
    return f


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


def evaluate(individual):
    return simulation(individual),

def test_champion_on_all_enemies(champion):
    results = {}
    for enemy in range(1, 9):
        chosen_env = envs.get(enemy) or envs[ENEMY_TRAIN_SET[0]]  # Get the env for the enemy or fallback to a default env
        chosen_env.update_parameter('enemies', [enemy])
        fit_scores = [simulation(champion) for _ in range(5)]
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


def doomsday(population, generation, interval=20, survival_rate=0.):
    if generation % interval == 0:
        num_survivors = int(len(population) * survival_rate)
        survivors = tools.selBest(population, num_survivors)  # Keeping the best as survivors
        new_individuals = [toolbox.individual() for _ in range(len(population) - num_survivors)]
        
        # Replace old population with the survivors and new individuals
        population[:num_survivors] = survivors
        population[num_survivors:] = new_individuals


toolbox = base.Toolbox()
IND_SIZE = 10

creator.create("Individual", list, fitness=creator.FitnessMax, mutation_rate=float)


def init_individual(icls, content):
    ind = icls(content[:-1])
    ind.mutation_rate = content[-1]
    return ind

def attribute():
    return [random.uniform(dom_l, dom_u) for _ in range(n_vars)] + [random.uniform(0.2, 0.5)]

toolbox.register("individual", init_individual, creator.Individual, content=attribute())

def self_adaptive_mutate(individual):
    if random.random() < individual.mutation_rate:
        # Perform mutation on the solution
        for i in range(len(individual) - 1):
            if random.random() < individual.mutation_rate:
                individual[i] = random.uniform(dom_l, dom_u)
        
    # Mutate the mutation rate itself
    individual.mutation_rate += random.uniform(-0.01, 0.01)
    individual.mutation_rate = max(0.1, min(0.3, individual.mutation_rate))  # Keep in [0.01, 0.1] range
    return individual,

toolbox.register("mutate", self_adaptive_mutate)

def custom_crossover(ind1, ind2):
    # Blend crossover for solution
    for i in range(len(ind1) - 1):  # -1 to avoid mutation rate
        alpha = random.random()
        ind1[i], ind2[i] = alpha * ind1[i] + (1 - alpha) * ind2[i], alpha * ind2[i] + (1 - alpha) * ind1[i]
    
    # Average crossover for mutation rates
    ind1.mutation_rate, ind2.mutation_rate = (ind1.mutation_rate + ind2.mutation_rate) / 2, (ind1.mutation_rate + ind2.mutation_rate) / 2
    
    return ind1, ind2

toolbox.register("mate", custom_crossover)



toolbox.register("attribute", random.uniform, dom_l, dom_u)
#toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=n_vars)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
#toolbox.register("mate", tools.cxBlend, alpha=0.5)
#toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=5)

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
