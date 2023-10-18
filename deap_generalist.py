from deap import base, creator, tools, algorithms
from research_envs import default_env, no_time_env, time_bonus_env
import random
import numpy as np
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
mu_size = 50  # Number of initial individuals
lambda_size = 50 # Number of children to produce at each generation
crossover_prob = 0.7
mutation_prob = 0.2
gen_count = 250
dom_u = 1
dom_l = -1

# initializes simulation in individual evolution mode, for single static enemy.
env = default_env(experiment_name, player_controller(n_hidden_neurons))
# default environment fitness is assumed for experiment

env.state_to_log()  # checks environment state

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

additional_data = []

# runs simulation
def simulation(env, x):
    f, p, e, t = env.play(np.array(x))
    return f, p, e, t

evaluations = 0

def evaluate(individual):
    f, p, e, t = simulation(env, individual)

    additional_data.append((p, e, t))

    return f,

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
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
stats.register("mean_fitness", np.mean)
stats.register("max_fitness", np.max)
stats.register("std_fitness", np.std)


def train(run = 0, file_name_prefix = ''): 
    additional_data.clear()
    pop = toolbox.population(n=mu_size)
    hof = tools.HallOfFame(100)
    _, logbook1 = algorithms.eaMuCommaLambda(pop, toolbox, mu_size, lambda_size, crossover_prob, mutation_prob, gen_count,
                                stats=stats, halloffame=hof, verbose=True)

    with open(f'{experiment_name}/{file_name_prefix}{run}_hof100.pkl', 'wb') as f:
        pickle.dump(hof, f)

    with open(f"{experiment_name}/{file_name_prefix}{run}_statistics.csv", "w", newline="") as csvfile:
        fieldnames = ['gen', 'mean_fitness', 'max_fitness', 'std_fitness', 'mean_player_life', 'mean_enemy_life', 'mean_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()

        prev_evals = 0
        current_evals = 0

        for entry in logbook1:
            gen = entry['gen']
            current_evals = entry['nevals']

            lower_ind = prev_evals
            upper_ind = prev_evals + current_evals

            writer.writerow({
                'gen': gen,
                'mean_fitness': entry['mean_fitness'],
                'max_fitness': entry['max_fitness'],
                'std_fitness': entry['std_fitness'],
                'mean_player_life': np.mean([x[0] for x in additional_data[lower_ind:upper_ind]]),
                'mean_enemy_life': np.mean([x[1] for x in additional_data[lower_ind:upper_ind]]),
                'mean_time': np.mean([x[2] for x in additional_data[lower_ind:upper_ind]])
            })

            prev_evals = current_evals

def run_train(run = 1, enemy = 0):
    global env

    enemies = [2,3,4,5,6,8] if enemy == 0 else [1,4,6,8]

    envs = {
        'def': default_env,
        'no_time': no_time_env,
        'bonus_time': time_bonus_env,
    }

    for key, e in envs.items():
        env = e(experiment_name, player_controller(n_hidden_neurons))
        env.enemies = enemies
        train(file_name_prefix=f'{key}_{enemy}_', run=run)

def run_pop():
    # Load pickled population
    with open(f'{experiment_name}/hof1.pkl', 'rb') as f:
        hof = pickle.load(f)

        global env
        env = default_env(experiment_name, player_controller(n_hidden_neurons))
        env.enemies = [1, 2, 3, 4, 5, 6, 7, 8]

        res = map(lambda p: (toolbox.evaluate(p), p), hof)
        fit = map(lambda r: r[0], res)
        for f in fit:
            print(f)

        
    np.savetxt('winner.txt', hof[np.argmax(fit)])
    winner = hof[np.argmax(fit)]

        # Run individual with highest fitness
    env.visuals = True
    env.speed = 'normal'
    env.multiplemode = 'no'

    for i in range(1, 9):
        env.enemies = [i]
        f, p, e, t = simulation(env, winner)
        print(f'Enemy: {i}, Fitness: {f}, Gain: {p - e}, Time: {t}')
    # fitness = simulation(env, hof[np.argmax(fit)]) 
    # print(hof[np.argmax(fit)])
    # print(f'Fitness: {fitness}')


def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run DEAP generalist algorithm')
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--run', action='store_true', help='Run population')
    parser.add_argument('--instance', type=int, help='Instance number')
    parser.add_argument('--enemy', type=int, help='Enemy number')

    args = parser.parse_args()

    if args.train and args.instance is not None and args.enemy is not None:
        run_train(run=args.instance, enemy=args.enemy)
    elif args.run:
        run_pop()

if __name__ == "__main__":
    main()