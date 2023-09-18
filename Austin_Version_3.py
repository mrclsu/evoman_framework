#######################################################################################
# EvoMan EA			                              		              #
# DEMO : perceptron neural network controller evolved by Genetic Algorithm            #
#                                                                                     #
# Author: Austin Dickerson       			                              #
#######################################################################################
# imports framework
import sys, os, random
from evoman.environment import Environment
from demo_controller import player_controller
import numpy as np
import pandas as pd


experiment_name = 'controller_specialist_demo'

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10
num_vars = n_hidden_neurons*(21) + (n_hidden_neurons+1)*5
replacement = 0.5
pop_size = 100
generations = 50

def initialize_population(pop_size, num_vars):
    pop = []

    for i in range(pop_size):
        member = np.zeros(num_vars)

        for j in range(num_vars):
            member[j] = random.uniform(-1,1)

        pop.append(member)   

    return pop
    
def simulate(env,unit):
    fit, _, _, _ = env.play(pcont=unit)

    return fit

def test_population(pop):
    scores = []

    for unit in pop:
        scores.append(simulate(env,unit))

    maximum = max(scores)
    ind = scores.index(maximum)
    scores = normalize(scores)

    return scores, maximum, ind

def normalize(scores):
    lo = min(scores)
    hi = max(scores)
    scores -= lo
    scores /= (hi-lo)

    return scores

def merge_genes(p1, p2):
    temp = np.random.random()
    zygote1 = temp*p1
    zygote2 = (1-temp)*p2
    child = zygote1 + zygote2

    return child

def combine_genes(p1, p2):
    child = p1

    for i in range(len(p1)):

        if random.random() > 0.5:
            child[i] = p2[i]

    return child

def select_parents(pop, low, high):
    first = random.randint(low,high-1)
    second = random.randint(low,high-1)

    while first == second:
        second = random.randint(low,high-1)

    p1 = pop[first]
    p2 = pop[second]

    return p1, p2

def mutate_offspring(child, reset=True):
    for i in range(len(child)):

        if random.random() < mutagenic_temperature:

            if reset == True:
                child[i] = np.random.uniform(-1,1)
            else:
                child[i] += np.random.normal(0, mutation_intensity)

                if child[i] < -1:
                    child[i] = -1
                if child[i] > 1:
                    child[i] = 1

    return child

def reproduce(pop, fit_scores, combine):
    sorted_pairs = sorted(zip(pop, fit_scores), key=lambda x: x[1])
    pop, fit_scores = zip(*sorted_pairs)
    pop = list(pop)
    new_pop = []
    new_fit = []

    for i in range(pop_size):
        p1, p2 = select_parents(pop, int(pop_size/2), len(pop))

        if combine == True:
            child = combine_genes(p1, p2)
            child2 = combine_genes(p1, p2)
        else:
            child = merge_genes(p1, p2)
            child2 = merge_genes(p1, p2)

        new_pop.append(mutate_offspring(child))
        new_pop.append(mutate_offspring(child2))
        new_fit.append(simulate(env, child))
        new_fit.append(simulate(env, child2))

    new_fit_norm = normalize(new_fit)
    probabilities = new_fit_norm/new_fit_norm.sum()

    selected = np.random.choice(pop_size*2, pop_size-5, p=probabilities,
                                replace=False)
    new_pop = np.array(new_pop)
    new_pop = new_pop[selected]
    new_pop = np.concatenate((new_pop, pop[-5:]))

    return new_pop

def reseed_event(pop, fit_scores):
    print("RESEED")
    indices = sorted(range(len(fit_scores)), key=lambda x: fit_scores[x], reverse=True)[:5]
    indices2 = sorted(range(len(fit_scores)), key=lambda x: fit_scores[x], reverse=True)[int(-len(pop)/4):]
    
    for count, unit in enumerate(pop):

        if count in indices2:
            parent = random.randint(0,4)
            geno_fidelity = random.random()

            for i in range(len(unit)):

                if random.random() > geno_fidelity:
                    unit[i] = random.uniform(-1,1)
                else:
                    unit[i] = pop[indices[parent]][i]

    return pop

def test_mutation(mutagenic_temperature, mutation_intensity, combine):
    globals() ['mutagenic_temperature'] = mutagenic_temperature
    globals() ['mutation_intensity'] = mutation_intensity
    mean_stat = []
    peak_stat = []
    upper_avg_stat = [0]
    performance = [0]
    upper_avg = [0]  
    pop = initialize_population(pop_size, num_vars)
    best = 0
    reseed_count = 0

    for i in range(generations):
        fit_scores, maxi, ind = test_population(pop)
        performance.append(maxi)
        upper_avg = np.mean(fit_scores[-10:])

        if upper_avg >= max(upper_avg_stat):
            best = pop[ind]

        upper_avg_stat.append(upper_avg)
        mean_stat.append(np.mean(fit_scores))
        peak_stat.append(max(fit_scores))

        if upper_avg < upper_avg_stat[-2]:
            reseed_count += 1
        else:
            reseed_count = 0

        if (len(performance) > 5):

            if reseed_count >= 10:
                print(f"Generation {i+1}:")
                pop = reseed_event(pop, fit_scores)
                reseed_count = 0
            else:
                pop = reproduce(pop, fit_scores, combine)

        else:
            pop = reproduce(pop, fit_scores, combine)

        print(upper_avg)

        if upper_avg >= 0.98:
            print(performance)
            print(max(performance))
            return best, mean_stat, peak_stat

    print(performance)
    print(max(performance))
    
    return best, mean_stat, peak_stat

def train_all_8(string, mutation_intensity=1, combine=True):
    bests = []
    means = []
    peaks = []

    for j in range(1,9):
        global env
        env = Environment(experiment_name=experiment_name,
				    playermode="ai",
                    enemies=[j],
				    player_controller=player_controller(n_hidden_neurons),
			  	    speed="fastest",
				    enemymode="static",
				    level=2,
				    visuals=False)
        best, mean_stat, peak_stat = test_mutation(1/pop_size, mutation_intensity, combine)
        bests.append(best)
        means.append(mean_stat)
        peaks.append(peak_stat)

    np.savetxt(f"{string}_Params.csv", bests, delimiter=",")
    #np.savetxt(f"{string}_Means.csv", means, delimiter=",")
    #np.savetxt(f"{string}_Peaks.csv", peaks, delimiter=",")

def test_params(string):
    bests = pd.read_csv(f"{string}_Params.csv", delimiter=",", header=None)
    performance = []
    print(bests)

    for j in range(1,9):
        scores = []

        env = Environment(experiment_name=experiment_name,
				  playermode="ai",
                  enemies=[j],
				  player_controller=player_controller(n_hidden_neurons),
			  	  speed="normal",
				  enemymode="static",
				  level=2,
				  visuals=True)
    
        for i in range(5):    
            score = simulate(env,np.array(bests2.iloc[j-1]))
            scores.append(score)

        performance.append(np.mean(np.array(scores)))

    for i in range(len(performance)):
        print(f"Boss Number {i+1} average score {performance[i]}")

    return performance

train_all_8("V18")
test_params("V18")
