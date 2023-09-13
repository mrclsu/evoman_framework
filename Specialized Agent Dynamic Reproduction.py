#######################################################################################
# EvoMan EA			                              		                        	  #
# DEMO : perceptron neural network controller evolved by Genetic Algorithm        	  #
#                                                                                     #
# Author: Austin Dickerson       			                                  		  #
#######################################################################################
#%%
# imports framework
import sys, os
from evoman.environment import Environment
from demo_controller import player_controller
import numpy as np
import random


experiment_name = 'controller_specialist_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

env = Environment()

n_hidden_neurons = 10
num_vars = n_hidden_neurons*(env.get_num_sensors()+1) + (n_hidden_neurons+1)*5
replacement = 0.5
pop_size = 200
generations = 30

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

def combine_genes(p1, p2):
    temp = np.random.random()
    first_key = np.random.random(len(p1))
    first_key = [1 if x > temp else 0 for x in first_key]
    second_key = np.ones(len(p1))-first_key
    zygote1 = first_key*p1
    zygote2 = second_key*p2
    child = zygote1 + zygote2
    return child

def select_parents(pop, low, high):
    first = random.randint(low,high-1)
    second = random.randint(low,high-1)
    while first == second:
        second = random.randint(low,high-1)
    p1 = pop[first]
    p2 = pop[second]
    return p1, p2

def mutate_offspring(child):
    for i in range(len(child)):
        if random.random() < mutagenic_temperature:
            child[i] += random.uniform(-mutation_intensity, mutation_intensity)
            if child[i] < -1:
                child[i] = random.uniform(-1,1)
            if child[i] > 1:
                child[i] = random.uniform(-1,1)
    return child

def reproduce(pop, fit_scores):
    sorted_pairs = sorted(zip(pop, fit_scores), key=lambda x: x[1])
    pop, fit_scores = zip(*sorted_pairs)
    pop = list(pop)
    fit_scores = list(fit_scores)
    replaced = int(len(pop)*replacement)
    for i in range(replaced):
        p1, p2 = select_parents(pop, replaced, len(pop))
        child = combine_genes(p1, p2)
        pop[i] = mutate_offspring(child)
    return pop

def reseed_event(pop, fit_scores):
    print("RESEED")
    indices = sorted(range(len(fit_scores)), key=lambda x: fit_scores[x], reverse=True)[:5]
    for count, unit in enumerate(pop):
        if count not in indices:
            parent = random.randint(0,4)
            geno_fidelity = random.random()
            for i in range(len(unit)):
                if random.random() > geno_fidelity:
                    unit[i] = random.uniform(-1,1)
                else:
                    unit[i] = pop[indices[parent]][i]
    return pop

def test_mutation(mutagenic_temperature, mutation_intensity):
    globals() ['mutagenic_temperature'] = mutagenic_temperature
    globals() ['mutation_intensity'] = mutation_intensity
    performance = [0]  
    pop = initialize_population(pop_size, num_vars)
    best = 0
    reseed_timer = 0
    for i in range(generations):
        fit_scores, maxi, ind = test_population(pop)
        performance.append(maxi)
        if maxi >= max(performance):
            best = pop[ind]
        if (len(performance) > 5):
            if performance[-5] == performance[-1] and reseed_timer >= 5:
                print(f"Generation {i+1}:")
                pop = reseed_event(pop, fit_scores)
            else:
                reseed_timer += 1
                pop = reproduce(pop, fit_scores)
    print(performance)
    print(max(performance))
    return best

bests = []
for i in range(1,6):
    env = Environment(experiment_name=experiment_name,
				  playermode="ai",
                  enemies=[1],
				  player_controller=player_controller(n_hidden_neurons),
			  	  speed="fastest",
				  enemymode="static",
				  level=2,
				  visuals=False)
    best = test_mutation(0.5, 0.1*i)
    bests.append(best)
np.savetxt("Reseed_Boss_1.csv", bests, delimiter=",")

#%%
import sys, os
import numpy as np
import pandas as pd
from evoman.environment import Environment
from demo_controller import player_controller

def simulate(env,unit):
    fit, _, _, _ = env.play(pcont=unit)
    return fit

experiment_name = 'controller_specialist_demo'

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

bests2 = pd.read_csv("Reseed_Boss_1.csv", delimiter=",")
performance = []

for j in range(4):

    scores = []

    env = Environment(experiment_name=experiment_name,
				  playermode="ai",
                  enemies=[1],
				  player_controller=player_controller(n_hidden_neurons),
			  	  speed="normal",
				  enemymode="static",
				  level=2,
				  visuals=True)
    
    for i in range(5):    

        score = simulate(env,np.array(bests2.iloc[j]))
        scores.append(score)
    performance.append(np.mean(np.array(scores)))

for i in range(len(performance)):

    print(f"Mutation_Degree {(i+1)*0.1} average score {performance[i]}")



# %%
