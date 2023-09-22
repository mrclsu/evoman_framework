########################################
# An Evolutionary Algorithm Framework  #
# By Austin Dickerson                  #
# 10/9/2023                            #
########################################


# imports evoman framework
import os, random, time
from evoman.environment import Environment
from demo_controller import player_controller
import numpy as np
import pandas as pd


experiment_name = 'controller_specialist'

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10
num_vars = n_hidden_neurons*(21) + (n_hidden_neurons+1)*5
pop_size = 100
generations = 100
early_stop = False
discrete = False
elitism = False
half = False
midpoint= False
mutation_intensity = 1
mutagenic_temperature = 0.2
curve_parents = False
random.seed(579)
np.random.seed(135)

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
        scores.append(simulate(env, unit))

    maximum = max(scores)
    ind = scores.index(maximum)

    return scores, maximum, ind

def normalize(scores):
    lo = min(scores)
    hi = max(scores)
    scores -= lo
    scores /= (hi-lo)

    return scores

def combine_genes_discrete(p1, p2):
    child = p1

    for i in range(len(p1)):

        if random.random() > 0.5:
            child[i] = p2[i]

    return child

def combine_genes_cross(p1, p2):
    temp = np.random.random()
    zygote1 = temp*p1
    zygote2 = (1-temp)*p2
    child = zygote1 + zygote2

    return child

def relative_prob(fit_scores):
    fit_norm = normalize(fit_scores)
    probabilities = fit_norm/fit_norm.sum()
    probabilities = np.nan_to_num(probabilities)
    probabilities += 0.001
    probabilities = probabilities / np.sum(probabilities)

    return probabilities

def select_parents(pop, low=None, high=None, probabilities=None):
    if curve_parents == True:
        selected = np.random.choice(len(pop), 2, p=probabilities, replace=False)
        p1 = pop[selected[0]]
        p2 = pop[selected[1]]

    else:
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

def sort(pop, fit_scores):
    sorted_pairs = sorted(zip(pop, fit_scores), key=lambda x: x[1])
    pop, fit_scores = zip(*sorted_pairs)

    return pop, fit_scores

def reproduce_generational(pop, probabilities):
    pop = list(pop)
    new_pop = []
    new_fit = []

    for _ in range(pop_size):
        p1, p2 = select_parents(pop, int(pop_size/2), len(pop), probabilities)

        if discrete == True:
            child = combine_genes_discrete(p1, p2)
            child2 = combine_genes_discrete(p1, p2)
        else:
            child = combine_genes_cross(p1, p2)
            child2 = combine_genes_cross(p1, p2)

        new_pop.append(mutate_offspring(child))
        new_pop.append(mutate_offspring(child2))
        new_fit.append(simulate(env, child))
        new_fit.append(simulate(env, child2))

    new_fit_norm = normalize(new_fit)
    probabilities = new_fit_norm/new_fit_norm.sum()
    probabilities = np.nan_to_num(probabilities)
    probabilities += 0.001
    probabilities = probabilities / np.sum(probabilities)

    if type(elitism) == int and (midpoint == True or half == False):
        selected = np.random.choice(pop_size*2, pop_size-elitism, p=probabilities,
                                replace=False)
        new_pop = np.array(new_pop)
        new_pop = new_pop[selected]
        new_pop = np.concatenate((new_pop, pop[-5:]))
    else:
        selected = np.random.choice(pop_size*2, pop_size, p=probabilities,
                                replace=False)
        new_pop = np.array(new_pop)
        new_pop = new_pop[selected]

    return new_pop

def reproduce_steady(pop, probabilities):
    pop = list(pop)
    new_pop = []
    new_fit = []
    next_gen = int(pop_size/10)

    for _ in range(next_gen):
        p1, p2 = select_parents(pop, int(pop_size/2), len(pop), probabilities)

        if discrete == True:
            child = combine_genes_discrete(p1, p2)
            child2 = combine_genes_discrete(p1, p2)
        else:
            child = combine_genes_cross(p1, p2)
            child2 = combine_genes_cross(p1, p2)

        new_pop.append(mutate_offspring(child))
        new_pop.append(mutate_offspring(child2))
        new_fit.append(simulate(env, child))
        new_fit.append(simulate(env, child2))

    new_fit_norm = normalize(new_fit)
    probabilities = new_fit_norm/new_fit_norm.sum()
    probabilities = np.nan_to_num(probabilities)
    probabilities += 0.001
    probabilities = probabilities / np.sum(probabilities)

    selected = np.random.choice(next_gen*2, next_gen, p=probabilities,
                                replace=False)
    
    new_pop = np.array(new_pop)
    new_pop = new_pop[selected]
    new_pop = np.concatenate((new_pop, pop[-(pop_size-next_gen):]))

    return new_pop

def reproduce_comma_strategy(pop, probabilities):
    pop = list(pop)
    new_pop = []
    new_fit = []
    next_gen = pop_size*3

    for _ in range(next_gen):
        p1, p2 = select_parents(pop, int(pop_size/2), len(pop), probabilities)

        if discrete == True:
            child = combine_genes_discrete(p1, p2)
        else:
            child = combine_genes_cross(p1, p2)

        new_pop.append(mutate_offspring(child))
        new_fit.append(simulate(env, new_pop[-1]))

    new_pop, _ = sort(new_pop, new_fit)
    new_pop = np.array(new_pop)
    new_pop = new_pop[-(pop_size-1):]
    new_pop = np.concatenate((new_pop, [pop[-1]]))

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

def training_run(mutagenic_temperature, mutation_intensity, mutation_reset, discrete, steady, comma, reseed_cycle, elitism, half, curve_parents):
    globals() ['mutagenic_temperature'] = mutagenic_temperature
    globals() ['mutation_intensity'] = mutation_intensity
    globals() ['mutation_reset'] = mutation_reset
    globals() ['reseed_cycle'] = reseed_cycle
    globals() ['discrete'] = discrete
    globals() ['elitism'] = elitism
    globals() ['half'] = half
    globals() ['curve_parents'] = curve_parents
    mean_stat = [0]
    peak_stat = [0]
    st_devs = []
    upper_avg_stat = [0]
    performance = [0]
    upper_avg = [0]  
    pop = initialize_population(pop_size, num_vars)
    best = 0
    reseed_count = 0

    for i in range(generations):
        fit_scores, maxi, ind = test_population(pop)
        performance.append(np.round(maxi, decimals=2))
        upper_avg = np.mean(fit_scores[-10:])
        st_dev = np.std(fit_scores)

        if i > (generations/2):
            globals() ['midpoint'] = True

        if upper_avg >= max(upper_avg_stat):
            best = pop[ind]

        upper_avg_stat.append(np.round(upper_avg, decimals=2))
        mean_stat.append(np.round(np.mean(fit_scores), decimals=2))
        peak_stat.append(np.round(max(fit_scores), decimals=2))
        st_devs.append(st_dev)

        if reseed_cycle == True and i%20 == 9:
            pop = reseed_event(pop, fit_scores)

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
                pop, fit_scores = sort(pop, fit_scores)
                probabilities = relative_prob(fit_scores)

                if steady == True:
                    pop = reproduce_steady(pop, probabilities)
                elif comma == True:
                    pop = reproduce_comma_strategy(pop, probabilities)
                else:
                    pop = reproduce_generational(pop, probabilities)

        else:
            pop, fit_scores = sort(pop, fit_scores)
            probabilities = relative_prob(fit_scores)

            if steady == True:
                pop = reproduce_steady(pop, probabilities)
            elif comma == True:
                pop = reproduce_comma_strategy(pop, probabilities)
            else:
                pop = reproduce_generational(pop, probabilities)

        print(f"Generation {i+1} top 10% avg: {np.round(upper_avg, decimals=2)}")

        if upper_avg >= 99:
            early_stop = True
            print(performance)
            print(max(performance))
            return best, mean_stat, peak_stat, pop[ind], upper_avg_stat, st_devs

    print(performance)
    print(max(performance))
    
    return best, mean_stat, peak_stat, pop[ind], upper_avg_stat, st_devs

def train_set(string="V0", mutagenic_temperature=0.2, mutation_intensity=1, 
              mutation_reset=False, discrete=False, steady=False, comma=False, 
              reseed_cycle=False, elitism=False, half=False, curve_parents=False,
              set=[1,2,3,4,5,6,7,8]):
    
    bests = []
    means = []
    peaks = []
    lasts = []
    up_avg = []
    st_devs = []
    times = [time.time_ns()]

    for j in set:
        global env
        env = Environment(experiment_name=experiment_name,
				    playermode="ai",
                    enemies=[j],
				    player_controller=player_controller(n_hidden_neurons),
			  	    speed="fastest",
				    enemymode="static",
				    level=2,
				    visuals=False)
        best, mean_stat, peak_stat, last_best, upper_avg_stat, st_dev = training_run(mutagenic_temperature, mutation_intensity, mutation_reset, discrete, steady, comma, reseed_cycle, elitism, half, curve_parents)
        bests.append(best)
        means.append(mean_stat)
        peaks.append(peak_stat)
        lasts.append(last_best)
        up_avg.append(upper_avg_stat)
        st_devs.append(st_dev)
        times.append(time.time_ns())

    np.savetxt(f"{string}_Params.csv", bests, delimiter=",")
    np.savetxt(f"{string}_Last_Params.csv", lasts, delimiter=",")
    np.savetxt(f"{string}_Timestamps.csv", times, delimiter=",")

    if early_stop != True:
        np.savetxt(f"{string}_Means.csv", means, delimiter=",")
        np.savetxt(f"{string}_Peaks.csv", peaks, delimiter=",")
        np.savetxt(f"{string}_Upper_Avg.csv", up_avg, delimiter=",")
        np.savetxt(f"{string}_St_Dev.csv", st_devs, delimiter=",")

    print(f'Timestamps: {times}')

def test_params(string, runs=1, set=[1,2,3,4,5,6,7,8]):
    bests = pd.read_csv(f"{string}_Params.csv", delimiter=",", header=None)
    performance = []
    print(bests)

    for count, j in enumerate(set):
        scores = []
        env = Environment(experiment_name=experiment_name,
				  playermode="ai",
                  enemies=[j],
				  player_controller=player_controller(n_hidden_neurons),
			  	  speed="normal",
				  enemymode="static",
				  level=2,
				  visuals=True)
    
        for i in range(runs):    
            score = simulate(env,np.array(bests.iloc[count]))
            scores.append(score)

        performance.append(np.mean(np.array(scores)))

    for i in range(len(performance)):
        print(f"Boss Number {i+1} average score {np.round(performance[i], decimals=2)}")

    return performance

#Train New GAs

filename = "V22.1_Test"
train_set(filename, elitism=5, half=False, mutagenic_temperature=0.1, curve_parents=True, 
          discrete=False, reseed_cycle=False, set=[1,4,6])

#Test the best parameters

performance = test_params(filename, 1, set=[1,4,6])
