########################################
# A Genetic Algorithm Framework        #
# By Austin Dickerson                  #
# 10/9/2023                            #
########################################


# imports evoman framework
import os, random, time, csv
from evoman.environment import Environment
from demo_controller import player_controller
import numpy as np
import pandas as pd

# NN Parameters
n_hidden_neurons = 10
num_vars = n_hidden_neurons*(21) + (n_hidden_neurons+1)*5

# Experiment Parameters
pop_size = 100
generations = 10
early_stop = False

# Mutation Parameters
mutation_intensity = 1
mutagenic_temperature = 0.2

# Reproduction Parameters
discrete = False
individual_cross = False
curve_parents = False\

# Elitism Parameters
elitism = False
midpoint = False
half = False

# Species Paremeters
speciate = False
threshold = False
speciation_frequency = False

# Reproducability
random.seed(579)
np.random.seed(135)


filename = "TEST24"

experiment_name = 'controller_specialist'

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

stat_directory = "GA_Statistics"

if not os.path.exists(f'{stat_directory}_{filename}'):
    os.makedirs(f'{stat_directory}_{filename}')


# Create the ramized population
def initialize_population(pop_size, num_vars):
    pop = []

    for i in range(pop_size):
        member = np.zeros(num_vars)

        for j in range(num_vars):
            member[j] = random.uniform(-1,1)

        pop.append(member)   

    return pop

# Simulate to return just fit score
def simulate(env,unit):
    fit, _, _, _ = env.play(pcont=unit)

    return fit

# Run simulation for entire pop
def test_population(pop):
    scores = []

    for unit in pop:
        fit, p_life, e_life, gametime = env.play(pcont=unit)
        scores.append(fit)

    maximum = max(scores)
    ind = scores.index(maximum)

    return scores, maximum, ind

# Normalize with lowest ~~0
def normalize(scores):
    lo = min(scores)
    hi = max(scores)
    scores -= lo
    scores /= (hi-lo)

    return scores

# Normalize on full range
def species_normalize(scores):
    scores = np.array(scores)
    scores += 10
    scores /= 110

    return scores

# Return as normalized probabilities
def relative_prob(fit_scores):
    probabilities = np.array(fit_scores)/np.array(fit_scores).sum()
    probabilities = np.nan_to_num(probabilities)
    probabilities += 0.001
    probabilities = probabilities / np.sum(probabilities)

    return probabilities

# Find 2 parents to reproduce
def select_parents(pop, low=None, high=None, probabilities=None):

    # Select based on proportion to fitness scores
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

# Take one allele from each parent
def combine_genes_discrete(p1, p2):
    child = p1

    for i in range(len(p1)):

        if random.random() > 0.5:
            child[i] = p2[i]

    return child

# Merge all alleles at a random split ratio
def combine_genes_cross(p1, p2):
    temp = np.random.random()
    zygote1 = temp*p1
    zygote2 = (1-temp)*p2
    child = zygote1 + zygote2

    return child

# Merge each allele individually
def combine_genes_individual_cross(p1, p2):
    temp = np.random.rand(num_vars)
    zygote1 = temp*p1
    zygote2 = (1-temp)*p2
    child = zygote1 + zygote2

    return child

# Alter some alleles in a new child
def mutate_offspring(child, reset=True):
    for i in range(len(child)):

        if random.random() < mutagenic_temperature:

            # Either reset the allele or alter the existing
            if reset == True:
                child[i] = np.random.uniform(-1,1)
            else:
                child[i] += np.random.normal(0, mutation_intensity)

                if child[i] < -1:
                    child[i] = -1
                if child[i] > 1:
                    child[i] = 1

    return child

# List population in ascending order of fit score
def sort(pop, fit_scores):
    sorted_pairs = sorted(zip(pop, fit_scores), key=lambda x: x[1])
    pop, fit_scores = zip(*sorted_pairs)

    return pop, fit_scores

# Reproduce with all parents replaced by offspring and only parents from above the 50th %tile reproducing
def reproduce_generational(pop, probabilities):
    pop = list(pop)
    new_pop = []
    new_fit = []

    # Select and replace all the parents with their offspring and update fit scores
    for _ in range(pop_size):
        p1, p2 = select_parents(pop, int(pop_size/2), len(pop), probabilities)

        if discrete == True:
            child = combine_genes_discrete(p1, p2)
            child2 = combine_genes_discrete(p1, p2)
        elif individual_cross == True:
            child = combine_genes_individual_cross(p1, p2)
            child2 = combine_genes_individual_cross(p1, p2)
        else:
            child = combine_genes_cross(p1, p2)
            child2 = combine_genes_cross(p1, p2)

        new_pop.append(mutate_offspring(child))
        new_pop.append(mutate_offspring(child2))
        new_fit.append(simulate(env, child))
        new_fit.append(simulate(env, child2))

    new_fit_norm = species_normalize(new_fit)
    probabilities = relative_prob(new_fit_norm)

    # Copy the top performers or let them be replaced
    if type(elitism) == int and (midpoint == True or half == False):
        selected = np.random.choice(pop_size*2, pop_size-elitism, p=probabilities, replace=False)
        new_pop = np.array(new_pop)
        new_pop = new_pop[selected]
        new_pop = np.concatenate((new_pop, pop[-elitism:]))
    else:
        selected = np.random.choice(pop_size*2, pop_size, p=probabilities, replace=False)
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
        elif individual_cross == True:
            child = combine_genes_individual_cross(p1, p2)
            child2 = combine_genes_individual_cross(p1, p2)
        else:
            child = combine_genes_cross(p1, p2)
            child2 = combine_genes_cross(p1, p2)

        new_pop.append(mutate_offspring(child))
        new_pop.append(mutate_offspring(child2))
        new_fit.append(simulate(env, child))
        new_fit.append(simulate(env, child2))

    new_fit_norm = species_normalize(new_fit)
    probabilities = relative_prob(new_fit_norm)

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

    if type(elitism) == int and (midpoint == True or half == False):
        new_pop, _ = sort(new_pop, new_fit)
        new_pop = np.array(new_pop)
        new_pop = new_pop[-(pop_size-elitism):]
        new_pop = np.concatenate((new_pop, [pop[-elitism]]))
    else:
        new_pop, _ = sort(new_pop, new_fit)
        new_pop = np.array(new_pop)
        new_pop = new_pop[-pop_size:]

    return new_pop



def reproduce_by_species(pop, generation, species=None):

    # Speciate based on frequency
    if curve_parents == True and generation == 0:
        species_count, _, species = speciate_population(threshold, pop)
    elif curve_parents == True and (generation%speciation_frequency) == 0:
        species_count, _, species = speciate_population(threshold, pop)
    else:
        species_count, _, species = speciate_population(threshold, pop[-pop_size:])
    
    species_max = []
    species_avg = []
    new_pop = []
    norm_avg = []
    count_each_species = []
    probability_sets = []

    for i in range(species_count):
        scores, maximum, _ = test_population(species[f'species_{i+1}_list'])
        probability_sets.append(relative_prob(scores))
        scores2 = species_normalize(scores)
        norm_avg.append(np.average(scores2))
        species_max.append(maximum)
        species_avg.append(np.average(scores))

    for k in range(species_count):
        count_each_species.append(len(species[f'species_{k+1}_list']))

    print(f'Species Counts: {count_each_species}')
    print(f'Species Maximums: {np.round(species_max,2)}')
    print(f'Species Averages: {np.round(species_avg,2)}')
    probabilities = relative_prob(norm_avg)
    ratios = pop_size*probabilities
    ratios = [round(num) for num in ratios]
    approved = np.ones(species_count)
    reallocation = 0

    for i in range(species_count):

        if count_each_species[i] < 4:
                reallocation += ratios[i]
                approved[i] = 0

    probabilities = probabilities*approved
    probabilities = probabilities/probabilities.sum()
    ratios1 = ratios*approved
    ratios2 = reallocation*probabilities + ratios1
    ratios3 = [round(num) for num in ratios2]

    old_pop_ratios = np.array(count_each_species)*approved
    old_pop_ratios = (old_pop_ratios/old_pop_ratios.sum())*pop_size
    old_pop_ratios = [round(num) for num in old_pop_ratios]
    ratios = (np.array(ratios3)+np.array(old_pop_ratios))/2
    ratios = [round(num) for num in ratios]

    print(f'Old allocation: {old_pop_ratios}')
    print(f'After smoothing, species are now allocated: {ratios}')
    print(f'Totaling: {sum(ratios)}')

    species2 = {}

    for i in range(species_count):

        species2[f'species_{i+1}_list'] = []
        
        if count_each_species[i] >= 4:

            for _ in range(ratios[i]):

                if curve_parents == True:
                    p1, p2 = select_parents(species[f'species_{i+1}_list'], probability_sets[i])
                else:
                    p1, p2 = random.sample(species[f'species_{i+1}_list'], 2)

                if discrete == True:
                    child = combine_genes_discrete(p1, p2)
                    if curve_parents == False:
                        child2 = combine_genes_discrete(p1, p2)

                elif individual_cross == True:
                    child = combine_genes_individual_cross(p1, p2)
                    if curve_parents == False:
                        child2 = combine_genes_individual_cross(p1, p2)

                else:
                    child = combine_genes_cross(p1, p2)
                    if curve_parents == False:
                        child2 = combine_genes_cross(p1, p2)
                
                child = mutate_offspring(child)
                if curve_parents == False:
                    child2 = mutate_offspring(child2)

                species2[f'species_{i+1}_list'].append(child)
                new_pop.append(child)

                if curve_parents == False:
                    species2[f'species_{i+1}_list'].append(child2)  
                    new_pop.append(child2)

    return new_pop, species2

# Change the speciation threshold to keep in desired range
def dynamic_speciation(species_count):
    if species_count >= 10:
        globals() ['threshold'] += 0.025
    if species_count < 5:
        globals() ['threshold'] -= 0.025

# Separate a generation into species based on threshold
def speciate_population(threshold, pop):
    pop = np.flip(pop)
    representatives = [random.choice(pop)]
    species_count = 1
    species = {}
    species[f'species_{species_count}_list'] = []

    for individual in pop:
        found = False
        count = 0

        while found == False:

            # Check for individuals whose normalized average difference in parameters is below the threshold
            difference = np.average(np.abs(individual-representatives[count])/2)

            if difference < threshold:
                species[f'species_{count+1}_list'].append(individual)
                found = True

            count += 1

            # Add a new species if the current individual cannot be classified
            if count == species_count:
                species_count += 1
                representatives.append(individual)
                species[f'species_{species_count}_list'] = [individual]
                found = True

    # Modify the threshold for species difference to cultivate diversity
    if dynamic_speciation == True:
        dynamic_speciation(species_count)

    print(f'Thereshold: {np.round((threshold),3)}')

    return species_count, representatives, species

# Kill off a quarter of the population and replace them with highly mutated copies of the top 5 individuals
def reseed_event(pop, fit_scores):
    print("RESEED")
    indices = sorted(range(len(fit_scores)), key=lambda x: fit_scores[x], reverse=True)[:5]
    indices2 = sorted(range(len(fit_scores)), key=lambda x: fit_scores[x], reverse=True)[int(-len(pop)/4):]
    
    for count, unit in enumerate(pop):

        if count in indices2:
            parent = random.randint(0,4)
            geno_fidelity = random.random()

            for i in range(len(unit)):

                # Copy an allele or create a new random one
                if random.random() > geno_fidelity:
                    unit[i] = random.uniform(-1,1)
                else:
                    unit[i] = pop[indices[parent]][i]

    return pop

# Set up an envolutionary learning process for a single boss based on the number of generations
def training_run(mutagenic_temperature, mutation_intensity, mutation_reset, discrete, individual_cross, 
                 steady, comma, reseed_cycle, elitism, half, curve_parents, speciate, threshold, speciation_frequency):
    globals() ['mutagenic_temperature'] = mutagenic_temperature
    globals() ['mutation_intensity'] = mutation_intensity
    globals() ['speciation_frequency'] = speciation_frequency
    globals() ['individual_cross'] = individual_cross
    globals() ['mutation_reset'] = mutation_reset
    globals() ['curve_parents'] = curve_parents
    globals() ['reseed_cycle'] = reseed_cycle
    globals() ['threshold'] = threshold
    globals() ['discrete'] = discrete
    globals() ['speciate'] = speciate
    globals() ['elitism'] = elitism
    globals() ['half'] = half
    
    mean_stat = [0]
    peak_stat = [0]
    st_devs = [0]
    upper_avg_stat = [0]
    performance = [0]
    upper_avg = [0]  
    best = 0
    pop = initialize_population(pop_size, num_vars)

    # Loop through the selection and reproduction processes
    for g in range(generations):
        
        # Run simulations of the game with each individual 
        fit_scores, maxi, ind = test_population(pop)

         # Store statistics about this generation
        performance.append(np.round(maxi, decimals=2))
        upper_avg = np.mean(fit_scores[-10:])
        st_dev = np.std(fit_scores)

        if g == int(generations/2):
            globals() ['midpoint'] = True

        if upper_avg >= max(upper_avg_stat):
            best = pop[ind]

        upper_avg_stat.append(np.round(upper_avg, decimals=2))
        mean_stat.append(np.round(np.mean(fit_scores), decimals=2))
        peak_stat.append(np.round(max(fit_scores), decimals=2))
        st_devs.append(st_dev)

        if reseed_cycle == True and g%20 == 9:
            pop = reseed_event(pop, fit_scores)

        # Set up probabilities based on relative fitness
        pop, fit_scores = sort(pop, fit_scores)
        fit_scores_norm = species_normalize(fit_scores)
        probabilities = relative_prob(fit_scores_norm)
        
        # Run the chosen type of reproductions
        if steady == True:
            pop = reproduce_steady(pop, probabilities)
        elif comma == True:
            pop = reproduce_comma_strategy(pop, probabilities)
        elif speciate == True and g != 0:
            pop, species = reproduce_by_species(pop, g, species)
        elif speciate == True:
            pop, species = reproduce_by_species(pop, g)
        else:
            pop = reproduce_generational(pop, probabilities)

        print(f"Generation {g+1} top 10% avg: {np.round(upper_avg, decimals=2)}")

        if upper_avg >= 97:
            globals() ['early_stop'] = True
            print(performance)
            print(max(performance))
            return best, mean_stat, peak_stat, pop[ind], upper_avg_stat, st_devs

    print(performance)
    print(max(performance))
    
    return best, mean_stat, peak_stat, pop[ind], upper_avg_stat, st_devs

def train_set(string="V0", mutagenic_temperature=0.2, mutation_intensity=1, 
              mutation_reset=False, discrete=False, steady=False, comma=False, 
              reseed_cycle=False, elitism=False, half=False, curve_parents=False,
              speciate=False, threshold=False, speciation_frequency=False, 
              set=[1,2,3,4,5,6,7,8], runs=1):
    
    times = {enemy: [] for enemy in set}

    for j in set:
        for run in range(0, runs):
            global env
            env = Environment(experiment_name=experiment_name,
                        playermode="ai",
                        enemies=[j],
                        player_controller=player_controller(n_hidden_neurons),
                        speed="fastest",
                        enemymode="static",
                        level=2,
                        visuals=False)
            start_time = time.perf_counter_ns()
            best, mean_stat, peak_stat, last_best, upper_avg_stat, st_dev = training_run(mutagenic_temperature, mutation_intensity, 
                                                                                     mutation_reset, discrete, individual_cross, 
                                                                                     steady, comma, reseed_cycle, elitism, half, 
                                                                                     curve_parents, speciate, threshold, speciation_frequency)
            runtime = time.perf_counter_ns() - start_time
            times[j].append(runtime)
            
            np.savetxt(f"{stat_directory}_{string}/{string}_{j}_{run}_Means.csv", mean_stat, delimiter=",")
            np.savetxt(f"{stat_directory}_{string}/{string}_{j}_{run}_Peaks.csv", peak_stat, delimiter=",")
            np.savetxt(f"{stat_directory}_{string}/{string}_{j}_{run}_Upper_Avg.csv", upper_avg_stat, delimiter=",")
            np.savetxt(f"{stat_directory}_{string}/{string}_{j}_{run}_St_Dev.csv", st_dev, delimiter=",")
            np.savetxt(f"{stat_directory}_{string}/{string}_{j}_{run}_Params.csv", best, delimiter=",")

    with open(f"{stat_directory}_{string}/{string}_Runtimes.csv", 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['enemy', 'run', 'time'])
        for enemy in times:
            for run in range(0, len(times[enemy])):
                writer.writerow([enemy, run, times[enemy][run]])

# Test a set of NN parameters from the stored csv file
def test_params(string, attempts=1, set=[1,2,3,4,5,6,7,8], experiment_count=1):

    # Initialize the environment
    env = Environment(experiment_name=experiment_name,
				  playermode="ai",
                  enemies=[1],
				  player_controller=player_controller(n_hidden_neurons),
			  	  speed="normal",
				  enemymode="static",
				  level=2,
				  visuals=True)

    # Set up the environment params and game statistics
    for j in set:

        env.enemies = [j]
        performance = []
        avg_enemy_life = []
        avg_life = []
        avg_time = []

        # Test parameters from each separate experiment
        for run in range(experiment_count):

            scores = []
            life_left = []
            gametimes = []
            enemy_life = []
            best_params = pd.read_csv(f"{stat_directory}_{string}/{string}_{j}_{run}_Params.csv", delimiter=",", header=None)

            # Complete the number of runs for the boss and stores the statitics
            for _ in range(attempts):    
                fit, p_life, e_life, gametime = env.play(pcont=np.array(best_params.iloc[:,0]))
                print(f'Boss {j}, run {run}, fitness score {fit}')
                scores.append(fit)
                life_left.append(p_life)
                gametimes.append(gametime)
                enemy_life.append(e_life)

            performance.append(np.mean(np.array(scores)))
            avg_enemy_life.append(np.mean(np.array(enemy_life)))
            avg_life.append(np.mean(np.array(life_left)))
            avg_time.append(np.mean(np.array(gametimes)))

            np.savetxt(f"{stat_directory}_{string}/{string}_{j}_{run}_Fit_Scores.csv", scores, delimiter=",")
            np.savetxt(f"{stat_directory}_{string}/{string}_{j}_{run}_Enemy_Life.csv", enemy_life, delimiter=",")
            np.savetxt(f"{stat_directory}_{string}/{string}_{j}_{run}_Player_Life.csv", life_left, delimiter=",")
            np.savetxt(f"{stat_directory}_{string}/{string}_{j}_{run}_Gametime.csv", gametimes, delimiter=",")

        np.savetxt(f"{stat_directory}_{string}/{string}_{j}_avg_Fit_Scores.csv", performance, delimiter=",")
        np.savetxt(f"{stat_directory}_{string}/{string}_{j}_avg_Enemy_Life.csv", avg_enemy_life, delimiter=",")
        np.savetxt(f"{stat_directory}_{string}/{string}_{j}_avg_Player_Life.csv", avg_life, delimiter=",")
        np.savetxt(f"{stat_directory}_{string}/{string}_{j}_avg_Gametime.csv", avg_time, delimiter=",")

    return performance


if __name__ == '__main__':

    # Train New GAs
    #train_set(filename, elitism=5, half=False, mutagenic_temperature=0.1, curve_parents=True, discrete=False, reseed_cycle=False, set=[1,4,6], speciate=True, threshold=0.4, speciation_frequency=10, runs=10)
    train_set(filename, elitism=5, half=False, mutagenic_temperature=0.1, curve_parents=True, discrete=False, reseed_cycle=False, set=[1,4,6], runs=10)

    # Test the best parameters
    performance = test_params(filename, 2, set=[1,4,6], experiment_count=10)



