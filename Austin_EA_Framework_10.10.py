########################################
# An Evolutionary Algorithm Framework  #
# By Austin Dickerson                  #
# 10/9/2023                            #
########################################


# imports evoman and controller frameworks
from evoman.environment import Environment
from demo_controller import player_controller
from sklearn.cluster import KMeans
import os, random, time, csv, itertools
import numpy as np
import pandas as pd


# NN Parameters
n_hidden_neurons = 10
num_vars = n_hidden_neurons * (21) + (n_hidden_neurons + 1) * 5

# Reproducability
random.seed(579)
np.random.seed(135)
runs = 1
save = True

# Experiment state variables
pop = []

settings = {

# Game Parameters
'evoman': True,
'set': [5,6,8],
'generalist': True,
'multiple_mode': True,
'midpoint': False,

# Experiment Parameters
'pop_size': 500,
'generations': 20,
'early_stop': False,

# Mutation Parameters
'mutation_intensity': 0.1,
'mutagenic_temperature': 1,
'mutation_reset': False,

# Reproduction Parameters 
'discrete': False,
'individual_cross': False,
'crossover_line': False,
'curve_parents': True,

# Elitism Parameters
'elitism': 1,
'half': False,

# Optimization Parameters
'prioritize_life': False,
'prioritize_time': False,
'prioritize_gains': False,
'objective_switchpoint': False,

# Doomsday parameter
'reseed_cycle': False,


##### Reproduction Strategies ##### 
# reproduce_generational is default if all are false
'reproduce_steady': False,
'comma_strategy': False,
'evolutionary_programming': False,

# Non-traditional reproductive algorithms
'particle_swarm_optimization': False,
'pso_weights': [0.5, 0.5, 0.3],

'differential_evolution': False,
'crossover_rate': 0.9, 
'scaling_factor': 0.8,

# Speciation Paremeters
'speciate': False,
'dynamic_speciation': False,
'threshold': 0.35,
'num_kmeans_clusters': 8,
'speciation_frequency': 10,

}


# Name of the experiment
filename = 'GENERALIST_TEST'

experiment_name = 'controller_specialist'

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Name of the folder where statistics will be saved
stat_directory = 'Experiments_Evoman/EA_Statistics'

if not os.path.exists(f'{stat_directory}_{filename}'):
    os.makedirs(f'{stat_directory}_{filename}')


##### Population manipulating methods #####

# Create the randomized population
def initialize_population(pop_size, num_vars, var_range=1):
    pop = []

    for _ in range(pop_size):
        member = np.zeros(num_vars)

        for j in range(num_vars):
            member[j] = random.uniform(-var_range,var_range)

        pop.append(member)   

    return pop

# Initializes the population with additional parameters for sigma
def initialize_population_EP(pop_size, num_vars):
    pop = []
    perturbation_vectors = []

    for _ in range(pop_size):
        member = np.zeros(num_vars)
        perturbation = np.ones(num_vars)

        for j in range(num_vars):
            member[j] = random.uniform(-1,1)

        pop.append(member)   
        perturbation_vectors.append(perturbation)

    return pop, perturbation_vectors

# Run simulation for entire pop
def test_population(pop):
    scores = []
    lives = []
    gametimes = []
    gains = []
    pop = list(pop)

    for unit in pop:

        if settings['evoman']:
            fit, p_life, e_life, gametime = env.play(pcont=unit)
            gain = p_life - e_life

        scores.append(fit)
        lives.append(p_life)
        gains.append(gain)
        gametimes.append(gametime)

    # Base best parameters on the right metric
    if settings['prioritize_life']:
        maximum = max(lives)
        ind = lives.index(maximum)
    elif settings['prioritize_time']:
        maximum = max(gametimes)    
        ind = gametimes.index(maximum)
    elif settings['prioritize_gains']:
        maximum = max(gains)    
        ind = gains.index(maximum)
    else:
        maximum = max(scores)
        ind = scores.index(maximum)

    return scores, maximum, ind, lives, gametimes, gains


##### Statistics #####

# Normalize with lowest ~~0
def normalize(scores):
    scores = np.array(scores,float)
    lo = min(scores)
    hi = max(scores)
    scores -= lo
    scores /= (hi-lo)
    if np.any(np.isnan(scores)):
        print("There are NaN values in the normalized scores!")

    return scores

# Normalize on full range
def evo_normalize(scores):
    scores = np.array(scores)

    # Accommodate the greater range of fit scores during multi_mode (min varies by st. dev.)
    if settings['multiple_mode']:
        scores += 20
        scores /= 120
    else:
        scores += 10
        scores /= 110

    return scores

# Return as normalized probabilities
def relative_prob(fit_scores):
    probabilities = np.array(fit_scores)/np.array(fit_scores).sum()
    probabilities = np.nan_to_num(probabilities)
    probabilities = np.clip(probabilities, 0, 1)
    probabilities = probabilities / np.sum(probabilities)

    return probabilities

# List population in ascending order of fit score
def sort(pop, fit_scores):
    sorted_pairs = sorted(zip(pop, fit_scores), key=lambda x: x[1])
    pop, fit_scores = zip(*sorted_pairs)

    return pop, fit_scores


##### Selection #####

# Find 2 parents to reproduce
def select_parents(pop, low=None, high=None, probabilities=None):

    # Select based on proportion to fitness scores
    if settings['curve_parents']:
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


##### Recombination #####

# Returns one or two children from two parents
def reproduction(p1, p2):
    if settings['discrete']:
        child = combine_genes_discrete(p1, p2)

    elif settings['individual_cross']:
        child = combine_genes_individual_cross(p1, p2)

    else:
        child = combine_genes_cross(p1, p2)

    return child

# Take one allele from each parent with uniform probability
def combine_genes_discrete(p1, p2):
    child = p1

    for i in range(len(p1)):

        if random.random() > 0.5:
            child[i] = p2[i]

    return child

# Merge all alleles at a random split ratio
def combine_genes_cross(p1, p2):
    temp = np.random.random()

    if settings['crossover_line']:
        ind = int(len(p1) * temp)
        child = np.concatenate((p1[:ind], p2[ind:]))
    else:
        zygote1 = temp * p1
        zygote2 = (1 - temp) * p2
        child = zygote1 + zygote2

    return child

# Merge each allele individually
def combine_genes_individual_cross(p1, p2):
    temp = np.random.rand(num_vars)
    zygote1 = temp * p1
    zygote2 = (1 - temp) * p2
    child = zygote1 + zygote2

    return child


##### Mutation #####

# Alter some alleles in a new child
def mutate_offspring(child):
    for i in range(len(child)):

        if random.random() < settings['mutagenic_temperature']:

            # Either reset the allele or alter the existing
            if settings['mutation_reset']:
                child[i] = np.random.uniform(-1,1)
            else:
                child[i] += np.random.normal(0, settings['mutation_intensity'])
                child[i] = np.clip(child[i], -1, 1)

    return child

# Mutate based on Evolutionary Programming
def mutate_offspring_EP(child, perturbation_vector):

    # Update sigma parameters
    for i in range(num_vars):

        perturbation_vector[i] *= (0.2*np.random.normal(0, perturbation_vector[i])) + 1

        # Bound sigma at 0.1
        if perturbation_vector[i] < 0.1:
            perturbation_vector[i] = 0.1

        # Mutate NN parameters
        child[i] += np.random.normal(0, perturbation_vector[i])
        child[i] = np.clip(child[i], -1, 1)

    return child, perturbation_vector


##### Reproduction and Survival #####

# Combine selection, mutation and testing
def form_new_generation(pop, probabilities, gen_size):
    new_fit = []
    new_pop = []

    for _ in range(gen_size):
        p1, p2 = select_parents(pop, int(pop_size/2), len(pop), probabilities)

        child = reproduction(p1, p2)

        new_pop.append(mutate_offspring(child))

        if settings['evoman']:
            fit, p_life, e_life, gametime = env.play(pcont=child)
            gain = p_life - e_life
            
            # Add the metric being optimized
            if settings['prioritize_life']:
                new_fit.append(p_life)
            elif settings['prioritize_time']:
                new_fit.append(gametime)
            elif settings['prioritize_gains']:
                new_fit.append(gain)
            else:
                new_fit.append(fit)

    new_pop, new_fit = sort(new_pop, new_fit)

    if settings['prioritize_life']:
        probabilities = relative_prob(new_fit)
    else:
        new_fit_norm = normalize(new_fit)
        probabilities = relative_prob(new_fit_norm)

    new_pop = np.array(new_pop)
    new_fit = np.array(new_fit)

    return new_pop, new_fit, probabilities

# Reproduce with all parents replaced by offspring
def reproduce_generational(pop, probabilities, old_fit):
    pop = list(pop)
    new_pop = []
    new_fit = []

    # Select and replace all the parents with their offspring and update fit scores
    new_pop, new_fit, probabilities = form_new_generation(pop, probabilities, pop_size*2)
    
    # Copy the top performers or let them be replaced
    if type(settings['elitism']) == int and (settings['midpoint'] == True or settings['half'] == False):
        selected = np.random.choice(pop_size*2, pop_size-settings['elitism'], p=probabilities, replace=False)
        new_pop = new_pop[selected]
        new_fit = new_fit[selected]
        new_pop = np.concatenate((new_pop, pop[-settings['elitism']:]))
        new_fit = np.concatenate((new_fit, old_fit[-settings['elitism']:]))
    else:
        selected = np.random.choice(pop_size*2, pop_size, p=probabilities, replace=False)
        new_pop = new_pop[selected]
        new_fit = new_fit[selected]

    return new_pop, new_fit

# Replace only a small part of the population with each generation
def reproduce_steady_pop(pop, probabilities, old_fit):
    pop = list(pop)
    new_pop = []
    new_fit = []
    next_gen = int(pop_size/5)

    new_pop, new_fit, probabilities = form_new_generation(pop, probabilities, next_gen*2)

    selected = np.random.choice(next_gen*2, next_gen, p=probabilities,
                                replace=False)
    
    new_pop = new_pop[selected]
    new_fit = new_fit[selected]
    new_pop = np.concatenate((new_pop, pop[-(pop_size-next_gen):]))
    new_fit = np.concatenate((new_fit, old_fit[-(pop_size-next_gen):]))

    return new_pop, new_fit

# Create a large new generation and keep the top performers only
def reproduce_comma_strategy(pop, probabilities, old_fit):
    pop = list(pop)
    new_pop = []
    new_fit = []
    next_gen = pop_size * 7
    elitism = settings['elitism']

    new_pop, new_fit, probabilities = form_new_generation(pop, probabilities, next_gen)

    # Handle elitism parameters is any
    if type(elitism) == int and (settings['midpoint'] or not settings['half']):
        new_pop = new_pop[-(pop_size - elitism):]
        new_fit = new_fit[-(pop_size - elitism):]
        new_pop = np.concatenate((new_pop, [pop[-elitism]]))
        new_fit = np.concatenate((new_fit, [old_fit[-elitism]]))
    else:
        new_pop = new_pop[-pop_size:]
        new_fit = new_fit[-pop_size:]

    return new_pop, new_fit

# Reproduce in the EP setup where individual parents are mutated
def reproduce_EP(pop, perturbation_vectors):
    pop = np.array(pop)
    perturbation_vectors = np.array(perturbation_vectors)
    new_pop = []
    new_vectors = []

    for i in range(pop_size):
        child, new_perturbation_vector = mutate_offspring_EP(pop[i], perturbation_vectors[i])
        new_pop.append(child)
        new_vectors.append(new_perturbation_vector)

    new_pop = np.array(new_pop)
    new_vectors = np.array(new_vectors)
    pop = np.vstack((pop,new_pop))
    perturbation_vectors = np.vstack((perturbation_vectors,new_vectors))

    scores, maxi, ind, lives, gametimes, gains = test_population(pop)

    if settings['prioritize_life']:
        scores2 = normalize(lives)
        scores = lives
    elif settings['prioritize_time']:
        scores2 = normalize(gametimes)
        scores = gametimes
    elif settings['prioritize_gains']:
        scores2 = normalize(gains)
        scores = gains
    else:
        scores2 = evo_normalize(scores)
        
    probabilities = relative_prob(scores2)

    # Survival based probabilistically on relative fitness
    selected = np.random.choice(pop_size*2, pop_size, p=probabilities, replace=False)
    pop = pop[selected]
    perturbation_vectors = perturbation_vectors[selected]
    scores = np.array(scores)[selected]

    return pop, perturbation_vectors, scores

# Reproduce with the DE strategy
def reproduce_differential_evolution(pop, fit_scores):
    pop = np.array(pop)
    fit_scores = np.array(fit_scores)
    mutation_vectors = np.zeros((pop_size, num_vars))
    advantage = 0

    for i in range(pop_size):

        # Assign perturbation vector
        choices = [item for j, item in enumerate(pop) if j != i]
        Ai, Bi, Ci = random.sample(choices, 3)

        mutation_vectors[i] = Ai + settings['scaling_factor'] * (Bi - Ci)

        # Create child Ui
        for j in range(num_vars):
            if random.random() > settings['crossover_rate']:
                mutation_vectors[i,j] = pop[i,j]

        # Tournament between progenitor and offspring for survival
        if settings['evoman']:
            fit, p_life, e_life, gametime = env.play(pcont=mutation_vectors[i])
            gain = p_life - e_life

            if settings['prioritize_life']:
                fit = p_life
            elif settings['prioritize_time']:
                fit = gametime
            elif settings['prioritize_gains']:
                fit = gain

        if fit > fit_scores[i]:
            pop[i] = mutation_vectors[i]
            fit_scores[i] = fit
            advantage += 1

    # Tracks effectiveness of mutations
    print(f'{np.round(((advantage/pop_size)*100), 2)} percent of mutations conferred advantage')

    return pop, fit_scores

# Modify the velocity and positioan of particles in a PSO algorithm
def vector_shift_PSO(pop, velocity_vectors, previous_bests, previous_scores, global_best, best_score):
    pop = np.array(pop)
    velocity_vectors = np.array(velocity_vectors)
    pso_weights = settings['pso_weights']

    # Reset the score trackers to a different scale
    if not settings['prioritize_time'] and best_score > 100:
        best_score = -50
        velocity_vectors = initialize_population(pop_size, num_vars, 0.2)
        previous_scores[:] = -50

    for i in range(pop_size):
        prev_velocity = velocity_vectors[i]

        # Calculate new particle 
        velocity_vectors[i] = (pso_weights[0] * velocity_vectors[i]) + (pso_weights[1] * random.random() * 
                            (previous_bests[i] - pop[i])) + (pso_weights[2] * random.random() * (global_best - pop[i]))
        
        # Update and test new position
        pop[i] = pop[i] + prev_velocity

        for j in range(num_vars):
            np.clip(pop[i,j],-1,1)

        if settings['evoman']:
            score, p_life, e_life, gametime = env.play(pcont=pop[i])
            gain = p_life - e_life
            
            if settings['prioritize_life']:
                score = p_life
            elif settings['prioritize_time']:
                score = gametime
            elif settings['prioritize_gains']:
                score = gain
            
        if score > best_score:
            global_best = pop[i]
            best_score = score
        if score > previous_scores[i]:
            previous_bests[i] = pop[i]
            previous_scores[i] = score

    print(f'Average velocity per variable is {np.average(np.abs(velocity_vectors))}')

    return  pop, velocity_vectors, previous_bests, previous_scores, global_best, best_score 

# Reproduce within subgroups of individuals only
def reproduce_by_species(pop, generation, species_count=0, species=None):

    # Speciate based on frequency
    if generation == 0:
        species_count, _, species = speciate_population(settings['threshold'], pop)
    elif settings['speciation_frequency'] != None and (generation % settings['speciation_frequency']) == 0:
        species_count, _, species = speciate_population(settings['threshold'], pop)

    # Initialize test statistics
    species_max = []
    species_avg = []
    new_pop = []
    norm_avg = []
    count_each_species = []
    probability_sets = []
    species_scores = np.array([])

    # Test each species with viable size
    for i in range(species_count):

        if len(species[f'species_{i+1}_list']) != 0:
            scores, maximum, p_life, e_life, gametime = test_population(species[f'species_{i+1}_list'])
            gain = p_life - e_life
            probability_sets.append(relative_prob(scores))
            scores2 = evo_normalize(scores)
            norm_avg.append(np.average(scores2))
            species_max.append(maximum)
            species_avg.append(np.average(scores))
            species_scores = np.append(species_scores, scores)
        else:
            species_count -= 1

    for k in range(species_count):
        count_each_species.append(len(species[f'species_{k+1}_list']))

    print(f'Species Counts: {count_each_species}')
    print(f'Species Maximums: {np.round(species_max,2)}')
    print(f'Species Averages: {np.round(species_avg,2)}')

    # Allocate next generation spaces among species
    probabilities = relative_prob(norm_avg)
    ratios = pop_size * probabilities
    ratios = [round(num) for num in ratios]
    approved = np.ones(species_count)
    reallocation = 0

    # Sift out species with too few to reproduce
    for i in range(species_count):

        if count_each_species[i] < 4:
                reallocation += ratios[i]
                approved[i] = 0

    # Renormalize allocations
    probabilities = probabilities * approved
    probabilities = probabilities / probabilities.sum()
    ratios1 = ratios * approved
    ratios2 = reallocation * probabilities + ratios1
    ratios3 = [round(num) for num in ratios2]

    old_pop_ratios = np.array(count_each_species) * approved
    old_pop_ratios = (old_pop_ratios / old_pop_ratios.sum()) * pop_size
    old_pop_ratios = [round(num) for num in old_pop_ratios]
    ratios = (np.array(ratios3) + np.array(old_pop_ratios)) / 2
    ratios = [round(num) for num in ratios]

    print(f'Old allocation: {old_pop_ratios}')
    print(f'After smoothing, species are now allocated: {ratios}')
    print(f'Totaling: {sum(ratios)}')

    species2 = {}

    for i in range(species_count):

        # Initialize the new generation for each species
        species2[f'species_{i+1}_list'] = []
        
        if count_each_species[i] >= 4:

            for _ in range(ratios[i]):

                # Selection, reproduction, and mutation
                p1, p2 = select_parents(species[f'species_{i+1}_list'], probability_sets[i])
                child = reproduction(p1, p2)
                child = mutate_offspring(child)

                species2[f'species_{i+1}_list'].append(child)
                new_pop.append(child)

    return new_pop, species2, species_scores, species_count


##### Speciation #####

# Change the speciation threshold to keep in desired range
def dynamic_speciation(species_count):
    if species_count >= 10:
        settings['threshold'] += 0.005
    if species_count < 5:
        settings['threshold'] -= 0.005

    threshold = settings['threshold']
    print(f'Threshold: {np.round(threshold, 3)}')

# Separate a generation into species based on threshold
def speciate_population(species_threshold, pop):
    pop = np.flip(pop)
    representatives = [random.choice(pop)]
    species = {}
    
    # Standard speciation practice with a dynamic threshold
    if settings['dynamic_speciation']:
        species_count = 1
        species[f'species_{species_count}_list'] = []

        for individual in pop:
            found = False
            count = 0

            while not found:

                # Check for individuals whose normalized average difference in parameters is below the threshold
                difference = np.average(np.abs(individual - representatives[count]) / 2)

                if difference < species_threshold:
                    species[f'species_{count+1}_list'].append(individual)
                    found = True

                count += 1

                # Add a new species if the current individual cannot be classified
                if count == species_count:
                    species_count += 1
                    representatives.append(individual)
                    species[f'species_{species_count}_list'] = [individual]
                    found = True

        # Count the number of viable species
        true_count = 0
        for i in range(species_count):
            if len(species[f'species_{i+1}_list']) >= 4:
                true_count += 1

        # Modify the threshold for species difference to maintain diversity
        if true_count < 5 or true_count > 10:
            dynamic_speciation(true_count)
            species_count, representatives, species = speciate_population(settings['threshold'], pop)

    # Speciate using KMeans clustering
    else:
        species_count = settings['num_kmeans_clusters']
        kmeans = KMeans(n_clusters=species_count, random_state=0, n_init="auto")
        kmeans.fit(pop)

        for i in range(species_count):
            species[f'species_{i+1}_list'] = []

        for j in range(len(pop)):
            species[f'species_{kmeans.labels_[j]+1}_list'].append(pop[j])

    return species_count, representatives, species


##### Doomsday #####

# Kill off a quarter of the population and replace them with highly mutated copies of the top 5 individuals
def reseed_event(pop, fit_scores):
    print("RESEED")
    indices = sorted(range(len(fit_scores)), key=lambda x: fit_scores[x], reverse=True)[:5]
    indices2 = sorted(range(len(fit_scores)), key=lambda x: fit_scores[x], reverse=True)[int(-len(pop) / 4):]
    
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


##### Evolutionary Algorithm for Neural Network Parameters #####

# Set up an envolutionary learning process based on the style of EA and the number of generations
def training_run(pop):
    mean_stat = [0]
    st_devs = [0]
    upper_avg_stat = [0]
    performance = [0]
    upper_avg = [0]  
    best = 0

    # Initialize the population in a manner fitting the EA approach
    if len(pop) == 0 and not settings['evolutionary_programming'] and not settings['particle_swarm_optimization']:
        pop = initialize_population(pop_size, num_vars)

    elif settings['evolutionary_programming']:
        pop, perturbation_vectors = initialize_population_EP(pop_size, num_vars)

    elif settings['particle_swarm_optimization']:
        if len(pop) == 0:
            pop = initialize_population(pop_size, num_vars)

        velocity_vectors = initialize_population(pop_size, num_vars, 0.2)
        previous_bests = np.zeros((pop_size, num_vars))
        previous_scores = np.zeros(pop_size)
        global_best = np.zeros((1, num_vars))
        best_score = 0

    # Loop through the selection and reproduction processes
    for g in range(settings['generations']):

        if settings['evoman']:

            if settings['objective_switchpoint'] != False and settings['objective_switchpoint'] == g:
                print('switch')
                settings['prioritize_life'] = False
                settings['prioritize_time'] = False

            if g == 0:
                # Run simulation of the game with each individual 
                fit_scores, maxi, ind, lives, gametimes, gains = test_population(pop)
                
                if settings['prioritize_life']:
                    fit_scores = lives
                elif settings['prioritize_time']:
                    fit_scores = gametimes
                elif settings['prioritize_gains']:
                    fit_scores = gains

            maxi = max(fit_scores)
            ind = list(fit_scores).index(maxi)
            
        # Keep track of the best set of parameters so far
        if np.round(maxi, decimals=2) > max(performance) or (type(settings['objective_switchpoint']) == int and settings['objective_switchpoint'] <= g 
                                                             and np.round(maxi, decimals=2) > max(performance[settings['objective_switchpoint']:])):
            best = pop[ind]
            discovery_gen = g

        # Sort the population by their scores
        pop, fit_scores = sort(pop, fit_scores)

        # Store statistics about this generation
        performance.append(np.round(maxi, decimals=2))
        
        upper_avg = np.mean(fit_scores[-int(pop_size / 10):])
        st_dev = np.std(fit_scores)

        upper_avg_stat.append(np.round(upper_avg, decimals=2))
        mean_stat.append(np.round(np.mean(fit_scores), decimals=2))
        st_devs.append(st_dev)

        if g == int(settings['generations'] / 2):
            settings['midpoint'] = True

        if settings['reseed_cycle'] and g % 20 == 10:
            pop = reseed_event(pop, fit_scores)

        # Set up probabilities based on relative fitness
        if settings['prioritize_life']:
            fit_scores_norm = fit_scores
        elif settings['prioritize_time']:
            fit_scores_norm = normalize(np.array(fit_scores))
        else:
            fit_scores_norm = evo_normalize(fit_scores)

        probabilities = relative_prob(fit_scores_norm)
        
        # Run the chosen type of reproduction
        if settings['reproduce_steady']: 
            pop, fit_scores = reproduce_steady_pop(pop, probabilities, fit_scores)

        elif settings['comma_strategy']: 
            pop, fit_scores = reproduce_comma_strategy(pop, probabilities, fit_scores)

        elif settings['evolutionary_programming']: 
            pop, perturbation_vectors, fit_scores = reproduce_EP(pop, perturbation_vectors)

        elif settings['differential_evolution']: 
            pop, fit_scores = reproduce_differential_evolution(pop, fit_scores)

        elif settings['particle_swarm_optimization']: 
            pop, velocity_vectors, previous_bests, fit_scores, global_best, best_score = vector_shift_PSO(pop, velocity_vectors, previous_bests, 
                                                                                                            previous_scores, global_best, best_score)
        
        elif settings['speciate'] and g == 0: 
            pop, species, fit_scores, species_count = reproduce_by_species(pop, g)

        elif settings['speciate']: 
            pop, species, fit_scores, species_count = reproduce_by_species(pop, g, species_count, species)

        else: 
            pop, fit_scores = reproduce_generational(pop, probabilities, fit_scores)


        print(f"Generation {g+1} top 10% avg: {np.round(upper_avg, decimals=2)}, Full avg: {np.round(np.mean(fit_scores), decimals=2)}")

        # Stop the experiment if performance has crossed threshold
        if upper_avg >= 97 and not settings['prioritize_time'] and not settings['objective_switchpoint']:
            settings['early_stop'] = True
            print(performance)
            print(max(performance))
            return best, discovery_gen, mean_stat, performance, upper_avg_stat, st_devs, pop

    print(performance)
    print(f"Best Discovered in Generation {discovery_gen+1}")
    pop, fit_scores = sort(pop, fit_scores)
    
    return best, discovery_gen, mean_stat, performance, upper_avg_stat, st_devs, pop


##### Evoman EA Training #####

# Train a specilist or generalist agent through the EA framework against the selected bosses all experiment parameters 
# are specified as parameters
def evoman_train_set(string, pop, runs, save, **kwargs):

    for key, value in kwargs.items():
        if key in settings:
            settings[key] = value
        
        else:
            raise ValueError(f"Unknown setting: {key}")
        
    global pop_size
    pop_size = settings['pop_size']
    
    times = {enemy: [] for enemy in settings['set']}
    settings['last_boss'] = settings['set'][-1]

    if settings['multiple_mode']:
        enemies = [0]
    else:
        enemies = settings['set']
    
    # Iterate through each boss
    for j in enemies:

        # Iterate through each experiment run
        for run in range(0, runs):

            # Set up environment and complete one training run
            if settings['multiple_mode']:
                env.enemies = settings['set']
                env.multiplemode = "yes"

            best, discovery_gen, mean_stat, performance, upper_avg_stat, st_dev, pop = training_run(pop)

            # Test the winning parameters
            print(f'Test score is {env.play(pcont=best)}')
            
            # Save all the important stats in the specified folder
            if save:
                np.savetxt(f"{stat_directory}_{string}/{string}_{j}_{run}_Means.csv", mean_stat, delimiter=",")
                np.savetxt(f"{stat_directory}_{string}/{string}_{j}_{run}_Peaks.csv", performance, delimiter=",")
                np.savetxt(f"{stat_directory}_{string}/{string}_{j}_{run}_Upper_Avg.csv", upper_avg_stat, delimiter=",")
                np.savetxt(f"{stat_directory}_{string}/{string}_{j}_{run}_St_Dev.csv", st_dev, delimiter=",")
                np.savetxt(f"{stat_directory}_{string}/{string}_{j}_{run}_Params.csv", best, delimiter=",")

    # Save runtimes for computational comparison
    if save:
        with open(f"{stat_directory}_{string}/{string}_Runtimes.csv", 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['enemy', 'run', 'time'])

            for enemy in times:

                for run in range(0, len(times[enemy])):
                    writer.writerow([enemy, run, times[enemy][run]])

    return pop

# Test a set of NN parameters from the stored csv file
def evoman_test_params(string, attempts=1, set=[1,2,3,4,5,6,7,8], experiment_count=1, save=False, visuals=False):

    set_of_performances = []
    set_of_life = []
    set_of_gains = []

    # Set up the environment params and game statistics
    for j in set:

        # switch off for testing
        if settings['multiple_mode']:
            env.multiplemode = "no"

        if visuals:
            env.visuals = True
            env.speed = "normal"

        env.enemies = [j]
        performance = []
        avg_enemy_life = []
        avg_life = []
        avg_gains = []
        avg_time = []

        # Test parameters from each separate experiment
        for run in range(experiment_count):

            scores = []
            life_left = []
            gametimes = []
            enemy_life = []
            gains = []
            win_count = 0

            # Get the right parameters
            if settings['generalist']:
                if settings['multiple_mode']:
                    best_params = pd.read_csv(f"{stat_directory}_{string}/{string}_{0}_{run}_Params.csv", delimiter=",", header=None)
                else:
                    best_params = pd.read_csv(f"{stat_directory}_{string}/{string}_{settings['last_boss']}_{run}_Params.csv", delimiter=",", header=None)
            else:    
                best_params = pd.read_csv(f"{stat_directory}_{string}/{string}_{j}_{run}_Params.csv", delimiter=",", header=None)
            
            # Shed sigma values for cases with variation coded into the genetics
            params = best_params.iloc[:num_vars,0]

            # Complete the number of runs for the boss and store the statitics
            for _ in range(attempts):   
                
                fit, p_life, e_life, gametime = env.play(pcont=np.array(params))
                gain = p_life - e_life
                
                if e_life == 0:
                    win_count += 1

                print(f'Boss {j}, run {np.round(run,2)}, fitness score {np.round(fit,2)}, player life {np.round(p_life,2)}, enemy life {np.round(e_life,2)}, runtime {gametime}')
                
                scores.append(fit)
                life_left.append(p_life)
                gametimes.append(gametime)
                gains.append(gain)
                enemy_life.append(e_life)

            performance.append(np.mean(np.array(scores)))
            avg_enemy_life.append(np.mean(np.array(enemy_life)))
            avg_life.append(np.mean(np.array(life_left)))
            avg_gains.append(np.mean(np.array(gains)))
            avg_time.append(np.mean(np.array(gametimes)))

            # Save stats from individual experiments
            if save:
                np.savetxt(f"{stat_directory}_{string}/{string}_{j}_{run}_Fit_Scores.csv", scores, delimiter=",")
                np.savetxt(f"{stat_directory}_{string}/{string}_{j}_{run}_Enemy_Life.csv", enemy_life, delimiter=",")
                np.savetxt(f"{stat_directory}_{string}/{string}_{j}_{run}_Player_Life.csv", life_left, delimiter=",")
                np.savetxt(f"{stat_directory}_{string}/{string}_{j}_{run}_Gains.csv", gains, delimiter=",")
                np.savetxt(f"{stat_directory}_{string}/{string}_{j}_{run}_Gametime.csv", gametimes, delimiter=",")


        # Save stats from averages of experiments
        if save:
            np.savetxt(f"{stat_directory}_{string}/{string}_{j}_avg_Fit_Scores.csv", performance, delimiter=",")
            np.savetxt(f"{stat_directory}_{string}/{string}_{j}_avg_Enemy_Life.csv", avg_enemy_life, delimiter=",")
            np.savetxt(f"{stat_directory}_{string}/{string}_{j}_avg_Player_Life.csv", avg_life, delimiter=",")
            np.savetxt(f"{stat_directory}_{string}/{string}_{j}_avg_Gains.csv", avg_gains, delimiter=",")
            np.savetxt(f"{stat_directory}_{string}/{string}_{j}_avg_Gametime.csv", avg_time, delimiter=",")

        set_of_performances.append(np.round(np.mean(performance),2))
        set_of_life.append(np.round(np.mean(avg_life),2))
        set_of_gains.append(np.round(np.mean(avg_gains),2))

    print(f'Average of Fit Scores: {np.round(np.mean(set_of_performances),2)} and Gains: {np.round(np.mean(set_of_gains),2)}')

    return set_of_performances, set_of_life, set_of_gains

env = Environment(experiment_name=experiment_name,
                            playermode="ai",
                            enemies=[1],
                            player_controller=player_controller(n_hidden_neurons),
                            speed="fastest",
                            enemymode="static",
                            level=2,
                            visuals=False)


#############################################################################################################
# Starter culture

populations = {}

for i in range(1,9):

    env.enemies = [i]
    populations[f'pop_{i}'] = evoman_train_set(filename, pop, runs, save=False, set=[i], pop_size=100, generations=20, curve_parents=True, elitism=1)

pop = np.array(populations[f'pop_{1}'])

for j in range(2,9):
    pop = np.vstack((pop,np.array(populations[f'pop_{j}'])))

print(pop.shape)
np.savetxt(f"{stat_directory}_{filename}/Starter_Culture.csv", pop, delimiter=",")

##############################################################################################################


# Train against bosses [5-8]
filename = f'GENERALIST_8_POP_V1_[5,6,7,8]'
print(filename)

if not os.path.exists(f'{stat_directory}_{filename}'):
    os.makedirs(f'{stat_directory}_{filename}')

pop1 = pd.read_csv(f"{stat_directory}_GENERALIST_8_POPV1/Starter_Culture.csv", delimiter=",", header=None)
new_pop = []
for i in range(4,8):
    new_pop.append(pop1.iloc[(i*100)+80:(i*100)+100])

pop1 = np.vstack(new_pop)

# Train new algorithms
starter_pop1 = evoman_train_set(filename, pop1, runs, save=True, pop_size=500, generalist=True, set=[5,6,7,8], individual_cross=True, differential_evolution=False, multiple_mode=True, generations=30)
starter_pop1 = np.array(starter_pop1)
np.savetxt(f"{stat_directory}_GENERALIST_8_POPV1/Starter_Culture_Backup_[5-8].csv", starter_pop1, delimiter=",")


# Test the best parameters
performance, avg_life, avg_gain = evoman_test_params(filename, 1, set=[1,2,3,4,5,6,7,8], experiment_count=1, save=True)



# Train against bosses [1-4]
starter_pop1 = pd.read_csv(f"{stat_directory}_GENERALIST_8_POPV1/Starter_Culture_Backup_[5-8].csv", delimiter=",", header=None)
starter_pop1 = np.array(starter_pop1)
filename = f'GENERALIST_8_POP_V1_[1,2,3,4]'
print(filename)

if not os.path.exists(f'{stat_directory}_{filename}'):
    os.makedirs(f'{stat_directory}_{filename}')

pop2 = pd.read_csv(f"{stat_directory}_GENERALIST_8_POPV1/Starter_Culture.csv", delimiter=",", header=None)
new_pop = []
for i in range(0,4):
    new_pop.append(pop2.iloc[(i*100)+80:(i*100)+100])

pop2 = np.vstack(new_pop)

# Train new algorithms
starter_pop2 = evoman_train_set(filename, pop2, runs, save=True, set=[1,2,3,4], generalist=True, individual_cross=True, multiple_mode=True, generations=30)
starter_pop2 = np.array(starter_pop2)
np.savetxt(f"{stat_directory}_GENERALIST_8_POPV1/Starter_Culture_Backup_[1-4].csv", starter_pop2, delimiter=",")

# Test the best parameters
performance, avg_life, avg_gain = evoman_test_params(filename, 1, set=[1,2,3,4,5,6,7,8], experiment_count=1, save=True)


# Train against all bosses
full_starter = np.vstack((starter_pop1, starter_pop2))
np.savetxt(f"{stat_directory}_GENERALIST_8_POPV1/Starter_Culture_NEW.csv", full_starter, delimiter=",")

population = pd.read_csv(f"{stat_directory}_GENERALIST_8_POPV1/Starter_Culture_NEW.csv", delimiter=",", header=None)
population = np.array(population)

filename = f'GENERALIST_FULL_STARTER_TEST1'
print(filename)

if not os.path.exists(f'{stat_directory}_{filename}'):
    os.makedirs(f'{stat_directory}_{filename}')

final_pop = evoman_train_set(filename, population, runs, save=True, set=[1,2,3,4,5,6,7,8], pop_size=1000, generalist=True, differential_evolution=True, multiple_mode=True, generations=50)
final_pop = np.array(final_pop)
#np.savetxt(f"{stat_directory}_GENERALIST_8_POPV1/Starter_Culture_FINALPOP.csv", final_pop, delimiter=",")

# Test the best parameters
performance, avg_life, avg_gain = evoman_test_params(filename, 1, set=[1,2,3,4,5,6,7,8], experiment_count=1, save=True, visuals=True)

