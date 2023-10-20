import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb

def read_and_aggregate_files(ea_type, enemy_group):
    avg_mean_fitness = []
    avg_max_fitness = []
    std_mean_fitness = []
    std_max_fitness = []
    
    if ea_type == "EA1":
        prefix = "bonus_time"
    elif ea_type == "EA2":
        prefix = "def"
    else:
        prefix = "no_time"

    for gen in range(351):
        mean_fitness_values = []
        max_fitness_values = []
        
        for run_number in range(0, 10):
            filepath = f"../deap_specialist/350_gens"
            filename = f"{filepath}/{prefix}_{enemy_group}_{run_number}_statistics.csv"
            df = pd.read_csv(filename)
            
            mean_fitness_values.append(df.iloc[gen]['mean_fitness'])
            max_fitness_values.append(df.iloc[gen]['max_fitness'])
        
        avg_mean_fitness.append(np.mean(mean_fitness_values))
        avg_max_fitness.append(np.mean(max_fitness_values))
        std_mean_fitness.append(np.std(mean_fitness_values))
        std_max_fitness.append(np.std(max_fitness_values))

    return avg_mean_fitness, avg_max_fitness, std_mean_fitness, std_max_fitness

def plot_combined_results(ea_data, enemy_group):
    generations = list(range(0, len(ea_data['EA1']['avg_mean'])))

    if enemy_group == 0:
        enemies = '2,3,4,5,6,8'
    elif enemy_group == 1:
        enemies = '1,4,6,8'

    plt.figure(figsize=(10, 6))
    for ea_type, data in ea_data.items():
        color = {'EA1': 'b', 'EA2': 'g', 'EA3': 'r'}[ea_type]
        plt.fill_between(generations, np.subtract(data['avg_mean'], data['std_mean']), 
                         np.add(data['avg_mean'], data['std_mean']), color=color, alpha=0.1)
        plt.plot(generations, data['avg_mean'], '--', label=f'Mean Fitness {ea_type}', color=color)
        plt.fill_between(generations, np.subtract(data['avg_max'], data['std_max']), 
                         np.add(data['avg_max'], data['std_max']), color=color, alpha=0.1)
        plt.plot(generations, data['avg_max'], label=f'Max Fitness {ea_type}', color=color)

    plt.title(f'Enemy {enemies}', fontsize=20)
    plt.xlabel('Generations', fontsize=20)
    plt.ylabel('Fitness', fontsize=20)
    ax = plt.gca() 
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.xlim(0, 351)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.grid()
    plt.savefig(save_path, format='png', dpi=300) 
    plt.close()

for enemy_group in [0, 1]:
    ea_data = {}
    for ea_type in ["EA1", "EA2", "EA3"]:
        avg_mean, avg_max, std_mean, std_max = read_and_aggregate_files(ea_type, enemy_group)
        ea_data[ea_type] = {'avg_mean': avg_mean, 'avg_max': avg_max, 'std_mean': std_mean, 'std_max': std_max}
    
    if enemy_group == 0:
        enemies = [2,3,4,5,6,8]
    elif enemy_group == 1:
        enemies = [1,4,6,8]

    save_path = f"lineplot_{enemies}.png"
    plot_combined_results(ea_data, enemy_group)

