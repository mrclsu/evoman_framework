import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # Import seaborn for color palettes

# Define warm and cool color palettes
warm_palette = sns.color_palette("YlOrRd", 3)
cool_palette = sns.color_palette("Blues", 3)

# Define the enemy numbers and run numbers for NEAT and GA
neat_enemies = [1, 4, 6]
ga_enemies = [1, 4, 6]
runs = 10

# Create data dictionaries for NEAT and GA
neat_data = {}
ga_data = {}

# Load data for NEAT
for enemy in neat_enemies:
    mean_fitness_list = []
    for run in range(1, runs):
        filename = f'neat-controller/stats/winners-{enemy}.csv'
        df = pd.read_csv(filename)
        
        # Get the max fitness of the last generation (or use max() if it's not the last row)
        for gain in df['individual_gain']:
            mean_fitness_list.append(gain)
    
    neat_data[f'NEAT | Enemy {enemy}'] = mean_fitness_list

# Load data for GA
for enemy_number in ga_enemies:
    enemy_mean_gain_values = []  # Store mean individual gain values for the current enemy number
    for run_number in range(runs):
        filename_player = f"GA_Statistics_V22.1_Test/V22.1_Test_{enemy_number}_{run_number}_Player_Life.csv"
        filename_enemy = f"GA_Statistics_V22.1_Test/V22.1_Test_{enemy_number}_{run_number}_Enemy_Life.csv"
        
        try:
            with open(filename_player, 'r') as file_player, open(filename_enemy, 'r') as file_enemy:
                player_lines = file_player.readlines()
                enemy_lines = file_enemy.readlines()
                # Calculate individual gain as the difference between player life and enemy life
                individual_gain_values = [float(player.strip()) - float(enemy.strip()) for player, enemy in zip(player_lines, enemy_lines)]
                enemy_mean_gain_values.extend(individual_gain_values)
        except FileNotFoundError:
            # Skip missing files
            pass
    # Calculate the mean of individual gain values for this enemy number
    ga_data[f'GA | Enemy {enemy_number}'] = enemy_mean_gain_values

# Create a boxplot with NEAT and GA data side by side
plt.figure(figsize=(14, 6))

# Plot NEAT box plots with warm colors
neat_positions = np.arange(1, len(neat_data) + 1)
bp_neat = plt.boxplot(neat_data.values(), positions=neat_positions, labels=neat_data.keys(), patch_artist=True)
for box, color in zip(bp_neat['boxes'], warm_palette):
    box.set_facecolor(color)

    # Change line color for warm boxes
    for line in bp_neat['medians']:
        line.set(color='darkred', linewidth=1.5)
    for line in bp_neat['whiskers']:
        line.set(color='darkred', linewidth=1.5)

# Plot GA box plots with cool colors
ga_positions = np.arange(len(neat_data) + 1, len(neat_data) + len(ga_data) + 1)
bp_ga = plt.boxplot(ga_data.values(), positions=ga_positions, labels=ga_data.keys(), patch_artist=True)
for box, color in zip(bp_ga['boxes'], cool_palette):
    box.set_facecolor(color)

    # Change line color for cool boxes
    for line in bp_ga['medians']:
        line.set(color='darkblue', linewidth=1.5)
    for line in bp_ga['whiskers']:
        line.set(color='darkblue', linewidth=1.5)

plt.xticks(rotation=45)
plt.title('Distribution of Max Fitness (NEAT vs. GA) Across 10 Runs', fontsize='xx-large')
plt.ylabel('Individual Gain', fontsize='xx-large')

plt.tight_layout()
plt.show()
