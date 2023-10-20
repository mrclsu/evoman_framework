import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
import matplotlib.patches as mpatches

# Define warm and cool color palettes
warm_palette = sns.color_palette("YlOrRd", 3)
cool_palette = sns.color_palette("Blues", 3)

# Define the enemy numbers and run numbers for NEAT and GA
enemies = [1, 2, 3, 4, 5, 6, 7, 8]
runs = 10

# Create data dictionaries for the 3 fitness functions
data1 = {}
data2 = {}
data3 = {}

# Load the data from the three CSV files
data1 = pd.read_csv('deap_specialist/box_plot/def_top_10.csv')
data2 = pd.read_csv('deap_specialist/box_plot/no_time_top_10.csv')
data3 = pd.read_csv('deap_specialist/box_plot/bonus_time_top_10.csv')

# Calculate the difference between player_life and enemy_life for each dataset
data1['life_diff'] = data1['player_life'] - data1['enemy_life']
data2['life_diff'] = data2['player_life'] - data2['enemy_life']
data3['life_diff'] = data3['player_life'] - data3['enemy_life']

# Combine the data into a single DataFrame
combined_data = pd.concat([data1['life_diff'], data2['life_diff'], data3['life_diff']], axis=1)
combined_data.columns = ['Fitness Function with Time Penalty', 'Fitness Function without Time Penalty', 'Fitness Function with Time Bonus']

# Create a box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=combined_data, palette=['#e4c1f9', '#a9def9', '#d0f4de'])
plt.title('Distribution of Max Fitness Across 10 Runs', fontsize='xx-large')
plt.ylabel('Energy Gain', fontsize='xx-large')
legend_handles = [mpatches.Patch(color='#e4c1f9', label='Fitness Function with Time Penalty'),
                  mpatches.Patch(color='#a9def9', label='Fitness Function without Time Penalty'),
                  mpatches.Patch(color='#d0f4de', label='Fitness Function with Time Bonus')]
plt.legend(handles=legend_handles, title='Fitness Functions', title_fontsize='xx-large', fontsize='large')
plt.show()



# # Load data for 1st fitness function with default fitness function
# for enemy in enemies:
#     energy_gains = []
#     for run in range(1, runs):
#         filename = f'deap_specialist/box_plot/def_top_10.csv'
#         df = pd.read_csv(filename)
        
#         energy_gain = df['player_life'] - df['enemy_life']
#         energy_gains.extend(energy_gain)
    
#     data1[f'Default Fitness Function'] = energy_gains

# # Load data for 1st fitness function with bonus time fitness function
# for enemy in enemies:
#     energy_gains = []
#     for run in range(1, runs):
#         filename = f'deap_specialist/box_plot/bonus_time_top_10.csv'
#         df = pd.read_csv(filename)
        
#         energy_gain = df['player_life'] - df['enemy_life']
#         energy_gains.extend(energy_gain)
    
#     data2[f'Default Fitness Function'] = energy_gains


# # Load data for 1st fitness function with no time fitness function
# for enemy in enemies:
#     energy_gains = []
#     for run in range(1, runs):
#         filename = f'deap_specialist/box_plot/no_time_top_10.csv'
#         df = pd.read_csv(filename)
        
#         energy_gain = df['player_life'] - df['enemy_life']
#         energy_gains.extend(energy_gain)
    
#     data3[f'Default Fitness Function'] = energy_gains



# # Create a boxplot with NEAT and GA data side by side
# plt.figure(figsize=(14, 6))

# # Plot NEAT box plots with warm colors
# neat_positions = np.arange(1, len(neat_data) + 1)
# bp_neat = plt.boxplot(neat_data.values(), positions=neat_positions, labels=neat_data.keys(), patch_artist=True)
# for box, color in zip(bp_neat['boxes'], warm_palette):
#     box.set_facecolor(color)

#     # Change line color for warm boxes
#     for line in bp_neat['medians']:
#         line.set(color='darkred', linewidth=1.5)
#     for line in bp_neat['whiskers']:
#         line.set(color='darkred', linewidth=1.5)

# # Plot GA box plots with cool colors
# ga_positions = np.arange(len(neat_data) + 1, len(neat_data) + len(ga_data) + 1)
# bp_ga = plt.boxplot(ga_data.values(), positions=ga_positions, labels=ga_data.keys(), patch_artist=True)
# for box, color in zip(bp_ga['boxes'], cool_palette):
#     box.set_facecolor(color)

#     # Change line color for cool boxes
#     for line in bp_ga['medians']:
#         line.set(color='darkblue', linewidth=1.5)
#     for line in bp_ga['whiskers']:
#         line.set(color='darkblue', linewidth=1.5)

# plt.xticks(rotation=45)
# plt.title('Distribution of Max Fitness (NEAT vs. GA) Across 10 Runs', fontsize='xx-large')
# plt.ylabel('Individual Gain', fontsize='xx-large')

# plt.tight_layout()
# plt.show()