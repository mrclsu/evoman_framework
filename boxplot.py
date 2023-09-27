import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the enemy numbers and run numbers
enemy_numbers = [1, 4, 6]
run_numbers = list(range(10))

# Create an empty list to store mean individual gain data for each enemy number
mean_individual_gain_data = []

# Load data for each enemy number and run number and calculate the mean
for enemy_number in enemy_numbers:
    enemy_mean_gain_values = []  # Store mean individual gain values for the current enemy number
    for run_number in run_numbers:
        filename = f"GA_Statistics_V22.1_Test/V22.1_Test_{enemy_number}_{run_number}_Player_Life.csv"
        try:
            with open(filename, 'r') as file:
                lines = file.readlines()
                individual_gain_values = [float(line.strip()) for line in lines]
                enemy_mean_gain_values.extend(individual_gain_values)
        except FileNotFoundError:
            # Skip missing files
            pass
    # Calculate the mean of individual gain values for this enemy number
    mean_individual_gain_data.append(enemy_mean_gain_values)

# Create a boxplot with all enemy numbers on one graph
plt.figure(figsize=(10, 6))
plt.boxplot(mean_individual_gain_data, labels=[f'Enemy {enemy}' for enemy in enemy_numbers])
plt.xlabel('Enemy Number')
plt.ylabel('Individual Gain (Mean)')
plt.title('Boxplots of Mean Individual Gain for Different Enemy Numbers')
plt.show()
