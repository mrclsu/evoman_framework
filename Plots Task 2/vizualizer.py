import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_data_seaborn(df, title, ax, linestyle, color):
    generations = df.index
    
    # Plot mean_fitness and its standard deviation
    ax.plot(generations, df['mean_fitness'], label=f'Mean Fitness {title}', 
            linestyle=linestyle, color=color)
    ax.fill_between(generations, 
                    df['mean_fitness'] - df['std_mean_fitness'], 
                    df['mean_fitness'] + df['std_mean_fitness'], 
                    alpha=0.3, color=color)
    
    # Plot max_fitness and its standard deviation
    ax.plot(generations, df['max_fitness'], label=f'Max Fitness {title}', 
            linestyle=linestyle, color=color, alpha=0.5)
    ax.fill_between(generations, 
                    df['max_fitness'] - df['std_max_fitness'], 
                    df['max_fitness'] + df['std_max_fitness'], 
                    alpha=0.2, color=color)
    
    # Set plot title, labels
    ax.set_title('Fitness Across Generations by the Generalist EAs')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')

# Read the final CSV files
final_bonus_time = pd.read_csv('final_bonus_time.csv')
final_no_time = pd.read_csv('final_no_time.csv')
final_def = pd.read_csv('final_def.csv')

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Combined plot showcasing all three groups
fig, ax = plt.subplots(figsize=(10, 5))

# Differentiate groups by linestyle and color
plot_data_seaborn(final_bonus_time, 'EA1', ax, linestyle='-', color='blue')
plot_data_seaborn(final_no_time, 'EA2', ax, linestyle='--', color='green')
plot_data_seaborn(final_def, 'EA3', ax, linestyle='-.', color='red')

ax.legend()
plt.tight_layout()
plt.grid(False)
plt.show()

# Save combined plot
fig.savefig('plots/combined_plot.png')
