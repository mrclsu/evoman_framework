import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_runtimes_seaborn(df, title, ax):
    generations = df.index

    # Plot mean_time and its standard deviation
    ax.plot(generations, df['mean_time'], label='Mean Time')
    ax.fill_between(generations, 
                    df['mean_time'] - df['std_time'], 
                    df['mean_time'] + df['std_time'], 
                    alpha=0.3)
    
    # Set plot title, labels, and legend
    ax.set_title(title)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Runtime')
    ax.legend()

# Read the final CSV files
final_bonus_time = pd.read_csv('final_bonus_time_2.csv')
final_no_time = pd.read_csv('final_no_time_2.csv')
final_def = pd.read_csv('final_def_2.csv')

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Create a 'plots' directory if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

# Individual runtime plots for each group
fig, axes = plt.subplots(3, 1, figsize=(10, 15))

plot_runtimes_seaborn(final_bonus_time, 'Bonus Time Runtime', axes[0])
plot_runtimes_seaborn(final_no_time, 'No Time Runtime', axes[1])
plot_runtimes_seaborn(final_def, 'Default Runtime', axes[2])

plt.tight_layout()
plt.savefig('plots/individual_runtimes.png')
plt.show()

# Combined runtime plot showcasing all three groups
fig, ax = plt.subplots(figsize=(10, 5))

plot_runtimes_seaborn(final_bonus_time, 'Bonus Time Runtime', ax)
plot_runtimes_seaborn(final_no_time, 'No Time Runtime', ax)
plot_runtimes_seaborn(final_def, 'Default Runtime', ax)

plt.title('Comparison of Runtimes Across Training Groups')
plt.tight_layout()
plt.savefig('plots/combined_runtimes.png')
plt.show()
