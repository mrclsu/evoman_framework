import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pingouin as pg
import pdb

# Constants
file_path = "../deap_specialist/350_gens"
enemy_groups = [0, 1]
ea_types = ['EA1', 'EA2', 'EA3']
runs = list(range(10))

# Function to get the correct prefix
def get_prefix(ea_type):
    if ea_type == "EA1":
        return "bonus_time"
    elif ea_type == "EA2":
        return "def"
    else:
        return "no_time"

# Placeholder for all the gains
all_gains = {ea: {group: [] for group in enemy_groups} for ea in ea_types}

# Load the data, compute the gains
for ea_type in ea_types:
    prefix = get_prefix(ea_type)
    for group in enemy_groups:
        for run in runs:
            filename = f"{file_path}/{prefix}_{group}_{run}_statistics.csv"
            df = pd.read_csv(filename)
            df['gain'] = df['mean_player_life'] - df['mean_enemy_life']
            avg_top5_gain = df.nlargest(5, 'gain')['gain'].mean()
            all_gains[ea_type][group].append(avg_top5_gain)

# Plotting the boxplots
fig, ax = plt.subplots()

# Group data for boxplots
boxplot_data = [
    all_gains['EA1'][0], all_gains['EA2'][0], all_gains['EA3'][0],
    all_gains['EA1'][1], all_gains['EA2'][1], all_gains['EA3'][1]
]

# Set up the boxplot labels
labels = [
    'EA1', 'EA2', 'EA3',
    'EA1', 'EA2', 'EA3'
]

boxplots = ax.boxplot(boxplot_data, vert=True, patch_artist=True, labels=labels)
colors = ['lightgreen', 'lightgreen', 'lightgreen', 'lightblue', 'lightblue', 'lightblue']

for patch, color in zip(boxplots['boxes'], colors):
    patch.set_facecolor(color)

# Add mean numbers on top of the median lines
medians = [median.get_ydata()[0] for median in boxplots['medians']]
for i, (median, label) in enumerate(zip(medians, labels)):
    ax.text(i + 1, median + 0.5, f'{median:.2f}', ha='center', va='center', fontsize=8, color='black')

# Add legends
legend_elements = [
    Patch(facecolor='lightgreen', edgecolor='black', label='[2,3,4,5,6,8]'),
    Patch(facecolor='lightblue', edgecolor='black', label='[1,4,6,8]')
]
ax.legend(handles=legend_elements, loc='lower right')

ax.set_title('Boxplots of Mean Gain for Different EAs and Enemy Groups')
ax.set_ylabel('Gain')
plt.show()

# Perform t-tests
for ea_type in ea_types:
    ttest_result = pg.ttest(all_gains[ea_type][0], all_gains[ea_type][1])
    print(f"T-test results for {ea_type} between Group 0 and Group 1:")
    print(ttest_result, "\n")


