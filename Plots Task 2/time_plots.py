import pandas as pd
import os
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Load data from CSV files
bonus_time_df = pd.read_csv('bt_times.csv')
no_time_df = pd.read_csv('nt_times.csv')
def_df = pd.read_csv('df_times.csv')

# # Define a function to check normality
def check_normality(data, label):
#     # Histogram / KDE
    sns.histplot(data, kde=True)
    plt.title(f'Histogram/KDE for {label}')
    # plt.show()

#     # Q-Q plot
    stats.probplot(data["mean_time"], plot=plt)
    plt.title(f'Q-Q Plot for {label}')
    # plt.show()

#     # Shapiro-Wilk test
    shapiro_stat, shapiro_p = stats.shapiro(data["mean_time"])
    if shapiro_p > 0.05:
         print(f"{label} appears to be normally distributed (Shapiro-Wilk p-value = {shapiro_p:.4f})")
    else:
         print(f"{label} does not appear to be normally distributed (Shapiro-Wilk p-value = {shapiro_p:.4f})")

# # Check normality for each dataset
check_normality(bonus_time_df, 'Bonus Time')
check_normality(no_time_df, 'No Time')
check_normality(def_df, 'Default')




# 1. Visual Comparisons: Boxplots
data = [bonus_time_df["mean_time"], no_time_df["mean_time"], def_df["mean_time"]]
labels = ['Bonus Time', 'No Time', 'Default']
plt.boxplot(data, labels=labels)
plt.title('Boxplots of Mean Runtimes for Each Group')
plt.ylabel('Time')
plt.show()

# 2. Mann-Whitney U Test
# Comparing Bonus Time vs No Time
u_statistic, p_value = stats.mannwhitneyu(bonus_time_df["mean_time"], no_time_df["mean_time"], alternative='two-sided')
print(f"Bonus Time vs No Time: Mann-Whitney U Test p-value = {p_value:.6f}")

# Repeat for other pairs...
# Comparing Bonus Time vs Default
u_statistic, p_value = stats.mannwhitneyu(bonus_time_df["mean_time"], def_df["mean_time"], alternative='two-sided')
print(f"Bonus Time vs Default: Mann-Whitney U Test p-value = {p_value:.6f}")

# Comparing No Time vs Default
u_statistic, p_value = stats.mannwhitneyu(no_time_df["mean_time"], def_df["mean_time"], alternative='two-sided')
print(f"No Time vs Default: Mann-Whitney U Test p-value = {p_value:.6f}")



# 3. Kruskal-Wallis H Test
h_statistic, p_value = stats.kruskal(bonus_time_df["mean_time"], no_time_df["mean_time"], def_df["mean_time"])
print(f"Kruskal-Wallis H Test p-value = {p_value:.4f}")

bonus_time_data = bonus_time_df["mean_time"].values
no_time_data = no_time_df["mean_time"].values
default_data = def_df["mean_time"].values


import numpy as np

def bootstrap_difference(data1, data2, num_iterations=10000):
    observed_difference = np.mean(data1) - np.mean(data2)
    bootstrap_differences = []
    
    for _ in range(num_iterations):
        bootstrap_sample1 = np.random.choice(data1, len(data1), replace=True)
        bootstrap_sample2 = np.random.choice(data2, len(data2), replace=True)
        
        bootstrap_difference = np.mean(bootstrap_sample1) - np.mean(bootstrap_sample2)
        bootstrap_differences.append(bootstrap_difference)
    
    lower_bound = np.percentile(bootstrap_differences, 2.5)
    upper_bound = np.percentile(bootstrap_differences, 97.5)
    
    return observed_difference, lower_bound, upper_bound

# Comparisons
bonus_no_time_diff, bonus_no_time_lower, bonus_no_time_upper = bootstrap_difference(bonus_time_data, no_time_data)
bonus_default_diff, bonus_default_lower, bonus_default_upper = bootstrap_difference(bonus_time_data, default_data)
no_time_default_diff, no_time_default_lower, no_time_default_upper = bootstrap_difference(no_time_data, default_data)

print("Difference between Bonus Time and No Time:", bonus_no_time_diff, "95% CI:", (bonus_no_time_lower, bonus_no_time_upper))
print("Difference between Bonus Time and Default:", bonus_default_diff, "95% CI:", (bonus_default_lower, bonus_default_upper))
print("Difference between No Time and Default:", no_time_default_diff, "95% CI:", (no_time_default_lower, no_time_default_upper))

import scikit_posthocs as sp
data_combined = bonus_time_df["mean_time"].tolist() + no_time_df["mean_time"].tolist() + def_df["mean_time"].tolist()
labels_combined = ['Bonus Time'] * len(bonus_time_df) + ['No Time'] * len(no_time_df) + ['Default'] * len(def_df)
p_values = sp.posthoc_dunn([bonus_time_df["mean_time"], no_time_df["mean_time"], def_df["mean_time"]], p_adjust='bonferroni')
print(f'| {p_values} |')

adjusted_p_value = p_value * 3
adjusted_p_value = min(adjusted_p_value, 1)  # Ensures p-value doesn't exceed 1


# Calculating the median absolute deviation
def mad(data):
    return np.median(np.abs(data - np.median(data)))

# Calculating Cohen's d for non-parametric tests
def non_parametric_cohens_d(group1, group2):
    pooled_mad = (mad(group1) + mad(group2)) / 2
    d = (np.median(group1) - np.median(group2)) / pooled_mad
    return d

# Calculate and print the Cohen's d effect size and actual differences
d_12 = non_parametric_cohens_d(bonus_time_df["mean_time"], no_time_df["mean_time"])
d_13 = non_parametric_cohens_d(bonus_time_df["mean_time"], def_df["mean_time"])
d_23 = non_parametric_cohens_d(no_time_df["mean_time"], def_df["mean_time"])

print("Cohen's d between Bonus Time and No Time:", d_12)
print("Cohen's d between Bonus Time and Default:", d_13)
print("Cohen's d between No Time and Default:", d_23)

print("\nMedian runtime for Bonus Time:", np.median(bonus_time_df["mean_time"]))
print("Median runtime for No Time:", np.median(no_time_df["mean_time"]))
print("Median runtime for Default:", np.median(def_df["mean_time"]))

# Visual Inspection
sns.boxplot(data=[bonus_time_df["mean_time"], no_time_df["mean_time"], def_df["mean_time"]], orient="v")
plt.xticks([0,1,2], ["Bonus Time", "No Time", "Default"])
plt.ylabel('Time')
plt.title('Boxplots of Mean Runtimes for Each Group')
plt.show()


