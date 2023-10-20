import pandas as pd
from scipy import stats

for train_group in [0, 1]:
    print(f"Conducting statistical analysis for training group {train_group}\n")

    # Load the data from the CSV files
    data1 = pd.read_csv(f'deap_specialist/350_gens/def_{train_group}_winners.csv')
    data2 = pd.read_csv(f'deap_specialist/350_gens/no_time_{train_group}_winners.csv')
    data3 = pd.read_csv(f'deap_specialist/350_gens/bonus_time_{train_group}_winners.csv')

    # Calculate the difference between player_life and enemy_life for each dataset
    data1['life_diff'] = data1['player_life'] - data1['enemy_life']
    data2['life_diff'] = data2['player_life'] - data2['enemy_life']
    data3['life_diff'] = data3['player_life'] - data3['enemy_life']

    # Perform the t-test between data1 and data2 (default and removing time penalty)
    t_stat, p_value = stats.ttest_ind(data1['life_diff'], data2['life_diff'], equal_var=False)

    # Output the results
    print(f"Statistics Analysis between default fitness and fitness removing time penalty) : {t_stat}")
    print(f"T-statistic: {t_stat}")
    print(f"P-value: {p_value}")
    # Interpret the results
    alpha = 0.05  # Set significance level to 5%
    if p_value < alpha:
        print("The difference between the means is statistically significant.")
    else:
        print("The difference between the means is not statistically significant.")



    # Perform the t-test between data1 and data3 (default and adding time as a bonus instead)
    t_stat, p_value = stats.ttest_ind(data1['life_diff'], data3['life_diff'], equal_var=False)

    # Output the results
    print(f"Statistics Analysis between default fitness and fitness function with time as a bonus) : {t_stat}")
    print(f"T-statistic: {t_stat}")
    print(f"P-value: {p_value}")

    # Interpret the results
    alpha = 0.05  # Set significance level to 5%
    if p_value < alpha:
        print("The difference between the means is statistically significant.")
    else:
        print("The difference between the means is not statistically significant.")


    print("\n")