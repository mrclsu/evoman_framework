import pandas as pd
import os

# Lists to store time values
bonus_time_times = []
no_time_times = []
def_times = []

# Extract times from CSV files
for file in os.listdir():
    if file.startswith('bonus_time_') and file.endswith('statistics.csv'):
        df = pd.read_csv(file)
        bonus_time_times.extend(df['mean_time'].tolist())
    elif file.startswith('no_time_') and file.endswith('statistics.csv'):
        df = pd.read_csv(file)
        no_time_times.extend(df['mean_time'].tolist())
    elif file.startswith('def_') and file.endswith('statistics.csv') :
        df = pd.read_csv(file)
        def_times.extend(df['mean_time'].tolist())

# Convert each list of times to a DataFrame and save to CSV
bonus_time_df = pd.DataFrame(bonus_time_times, columns=['mean_time'])
bonus_time_df.to_csv('bt_times.csv', index=False)

no_time_df = pd.DataFrame(no_time_times, columns=['mean_time'])
no_time_df.to_csv('nt_times.csv', index=False)

def_df = pd.DataFrame(def_times, columns=['mean_time'])
def_df.to_csv('df_times.csv', index=False)
