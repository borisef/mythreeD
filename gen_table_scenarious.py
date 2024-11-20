import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def generate_scenarios(params, N, output_csv="scenarios.csv", hist_dir="histograms",
                       parallel_plot_file="parallel_coordinates.png",
                       scaled_parallel_plot_file="scaled_parallel_coordinates.png"):
    # Extract the options for each parameter
    param_names = [param['name'] for param in params]
    param_options = [param['options'] for param in params]

    # Get the Cartesian product of all parameter options
    all_combinations = list(itertools.product(*param_options))

    # Ensure N is not greater than the total number of possible combinations
    if N > len(all_combinations):
        raise ValueError(f"N is too large. Maximum possible unique scenarios: {len(all_combinations)}")

    # Randomly select N unique scenarios
    selected_combinations = np.random.choice(range(len(all_combinations)), size=N, replace=False)

    # Create the scenario table
    scenario_table = pd.DataFrame([all_combinations[i] for i in selected_combinations], columns=param_names)

    # Sort the rows by all parameters in sequence
    scenario_table.sort_values(by=param_names, inplace=True)

    # Save the table as CSV
    scenario_table.to_csv(output_csv, index=False)
    print(f"Scenarios saved to {output_csv}")

    # Create the histogram directory if it doesn't exist
    if not os.path.exists(hist_dir):
        os.makedirs(hist_dir)

    # Create and save histograms for each parameter
    for param in params:
        param_name = param['name']
        param_values = scenario_table[param_name]

        plt.figure()

        # Count occurrences for each option and plot barplot to match exact options
        counts = param_values.value_counts().reindex(param['options'], fill_value=0)
        sns.barplot(x=counts.index, y=counts.values)

        plt.xticks(param['options'])  # Ensure ticks exactly at the options
        plt.title(f'Histogram of {param_name}')
        plt.xlabel(param_name)
        plt.ylabel('Count')
        hist_file = f"{hist_dir}/{param_name}_histogram.png"
        plt.savefig(hist_file)
        plt.close()
        print(f"Histogram for {param_name} saved to {hist_file}")

    # Label encoding and scaling for parallel coordinates plot
    label_encoders = {}
    scaled_scenario_table = scenario_table.copy()

    for param in params:
        param_name = param['name']
        if scenario_table[param_name].dtype == 'object':  # Check if the column is categorical
            le = LabelEncoder()
            scaled_scenario_table[param_name] = le.fit_transform(scenario_table[param_name])
            label_encoders[param_name] = le
        # Now scale each column to [0, 100]
        scaler = MinMaxScaler(feature_range=(0, 100))
        scaled_scenario_table[param_name] = scaler.fit_transform(scaled_scenario_table[[param_name]])

    # Create and save scaled parallel coordinates plot
    plt.figure()
    pd.plotting.parallel_coordinates(scaled_scenario_table, param_names[0], color=sns.color_palette("Set2"))
    plt.title('Scaled Parallel Coordinates Plot')
    plt.savefig(scaled_parallel_plot_file)
    plt.close()
    print(f"Scaled parallel coordinates plot saved to {scaled_parallel_plot_file}")

    return scenario_table



# Example usage:
params = [
    {"name": "target", "options": ["car", "tank", "plane","drone", "bus", "truck"]},
    {"name": "range", "options": [1, 2, 3, 4, 5,6,7,8,9,10]},
    {"name": "ILU", "options": ["True", "False"]},
    {"name": "maneuvers", "options": ["small", "large", "moderate"]},
    {"name": "max_occlusion", "options": [0, 0.1, 0.2,0.3, 0.5]},
    {"name": "time_of_day", "options": [2,4,6,8,10,12,14,16,18,20,22]},
    {"name": "percentage_occluded", "options": [0,20,40,60,80,100]},
    {"name": "lost_times", "options": [0,0,0,1,2,3]}


]
N = 50

scenario_table = generate_scenarios(params, N, output_csv="data/output/scenarios.csv", hist_dir="data/output/histograms",
                       parallel_plot_file="data/output/parallel_coordinates.png",
                        scaled_parallel_plot_file="data/output/scaled_parallel_coordinates.png")
print(scenario_table)
