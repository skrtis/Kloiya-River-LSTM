import numpy as np
np.set_printoptions(legacy='1.25')
import matplotlib.pyplot as plt
import pandas as pd

csv_file_path = "/Users/kurtisng/Documents/isci/FR/flowratemodel/std_analysis/kloiya.csv"

# read the csv and take flow points
df = pd.read_csv(csv_file_path)
flow_values = df["Value"]
time_values = df["Date"]

# Convert Date Format in Days since the first day
time_values = pd.to_datetime(time_values)

# Latex and Font
plt.rcParams.update({
    "text.usetex": True,
    "font.family"  : "CMU Serif",
    "font.size" : 20
})

# initial analysis, to find the mean, median, standard deviation, variance, coefficient of variation and range of flow rates (min, max)
def flow_analysis(a, b):
    # Convert input dates to datetime
    a = pd.to_datetime(a)
    b = pd.to_datetime(b)

    # Check if the dates exist in the time_values array
    if a not in time_values.values or b not in time_values.values:
        raise ValueError("One or both of the dates are not in the time_values array")

    # Find the index of the date a and b
    a_index = time_values[time_values == a].index[0]
    b_index = time_values[time_values == b].index[0]

    # Find the flow values in the range of date
    flow_values_range = flow_values[a_index:b_index]

    mean = np.mean(flow_values_range)
    median = np.median(flow_values_range)
    std = np.std(flow_values_range)
    var = np.var(flow_values_range)
    cv = std / mean
    min_flow = np.min(flow_values_range)
    max_flow = np.max(flow_values_range)

    # return all the values rounded
    return round(mean, 2), round(median, 2), round(std, 2), round(var, 2), round(cv, 2), round(min_flow, 2), round(max_flow, 2)

print(flow_analysis("2018-01-01", "2022-01-01"))

#creates a flow duration curve
def flow_duration_curve(a, b):
    # Convert input dates to datetime
    a = pd.to_datetime(a)
    b = pd.to_datetime(b)

    # Check if the dates exist in the time_values array
    if a not in time_values.values or b not in time_values.values:
        raise ValueError("One or both of the dates are not in the time_values array")

    # Find the indices
    a_index = time_values[time_values == a].index[0]
    b_index = time_values[time_values == b].index[0]

    # Find the flow values in the range of date
    flow_values_range = flow_values[a_index:b_index]

    #multiply the flow_values by the conversion factor, 11.01938196
    flow_values_range = flow_values_range * 11.01938196

    # Sort the flow values
    flow_values_range = np.sort(flow_values_range)

    # find ranks of flow values in a list
    rank = np.arange(1, len(flow_values_range) + 1)

    #find the exceedance probability of each flow value
    exceedance_probability = 1 - (rank / len(flow_values_range))

    plt.plot(exceedance_probability, flow_values_range, label="2018-2022 Spillway Flow", color='black', linewidth=2)

    plt.title("Dam Spillway Flow Duration Curve 2018-2022")
    plt.xlabel("Exceedance Probability")
    plt.ylabel("Flow Rate ($m^3/s$)")
    plt.grid()
    plt.legend()
    plt.show()

    # Create the flow duration curve
    return exceedance_probability, flow_values_range

flow_duration_curve("2018-01-01", "2022-01-01")

def plot_flow_duration_curve():
    # store the exceedance probability and flow values for each range of date, add 2018-2022
    exceedance_probability_2018, flow_values_2018 = flow_duration_curve("2018-01-01", "2022-01-01")
    exceedance_probability_2012, flow_values_2012 = flow_duration_curve("2012-01-01", "2022-01-01")
    exceedance_probability_2002, flow_values_2002 = flow_duration_curve("2002-01-01", "2022-01-01")
    exceedance_probability_1992, flow_values_1992 = flow_duration_curve("1992-01-01", "2022-01-01")
    exceedance_probability_1982, flow_values_1982 = flow_duration_curve("1982-01-01", "2022-01-01")

    #plots
    plt.plot(exceedance_probability_2018, flow_values_2018, label="2018-2022", color='purple', linewidth=0.8)
    plt.plot(exceedance_probability_2012, flow_values_2012, label="2012-2022", color='black', linewidth=0.8)
    plt.plot(exceedance_probability_2002, flow_values_2002, label="2002-2022", color='red', linewidth=0.8)
    plt.plot(exceedance_probability_1992, flow_values_1992, label="1992-2022", color='blue', linewidth=0.8)
    plt.plot(exceedance_probability_1982, flow_values_1982, label="1982-2022", color='green', linewidth=0.8)

    plt.title("Flow Duration Curves of Different Time Ranges")
    plt.xlabel("Exceedance Probability")
    plt.ylabel("Flow Rate ($m^3/s$)")
    plt.ylim(0,30)
    plt.grid()
    plt.legend()
    plt.show()
    
    return {
        "2018-2022": (exceedance_probability_2018, flow_values_2018),
        "2012-2022": (exceedance_probability_2012, flow_values_2012),
        "2002-2022": (exceedance_probability_2002, flow_values_2002),
        "1992-2022": (exceedance_probability_1992, flow_values_1992),
        "1982-2022": (exceedance_probability_1982, flow_values_1982)
    }

def find_closest_value(exceedance_probabilities, flow_values, target):
    idx = (np.abs(exceedance_probabilities - target)).argmin()
    return exceedance_probabilities[idx], flow_values[idx]

def print_flow_rate_table():
    data = plot_flow_duration_curve()
    targets = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

    print(f"{'Date Range':<12} {'Exceedance Probability':<25} {'Flow Rate (m^3/s)':<20}")
    print("="*60)
    for date_range, (exceedance_probabilities, flow_values) in data.items():
        for target in targets:
            closest_prob, closest_flow = find_closest_value(exceedance_probabilities, flow_values, target)
            print(f"{date_range:<12} {closest_prob:<25.2f} {closest_flow*11.01938196:<20.2f}")


# give the metrics 
#histogram distribution for a range of dates
def plot_histogram(a, b):
    # Convert input dates to datetime
    a = pd.to_datetime(a)
    b = pd.to_datetime(b)

    # Check if the dates exist in the time_values array
    if a not in time_values.values or b not in time_values.values:
        raise ValueError("One or both of the dates are not in the time_values array")

    # Find the index of the date a and b
    a_index = time_values[time_values == a].index[0]
    b_index = time_values[time_values == b].index[0]

    # Find the flow values in the range of date
    flow_values_range = flow_values[a_index:b_index]

    # Plot the histogram
    plt.hist(flow_values_range, bins = 30, color='black', edgecolor='white', linewidth=0.5)
    # make the title "Flow Rate Distribution from a-b" but only the year of a and b
    plt.title("Flow Rate Distribution from {}-{}".format(a.year, b.year))
    plt.xlabel("Flow Rate ($m^3/s$)")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()

# Plot date vs. flow points.
plt.plot(time_values, flow_values, label="Flow Points", color='black', linewidth=0.7, marker='o', markersize=1)
plt.title("Flow Rate ($m^3/s$) vs. Date from 1964-2022")
plt.xlabel("Date")
plt.ylabel("Flow Rate ($m^3/s$)")
plt.grid()
plt.show()

