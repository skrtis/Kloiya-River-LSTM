import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# flow rates 
flow_rates = np.array([51.24, 61.48, 71.75, 81.98, 102.47, 122.96, 133.21, 153.71, 174.20])

# efficiency levels 
efficiency_levels = np.array([80, 85, 87, 90, 85, 83, 80, 73, 60])

distributed = np.array([0.10] * len(flow_rates)) # distributed time by percentile

without_optimization = np.sum(distributed * efficiency_levels) # calculate efficiency without optimization,

q_eighty_opt = efficiency_levels.copy() # optimized efficiency for high flow rates
q_eighty_opt[flow_rates >= 81.98] = 90

with_optimization = np.sum(distributed * q_eighty_opt) # calculate efficiency with optimization

q_eighty_opt[flow_rates >= 61.48] = 90 # optimized efficiency for Q60+

q_sixty_opt = np.sum(distributed * q_eighty_opt) #extended op

gain_eighty = ((with_optimization - without_optimization) / without_optimization) * 100 #efficiency gain
gain_sixty = ((q_sixty_opt - without_optimization) / without_optimization) * 100

# maximum power output in megawatts
P_max = 4.070/0.723

# calculate power output for each scenario
no_opt_power = P_max * (without_optimization / 100)
eighty_opt_power = P_max * (with_optimization / 100)
sixty_opt_power = P_max * (q_sixty_opt / 100)

# calculate extra power gained
extra_power_output = eighty_opt_power - no_opt_power
extra_power_output_extended = sixty_opt_power - no_opt_power

# create a dataframe to display results
data_extended_optimization = {
    "Scenario": ["Without Optimization", "With Optimization (Q80+)", "With Extended Optimization (Q60+)"],
    "Efficiency (%)": [without_optimization, with_optimization, q_sixty_opt],
    "Power Output (kW)": [no_opt_power, eighty_opt_power, sixty_opt_power],
    "Extra Power Gained (kW)": [0, extra_power_output, extra_power_output_extended]
}

df_extended_results = pd.DataFrame(data_extended_optimization)

print(df_extended_results)
# plot flow rate vs efficiency 
plt.figure(figsize=(8, 5))
plt.plot(flow_rates, efficiency_levels, label="Efficiency", color='blue', marker='o')
plt.xlabel("Flow Rate (m³/s)")
plt.ylabel("Efficiency (%)")
plt.grid()
plt.title("Flow Rate vs. Efficiency")

plt.figure(figsize=(8, 5))
plt.bar(df_extended_results["Scenario"], df_extended_results["Power Output (kW)"], color=['blue', 'green', 'red'])
plt.xlabel("Optimization Scenario")
plt.ylabel("Power Output (kW)")
plt.title("Power Output Comparison for Different Optimization Scenarios")
plt.xticks(rotation=15)
plt.grid(axis='y', linestyle="--", alpha=0.7)
plt.show()
