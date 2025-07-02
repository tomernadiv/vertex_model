import numpy as np
import matplotlib.pyplot as plt

# Configuration
simulation_number = 3                    # Simulation number
# target_y = 300                              # Target Y value for horizontal line and root finding
# y_variable = "final_window_area"            # Choose: "final_window_area" or "max_velocity_at_10"

target_y = 1.5                              # Target Y value for horizontal line and root finding
y_variable = "max_velocity_at_10"            # Choose: "final_window_area" or "max_velocity_at_10"

# 1. Load your result file
filename = f"results/simulation_{simulation_number}_final_results.txt"
param_values = []
final_window_areas = []
max_velocities = []

with open(filename, 'r') as f:
    for line in f:
        if line.startswith("#") or line.strip() == "":
            continue
        parts = line.strip().split()
        param_values.append(float(parts[0]))
        final_window_areas.append(float(parts[1]))
        max_velocities.append(float(parts[2]))

param_values = np.array(param_values)
final_window_areas = np.array(final_window_areas)
max_velocities = np.array(max_velocities)

# 2. Choose the Y values and degree of polynomial
if y_variable == "final_window_area":
    y_values = final_window_areas
    fit_degree = 2
elif y_variable == "max_velocity_at_10":
    y_values = max_velocities
    fit_degree = 1
else:
    raise ValueError(f"Unknown y_variable: {y_variable}")

# 3. Fit polynomial
coeffs = np.polyfit(param_values, y_values, deg=fit_degree)
poly = np.poly1d(coeffs)

# 4. Generate smooth curve for plotting
x_fit = np.linspace(min(param_values), max(param_values), 200)
y_fit = poly(x_fit)

# 5. Plotting
plt.figure(figsize=(8, 6))
plt.plot(param_values, y_values, 'o', label='Simulation Data')
plt.plot(x_fit, y_fit, '-', label=f'{fit_degree}° Fit')
plt.axhline(y=target_y, color='r', linestyle='--', label=f'y = {target_y}')

plt.xlabel('Parameter Value')
plt.ylabel(y_variable.replace("_", " ").title())
plt.title(f'Simulation {simulation_number}: {y_variable.replace("_", " ").title()} vs Parameter')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. Solve for x where y = target_y
adjusted_coeffs = coeffs.copy()
adjusted_coeffs[-1] -= target_y
roots = np.roots(adjusted_coeffs)
real_roots = [r.real for r in roots if np.isreal(r) and min(param_values) <= r.real <= max(param_values)]

print(f"x value(s) where {y_variable} = {target_y}:", real_roots)
print("final window area:", final_window_areas[-1])
