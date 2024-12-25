import numpy as np
from sklearn.utils import resample

# Example: R^2 scores for each model from multiple cross-validation folds
# model_a_r2 = np.array([0.78, 0.81, 0.79, 0.82, 0.80])
model_a_r2 = np.array([0.75, 0.77, 0.76, 0.78, 0.74])
model_b_r2 = np.array([0.79, 0.80, 0.76, 0.81, 0.83])

# Number of bootstrap samples
n_iterations = 10
differences = []

# Perform bootstrapping
for _ in range(n_iterations):
    # Resample the data with replacement
    sample_a = resample(model_a_r2, random_state=42)
    sample_b = resample(model_b_r2, random_state=42)
    print('sample_a, sample_b', sample_a, sample_b)
    
    # Calculate the difference between the means of the two samples
    diff = np.mean(sample_a) - np.mean(sample_b)
    differences.append(diff)

# Convert the differences to a numpy array
differences = np.array(differences)

# Calculate the p-value by checking how often the difference is greater than 0
sum = np.sum(differences >= 0)
print(np.sum(differences <= 0))
print(sum)
p_value = np.sum(differences >= 0) / n_iterations

# Print results
print(f"Bootstrap p-value: {p_value}")
