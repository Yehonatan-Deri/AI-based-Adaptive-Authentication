# Path: local_outlier_factor.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

"""
ChatGPT: CHAT LOG: https://chatgpt.com/share/db2e3337-d356-47cc-a65d-97d4c26755d8
"""

# Generate sample data
np.random.seed(42)
X_inliers = 0.3 * np.random.randn(100, 2)
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

X = np.r_[X_inliers + 2, X_inliers - 2, X_outliers]


# Initialize the Local Outlier Factor model
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)

# Fit the model and predict the outliers
y_pred = lof.fit_predict(X)

# The negative outlier factor (NOF) for each sample
lof_scores = -lof.negative_outlier_factor_


# Identify outliers
outliers = X[y_pred == -1]
inliers = X[y_pred != -1]

print("Number of outliers detected:", len(outliers))

# Plot the results
plt.scatter(inliers[:, 0], inliers[:, 1], color='blue', label='Inliers')
plt.scatter(outliers[:, 0], outliers[:, 1], color='red', label='Outliers')
plt.title('Local Outlier Factor (LOF)')
plt.legend()
plt.show()


