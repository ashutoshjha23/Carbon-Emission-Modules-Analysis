import numpy as np
import time
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from codecarbon import EmissionsTracker

# List to store emissions data
emissions_data = []

# General function to track model emissions and execution time
def track_model_emissions(model_func, model_name, *args, **kwargs):
    tracker = EmissionsTracker()
    tracker.start()

    start_time = time.time()
    model_func(*args, **kwargs)
    execution_time = time.time() - start_time

    emissions = tracker.stop()
    emissions_data.append({
        "model": model_name,
        "emissions_kg": emissions,  
        "execution_time_s": execution_time
    })

# Ensure a model runs for a minimum amount of time
def run_for_min_time(model_func, min_time=10):
    start_time = time.time()
    model_func()  # Initial model run

    while time.time() - start_time < min_time:
        model_func()  # Re-run the model to meet the time constraint

# Helper function for generating random datasets
def generate_random_data(shape, integer=False, clusters=False):
    if clusters:
        return np.random.rand(*shape)
    return np.random.randint(0, 2, size=shape) if integer else np.random.rand(*shape)

# 1. Linear Regression Model
def linear_regression_model():
    X = generate_random_data((100, 1))
    y = 3 * X.squeeze() + np.random.randn(100)
    model = LinearRegression()
    model.fit(X, y)

# 2. Random Forest Classifier
def random_forest_model():
    X = generate_random_data((100, 10))
    y = generate_random_data((100,), integer=True)
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X, y)

# 3. K-Means Clustering
def kmeans_model():
    X = generate_random_data((100, 2))
    model = KMeans(n_clusters=3)
    model.fit(X)

# 4. Neural Network Classifier
def neural_network_model():
    X = generate_random_data((100, 20))
    y = generate_random_data((100,), integer=True)
    model = MLPClassifier(hidden_layer_sizes=(50,))
    model.fit(X, y)

# 5. Decision Tree Regressor
def decision_tree_model():
    X = generate_random_data((100, 1))
    y = 2 * X.squeeze() + np.random.randn(100)
    model = DecisionTreeRegressor()
    model.fit(X, y)

# 6. Support Vector Classifier (SVC)
def svc_model():
    X = generate_random_data((100, 20))
    y = generate_random_data((100,), integer=True)
    model = SVC(kernel='linear')
    model.fit(X, y)

# 7. Gradient Boosting Classifier
def gradient_boosting_model():
    X = generate_random_data((100, 20))
    y = generate_random_data((100,), integer=True)
    model = GradientBoostingClassifier(n_estimators=50)
    model.fit(X, y)

# 8. Ridge Regression (Linear Regression with L2 Regularization)
def ridge_regression_model():
    X = generate_random_data((100, 1))
    y = 3 * X.squeeze() + np.random.randn(100)
    model = Ridge(alpha=1.0)
    model.fit(X, y)

# 9. Gaussian Naive Bayes Classifier
def naive_bayes_model():
    X = generate_random_data((100, 20))
    y = generate_random_data((100,), integer=True)
    model = GaussianNB()
    model.fit(X, y)

# 10. AdaBoost Classifier
def adaboost_model():
    X = generate_random_data((100, 20))
    y = generate_random_data((100,), integer=True)
    model = AdaBoostClassifier(n_estimators=50)
    model.fit(X, y)

# Track emissions and execution time for each model
track_model_emissions(lambda: run_for_min_time(linear_regression_model), "Linear Regression")
track_model_emissions(lambda: run_for_min_time(random_forest_model), "Random Forest Classifier")
track_model_emissions(lambda: run_for_min_time(kmeans_model), "K-Means Clustering")
track_model_emissions(lambda: run_for_min_time(neural_network_model), "Neural Network Classifier")
track_model_emissions(lambda: run_for_min_time(decision_tree_model), "Decision Tree Regressor")
track_model_emissions(lambda: run_for_min_time(svc_model), "Support Vector Classifier (SVC)")
track_model_emissions(lambda: run_for_min_time(gradient_boosting_model), "Gradient Boosting Classifier")
track_model_emissions(lambda: run_for_min_time(ridge_regression_model), "Ridge Regression")
track_model_emissions(lambda: run_for_min_time(naive_bayes_model), "Gaussian Naive Bayes Classifier")
track_model_emissions(lambda: run_for_min_time(adaboost_model), "AdaBoost Classifier")

# Output emissions and execution data
for data in emissions_data:
    print(f"Model: {data['model']}, Emissions: {data['emissions_kg']:.4f} kg CO2, Execution Time: {data['execution_time_s']:.2f} seconds")
