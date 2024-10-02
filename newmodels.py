import numpy as np
import time
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeRegressor
from codecarbon import EmissionsTracker

emissions_data = []

def track_model_emissions(model_func, model_name):
    tracker = EmissionsTracker()
    tracker.start()

    model_func()

    emissions = tracker.stop()
    emissions_data.append({
        "model": model_name,
        "emissions_kg": emissions,  
    })

def run_for_min_time(model_func, min_time=10):
    start_time = time.time()
    model_func() 

    # Keep running the model until 10 seconds have passed
    while time.time() - start_time < min_time:
        pass

# 1. Linear Regression
def linear_regression_model():
    X = np.random.rand(100, 1)
    y = 3 * X.squeeze() + np.random.randn(100)
    model = LinearRegression()
    model.fit(X, y)

track_model_emissions(lambda: run_for_min_time(linear_regression_model), "Linear Regression")

# 2. Random Forest Classifier
def random_forest_model():
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, size=100)
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X, y)

track_model_emissions(lambda: run_for_min_time(random_forest_model), "Random Forest")

# 3. K-Means Clustering
def kmeans_model():
    X = np.random.rand(100, 2)
    model = KMeans(n_clusters=3)
    model.fit(X)

track_model_emissions(lambda: run_for_min_time(kmeans_model), "K-Means Clustering")

# 4. Neural Network
def neural_network_model():
    X = np.random.rand(100, 20)
    y = np.random.randint(0, 2, size=100)
    model = MLPClassifier(hidden_layer_sizes=(50,))
    model.fit(X, y)

track_model_emissions(lambda: run_for_min_time(neural_network_model), "Neural Network")

# 5. Decision Tree Regressor
def decision_tree_model():
    X = np.random.rand(100, 1)
    y = 2 * X.squeeze() + np.random.randn(100)
    model = DecisionTreeRegressor()
    model.fit(X, y)

track_model_emissions(lambda: run_for_min_time(decision_tree_model), "Decision Tree")


for data in emissions_data:
    print(f"Model: {data['model']}, Emissions: {data['emissions_kg']} kg CO2")
