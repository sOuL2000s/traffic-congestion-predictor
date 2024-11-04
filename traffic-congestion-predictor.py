import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample data: Day (0=Monday, 6=Sunday), Hour, Weather (0=Clear, 1=Rainy), Traffic Density (0=Low, 1=Moderate, 2=High)
data = {
    "Day": [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6],
    "Hour": [8, 9, 17, 18, 19, 20, 21, 8, 9, 17, 18, 19, 20, 21],
    "Weather": [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0],
    "TrafficDensity": [2, 2, 2, 1, 1, 0, 0, 2, 2, 1, 2, 1, 0, 1]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features and target variable
X = df[["Day", "Hour", "Weather"]]
y = df["TrafficDensity"]

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse:.2f}")

# Sample prediction
sample = pd.DataFrame({"Day": [3], "Hour": [9], "Weather": [0]})
predicted_traffic = model.predict(sample)
print(f"Predicted traffic density for the sample: {predicted_traffic[0]:.2f} (0=Low, 1=Moderate, 2=High)")
