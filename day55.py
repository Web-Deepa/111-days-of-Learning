#day55 -Review and Mini Project: House Price Prediction in Nepal
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Creating sample data
print("Creating house price dataset...")
np.random.seed(42)
n_houses = 1000

data = {
    "area_sqft": np.random.randint(800, 4500, n_houses),
    "bedrooms": np.random.randint(2, 8, n_houses),
    "bathrooms": np.random.randint(1, 6, n_houses),
    "floors": np.random.choice([1, 2, 2.5, 3, 5, 4], n_houses),
    "road_width_ft": np.random.randint(10, 26, n_houses),
}

df = pd.DataFrame(data)

# Calculate  price in Lakhs NPR 
df["price_lakhs"] = (
    (df["area_sqft"] * 8)
    + (df["bedrooms"] * 15)
    + (df["floors"] * 25)
    + (df["road_width_ft"] * 5)
    + np.random.normal(0, 30, n_houses)
)

x = df[["area_sqft", "bedrooms", "bathrooms", "floors", "road_width_ft"]]
y = df["price_lakhs"]

# 2. Split data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# 3. Scale the features 
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 4. Train model

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train_scaled, y_train)

# 5. Test and evaluate model 
predictions = model.predict(x_test_scaled)

# Metrics for Regression
maen= mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n--- Project Results ---")
print(f"Model Accuracy (R2 Score): {r2 * 100:.2f}%")
print(f"Average Prediction Error (MAE): {mae:.2f} Lakhs NPR")

# Sample Prediction Test
print("\n--- Testing Model with a Sample House ---")
#example:
sample_house = np.array([[1500, 4, 3, 2.5, 13]])
sample_scaled = scaler.transform(sample_house)
predicted_price = model.predict(sample_scaled)[0]
print(
    f"Predicted Price for 1500 sqft house: {predicted_price:.2f} Lakhs (~{predicted_price/100:.2f} Crore NPR)"
)

# 6. Save model
print("\nSaving the model and scaler for deployment...")
joblib.dump(model, "nepal_house_model.pkl")
joblib.dump(scaler, "nepal_house_scaler.pkl")
print("\n--- Project Results ---")
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=data.target_names))