import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import joblib

# Load dataset
df = pd.read_csv("learn_data.csv")

# Features and labels
X = df[["num1", "num2"]]
y = df["product"]

# Create a Polynomial Regression model
model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())

# Train the model
model.fit(X, y)

# Save the model
joblib.dump(model, "multiplication_model.pkl")

print("Polynomial Regression model trained and saved as multiplication_model.pkl")
