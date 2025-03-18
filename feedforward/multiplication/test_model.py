import pandas as pd
import joblib

# Load model
model = joblib.load("multiplication_model.pkl")

# Predict function
def predict_multiplication(a, b):
    import numpy as np
    return model.predict(pd.DataFrame([[a, b]], columns=["num1", "num2"]))[0]

# Test model
print("5 x 5 =", predict_multiplication(5, 5))  # Should return ~25
print("7 x 8 =", predict_multiplication(7, 8))  # Should return ~56
print("9 x 9 =", predict_multiplication(9, 9))  # Should return ~81
print("10 x 10 =", predict_multiplication(10, 10))  # Should return ~100
print("11 x 11 =", predict_multiplication(11, 11))  # Should return ~121
print("18 x 36 =", predict_multiplication(18, 36))  # Should return ~648