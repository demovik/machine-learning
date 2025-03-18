import torch
import torch.nn as nn

# Define the same model as in train_nn.py
class MultiplicationNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 1)  # Ensure fc3 exists
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)  # Ensure fc3 is used

# Load trained model
model = MultiplicationNN()
model.load_state_dict(torch.load("multiplication_nn.pth"))
model.eval()

# Test the model
def predict_multiplication(num1, num2):
    input_tensor = torch.tensor([[num1, num2]], dtype=torch.float32)
    output = model(input_tensor).item()
    return round(output, 2)  # Roun

# Test predictions
print("5 x 5 =", predict_multiplication(5, 5))  # Should return ~25
print("7 x 8 =", predict_multiplication(7, 8))  # Should return ~56
print("9 x 9 =", predict_multiplication(9, 9))  # Should return ~81
print("10 x 10 =", predict_multiplication(10, 10))  # Should return ~100
print("11 x 11 =", predict_multiplication(11, 11))  # Should return ~121
print("18 x 36 =", predict_multiplication(18, 36))  # Should return ~648