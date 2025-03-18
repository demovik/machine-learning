import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# Load dataset
df = pd.read_csv("learn_data.csv")
X = torch.tensor(df[["num1", "num2"]].values, dtype=torch.float32)
y = torch.tensor(df["product"].values, dtype=torch.float32).view(-1, 1)

# Define a better neural network
class MultiplicationNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 50)  # Increased neurons
        self.fc2 = nn.Linear(50, 50)  # New hidden layer
        self.fc3 = nn.Linear(50, 1)   # Output layer
        self.relu = nn.LeakyReLU()  # Better than plain ReLU

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# Initialize model
model = MultiplicationNN()
criterion = nn.L1Loss()

optimizer = optim.Adam(model.parameters(), lr=0.01)  # Start higher
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)  # Reduce LR over time

# Train model
for epoch in range(20000):  # Increase epochs for better learning
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    scheduler.step()  # Apply learning rate decay

    if epoch % 2000 == 0:  # Print progress every 2000 epochs
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Save model
torch.save(model.state_dict(), "multiplication_nn.pth")
print("Neural Network model trained and saved as multiplication_nn.pth")
