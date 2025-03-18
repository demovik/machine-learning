import torch
import torch.nn as nn
import torch.optim as optim

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define a simple feedforward neural network
class NumberPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

# Prepare data (flatten sequences and normalize)
data = torch.tensor([[2, 4, 6], [3, 6, 9], [5, 10, 15]], dtype=torch.float)  # Changed to float
targets = torch.tensor([8, 12, 20], dtype=torch.float)

# Normalize inputs for better training (optional, can remove if results are off)
data_mean, data_std = data.mean(), data.std()
data = (data - data_mean) / data_std
targets = (targets - data_mean) / data_std

# Move to device
data = data.to(device)
targets = targets.to(device)

# Initialize model
model = NumberPredictor(input_size=3).to(device)  # 3 input numbers per sequence
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Training loop
model.train()
for epoch in range(2000):
    optimizer.zero_grad()
    output = model(data)
    loss = loss_fn(output.squeeze(), targets)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# Test prediction
model.eval()
with torch.no_grad():
    test_data = torch.tensor([[7, 14, 21]], dtype=torch.float)
    test_data = (test_data - data_mean) / data_std  # Apply same normalization
    test_data = test_data.to(device)
    prediction = model(test_data).item()
    # Denormalize prediction
    prediction = prediction * data_std + data_mean
    print(f"Predicted next number: {prediction:.2f}")
    print(f"Expected number: 28")
    print(f"Model is on: {next(model.parameters()).device}")