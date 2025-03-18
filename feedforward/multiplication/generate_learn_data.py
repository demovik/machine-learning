import pandas as pd

# Create a dataset of all multiplication pairs from 0 to 9
data = []
for i in range(10):
    for j in range(10):
        data.append({"num1": i, "num2": j, "product": i * j})

# Convert to a DataFrame
df = pd.DataFrame(data)

# Save the dataset to a CSV file
df.to_csv("learn_data.csv", index=False)

print("Dataset saved as learn_data.csv")
