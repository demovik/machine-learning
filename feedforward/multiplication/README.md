# Simple Machine Learning Model (MLM) 

Small MLM just for multiplication from 0 to 9, and possibly predictions.

**Install the required Python libraries:**

`pip install pandas scikit-learn torch numpy`

**Generate all possible multiplication pairs from 0×0 to 9×9**

`python generate_learn_data.py`

Output: `learn_data.csv`

**Train a simple polynomial regression model**

`python train_model.py`

Output: `multiplication_model.pkl`

**Test a simple polynomial regression model**

`python test_model.py`

**Train a simple neural network model**

`python train_nn.py`

Output: `multiplication_nn.pth`

**Test a simple neural network model**

`python test_nn.py`

