# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Initialize weights randomly.

2.Compute predicted values.

3.Compute gradient of loss function.

4.Update weights using gradient descent.


## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: MOHAMED HAMEEM SAJITH J
RegisterNumber:  212223240090
*/

mport numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
    # Add a column of ones to X for the intercept term
    X = np.c_[np.ones(len(X1)), X1]

    # Initialize theta with zeros
    theta = np.zeros(X.shape[1]).reshape(-1,1)

    # Perform gradient descent
    for _ in range(num_iters):
        # Calculate predictions
        predictions = X.dot(theta).reshape(-1, 1)

        # Calculate errors
        errors = (predictions - y).reshape(-1, 1)

        # Update theta using gradient descent
        theta = theta - learning_rate * (1 / len(X1)) * X.T.dot(errors)

    return theta

data = pd.read_csv('50_Startups.csv') # Assuming the data has a header

# Assuming the last column is your target variable 'y' and the predictors are all columns except the last one
X1 = data.iloc[:, :-1].values.astype(float)
scaler = StandardScaler()

# Reshape y to a column vector
y = data.iloc[:, -1].values.reshape(-1, 1)

# Scale the features
X1_scaled = scaler.fit_transform(X1)
y_scaled = scaler.fit_transform(y)

# Learn model parameters
theta = linear_regression(X1_scaled, y_scaled)

# Predict target value for a new data point
new_data = np.array([165349.2, 136897.8, 471784.1]).reshape(1, -1)
new_scaled = scaler.transform(new_data)
new_scaled_with_intercept = np.append(1, new_scaled)  # Add intercept term
prediction = np.dot(new_scaled_with_intercept, theta)

# Inverse transform the prediction to get the original scale
prediction = scaler.inverse_transform(prediction.reshape(-1, 1))
print(f"Predicted value: {prediction}")
```

## Output:
![image](https://github.com/MOHAMED-HAMEEM-SAJITH-J/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/162780573/c6271841-17c4-4c26-b074-519906eb93e8)

![image](https://github.com/MOHAMED-HAMEEM-SAJITH-J/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/162780573/8b7f8da3-35db-4b56-a2f9-4619927b85ae)

![image](https://github.com/MOHAMED-HAMEEM-SAJITH-J/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/162780573/618376a0-e82a-47cb-b6d3-7ffd142cb562)

![image](https://github.com/MOHAMED-HAMEEM-SAJITH-J/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/162780573/215dff2b-9ec2-471c-b32a-7ac91377076f)

![image](https://github.com/MOHAMED-HAMEEM-SAJITH-J/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/162780573/fac7d12c-d7d9-4710-88ec-3a41e5e3577a)





![linear regression using gradient descent](sam.png)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
