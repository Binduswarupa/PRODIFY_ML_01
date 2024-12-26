# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Create a dataset or load an existing dataset
# Example data for houses (in reality, you would load data from a CSV or a database)
data = {
    'Square_Feet': [1500, 1800, 2400, 3000, 3500, 4000],
    'Bedrooms': [3, 3, 4, 4, 5, 5],
    'Bathrooms': [2, 2, 3, 3, 4, 4],
    'Price': [400000, 450000, 500000, 600000, 650000, 700000]
}

# Create a pandas DataFrame
df = pd.DataFrame(data)

# Step 3: Split the data into features (X) and target (y)
X = df[['Square_Feet', 'Bedrooms', 'Bathrooms']]  # Features
y = df['Price']  # Target

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Initialize the Linear Regression model
model = LinearRegression()

# Step 6: Train the model
model.fit(X_train, y_train)

# Step 7: Make predictions on the test data
y_pred = model.predict(X_test)

# Step 8: Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Output the results
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R^2 Score: {r2}')

# Step 9: Visualizing the predicted vs actual prices
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # line of perfect prediction
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices')
plt.show()
