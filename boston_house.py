from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the California housing dataset
california = fetch_california_housing()
X = california.data
y = california.target

#created the file.
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the house prices on the test set
y_pred = model.predict(X_test)

# Calculate the model accuracy (R^2 score)
accuracy = r2_score(y_test, y_pred)

print(f"Model Accuracy (R^2 Score): {accuracy}")

result_file_path = 'result.txt'

# Append the simulated accuracy to the file
with open(result_file_path, 'w') as file:
    file.write("Accuracy = " + str(accuracy))

# added acuracy
    

