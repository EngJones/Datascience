import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

'Importing the dataset'
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

'Splitting the data for testing'
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1/3, random_state=0)

'Fitting Simple linear regression to the Training set'
regressor = LinearRegression()
regressor.fit(X_train, y_train)

'Predicting the Test set results'
y_pred = regressor.predict(X_test)

'Visualization of the training set results'
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

'Visualization of the test set results'
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

