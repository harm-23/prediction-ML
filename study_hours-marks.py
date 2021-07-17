
# Importing the relevant libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set()


# Readinag the data
url = "http://bit.ly/w-data"
data = pd.read_csv(url)
data.head() # Shows first 5 rows of the data


# Plotting the data
data.plot(x="Hours", y="Scores", style = 'o')
plt.title('Hours vs percentage')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()


# ### **preparing the data**

# Slicing the data
X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values


# Using the train test split method to divide the data and train the model
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 


# Fitting the regression line
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# Predicting the test data values
print(X_test)
y_pred = regressor.predict(X_test)


# Making the dataframe of actual test values and predicted test values
df = pd.DataFrame({'actual':y_test, 'predicted':y_pred})
df


# predicting for 9.25 hours
hour = np.array([9.25])
hours = hour.reshape(1,-1)
predd = regressor.predict(hours)
print(predd)


# The error of the model
from sklearn import metrics
err = metrics.mean_absolute_error(y_test, y_pred)
print(err)

