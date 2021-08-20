###Linear Regression###
#Exercise6

import numpy as np # import numerical functions
from sklearn.linear_model import LinearRegression # import linear regression
from sklearn.metrics import mean_squared_error # import mean squared error
from matplotlib import pyplot as plt # import plotting tool
#Quadratic regression
# Holly
X = np.array([1940, 1945, 1950, 1955, 1960, 1965, 1970, 1975, 1980, 1985, 1990, 1995,2000]).reshape((-1,1))
Y1 = np.array([0.063, 0.284, 0.475, 1.248, 1.240, 1.836, 2.693, 3.235, 2.968, 3.037,1.837,1.325,0.848])
X_quad = np.hstack((X, X*X))
quadratic_model = LinearRegression()
quadratic_model.fit(X_quad, Y1)
plt.scatter(X,Y1)
plt.plot(X, quadratic_model.predict(X_quad), color='red')
plt.show()
print(quadratic_model.predict([[2005, 2005*2005]]))
#6.4-
print(quadratic_model.predict([[2005, 2005*2005], [2010, 2010*2010],[2015, 2015*2015], [2020, 2020*2020]]))
#Answer is [0.42506294] [ 0.42506294 -0.36916284 -1.28802897 -2.33153546]
# Problem 6.1,6.3
# Quadratic regression
# Henry
Y2 = np.array([5.749, 4.811, 3.835, 2.966, 2.293, 2.006, 1.638, 1.282, 1.113, 1.071, 1.039,1.253,1.492])
X_quad = np.hstack((X, X*X))
quadratic_model = LinearRegression()
quadratic_model.fit(X_quad, Y2)
plt.scatter(X,Y2)
plt.plot(X, quadratic_model.predict(X_quad), color='red')
plt.show()
print(quadratic_model.predict([[2005, 2005*2005]]))
#6.4
print(quadratic_model.predict([[2005, 2005*2005],[2010, 2010*2010],[2015, 2015*2015], [2020, 2020*2020]]))
#Answer is [1.94086014] [1.94086014 2.46101998 3.0968971  3.84849151]
#6.2- I chose a quadratic model because the data points look like a curve
#6.3-My model predicts 0.425 frequency of the name Holly and 1.941 frequency of the name Henry. The model was good and accurate as it was close to the actual values which are 0.510 and 1.964 which makes sense as the data we have looks quadratic and we are not predicting very far off the data we have as we have 1940-2000 and we are only predicting 5 years after.
#6.4- The quadratic regression model predicts frequencies of -0.36916284 -1.28802897 -2.33153546 for the name of Holly in the remaining shortcomings year of 2010,2015 and 2020 and frequencies of 2.46101998 3.0968971  3.84849151 for the name Henry for the year 2010,2015 and 2020
