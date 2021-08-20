###Linear Regression###
#Exercise5

import numpy as np # import numerical functions
from sklearn.linear_model import LinearRegression # import linear regression
from sklearn.metrics import mean_squared_error # import mean squared error
from matplotlib import pyplot as plt # import plotting tool
# Part 5.1 Solution
# Let's now input the data from the table into Python
# Y1 is the relative frequencies of the name "Holly"
X = np.array([1940, 1945, 1950, 1955, 1960, 1965, 1970, 1975, 1980, 1985]).reshape((-1,1))
Y1 = np.array([0.063, 0.284, 0.475, 1.248, 1.240, 1.836, 2.693, 3.235, 2.968, 3.037])
# We can also view it as a scatter plot
plt.scatter(X,Y1)
# Then we build a linear model:
linear_model = LinearRegression()
linear_model.fit(X,Y1)
# And we also want to plot the predicted regression line:
plt.plot(X, linear_model.predict(X), color='red')
plt.show()
# And we can also print out the slope and intercept
print("Slope: ", linear_model.coef_[0])
print("Intercept:", linear_model.intercept_)

#Answer is: The best-fit is y= 0.0779x-151

# Y2 is the relative frequencies of the name "Henry"
X = np.array([1940, 1945, 1950, 1955, 1960, 1965, 1970, 1975, 1980, 1985]).reshape((-1,1))
Y2 = np.array([5.749, 4.811, 3.835, 2.966, 2.293, 2.006, 1.638, 1.282, 1.113, 1.071])
# We can also view it as a scatter plot
plt.scatter(X,Y2)
# Then we build a linear model:
linear_model = LinearRegression()
linear_model.fit(X,Y2)
# And we also want to plot the predicted regression line:
plt.plot(X, linear_model.predict(X), color='red')
plt.show()
# And we can also print out the slope and intercept
print("Slope: ", linear_model.coef_[0])
print("Intercept:", linear_model.intercept_)


#Answer is Slope:  -0.10305939393939392
        #Intercept: 204.93046060606056
        #[0.35756364]
        #[ 0.35756364 -1.70362424]
        # So The best fit is y= -0.103x+205
# Part 5.2-5.4 Solution
# Let's use the model we built above to predict the average salary in 2000 and 2017
print(linear_model.predict([[1985]]))
print(linear_model.predict([[1985],[2005]]))

#Answer is [3.46112727]
#[3.46112727 5.01955152]

print(linear_model.predict([[1985]]))
print(linear_model.predict([[1985],[2005]]))

#answer is [0.35756364]
#[ 0.35756364 -1.70362424]
#5.2- 3.461 is the model prediction of the frequency of the name Holly in the year 1985
#0.358 is the model prediction of the frequency of the name Henry in the year 1985
#5.3- 5.020 is the model prediction of frequency of the name Holly in the year 2005
#-1.704 is the model prediction of frequency of the name Henry in the year 2005
#5.4- 3.461 is a decent prediction as it is not too far of the actual frequency value which is 3.037. That makes sense because the Henry name frequencies look roughly linear However 0.358 is a bad prediction as it is far of the actual frequency value which is 1.071 which is makes sense as it looks like it is leveling off in the end
#Unfortunately, the prediction for the year 2005 which is 5.020 for Holly and - 1.704 for Henry is most likely a bad prediction as in this particular case, we only had data for the years 1940-1985, and we were trying to predict the salary 20 years into the future, in 2005. This extrapolation is significantly outside the training data range, so the estimation is most likely flawed.

