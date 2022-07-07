# LINEAR REGRESSION FROM SCRATCH

![salary_prediction_best_fit_line](https://user-images.githubusercontent.com/40186859/177680055-5a2d5497-98b6-4f98-9913-ffab67cc7a19.png)

Regression is the method which measures the average relationship between two or more continuous variables in term of the response variable and feature variables. In other words, regression analysis is to know the nature of the relationship between two or more variables to use it for predicting the most likely value of dependent variables for a given value of independent variables. Linear regression is a mostly used regression algorithm.

For more concrete understanding, let’s say there is a high correlation between day temperature and sales of tea and coffee. Then the salesman might wish to know the temperature for the next day to decide for the stock of tea and coffee. This can be done with the help of regression.

The variable, whose value is estimated, predicted, or influenced is called a dependent variable. And the variable which is used for prediction or is known is called an independent variable. It is also called explanatory, regressor, or predictor variable.

## 1. LINEAR REGRESSION

Linear Regression is a supervised method that tries to find a relation between a continuous set of variables from any given dataset. So, the problem statement that the algorithm tries to solve linearly is to best fit a line/plane/hyperplane (as the dimension goes on increasing) for any given set of data.

This algorithm use statistics on the training data to find the best fit linear or straight-line relationship between the input variables (X) and output variable (y). Simple equation of Linear Regression model can be written as:

```
Y=mX+c ; Here m and c are calculated on training
```

In the above equation, m is the scale factor or coefficient, c being the bias coefficient, Y is the dependent variable and X is the independent variable. Once the coefficient m and c are known, this equation can be used to predict the output value Y when input X is provided.

Mathematically, coefficients m and c can be calculated as:

```
m = sum((X(i) - mean(X)) * (Y(i) - mean(Y))) / sum( (X(i) - mean(X))^2)
c = mean(Y) - m * mean(X)
```
![reg_error](https://user-images.githubusercontent.com/40186859/177681315-7233aae2-97e0-4f1a-9bcd-56a3b88ea148.png)

As you can see, the red point is very near the regression line; its error of prediction is small. By contrast, the yellow point is much higher than the regression line and therefore its error of prediction is large. The best-fitting line is the line that minimizes the sum of the squared errors of prediction.

## 1.1. LINEAR REGRESSION FROM SCRATCH

We will build a linear regression model to predict the salary of a person on the basis of years of experience from scratch. You can download the dataset from the link given below. Let’s start with importing required libraries:

```
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

We are using dataset of 30 data items consisting of features like years of experience and salary. Let’s visualize the dataset first.

```
dataset = pd.read_csv('salaries.csv')

#Scatter Plot
X = dataset['Years of Experience']
Y = dataset['Salary']

plt.scatter(X,Y,color='blue')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary Prediction Curves')
plt.show()
```
![salary_prediction_curve](https://user-images.githubusercontent.com/40186859/177685995-af8c1fc9-9145-4337-81b4-03159324d71a.png)

```
def mean(values):
    return sum(values) / float(len(values))

# initializing our inputs and outputs
X = dataset['Years of Experience'].values
Y = dataset['Salary'].values

# mean of our inputs and outputs
x_mean = mean(X)
y_mean = mean(Y)

#total number of values
n = len(X)

# using the formula to calculate the b1 and b0
numerator = 0
denominator = 0
for i in range(n):
    numerator += (X[i] - x_mean) * (Y[i] - y_mean)
    denominator += (X[i] - x_mean) ** 2
    
b1 = numerator / denominator
b0 = y_mean - (b1 * x_mean)

#printing the coefficient
print(b1, b0)
```
Finally, we have calculated the unknown coefficient m as b1 and c as b0. Here we have b1 = **9449.962321455077** and b0 = **25792.20019866869**.

Let’s visualize the best fit line from scratch.

![salary_prediction_best_fit_line_scratch](https://user-images.githubusercontent.com/40186859/177680348-ded24709-9e35-4c96-882e-f429423e7c17.png)

Now let’s predict the salary Y by providing years of experience as X:

```
def predict(x):
    return (b0 + b1 * x)
y_pred = predict(6.5)                      
print(y_pred)

Output: 87216.95528812669
```
### 1.2. LINEAR REGRESSION USING SKLEARN

```
from sklearn.linear_model import LinearRegression

X = dataset.drop(['Salary'],axis=1)                
Y = dataset['Salary'] 

reg = LinearRegression()  #creating object reg
reg.fit(X,Y)     # Fitting the Data set
```

Let’s visualize the best fit line using Linear Regression from sklearn

![salary_prediction_best_fit_line](https://user-images.githubusercontent.com/40186859/177680055-5a2d5497-98b6-4f98-9913-ffab67cc7a19.png)

Now let’s predict the salary Y by providing years of experience as X:

```
y_pred = reg.predict([[6.5]])  
y_pred

Output: 87216.95528812669
```

## 1.3. CONCLUSION

We need to able to measure how good our model is (accuracy). There are many methods to achieve this but we would implement Root mean squared error and coefficient of Determination (R² Score).

* Try Model with Different error metric for Linear Regression like Mean Absolute Error, Root mean squared error.
* Try algorithm with large data set, imbalanced & balanced dataset so that you can have all flavors of Regression.

