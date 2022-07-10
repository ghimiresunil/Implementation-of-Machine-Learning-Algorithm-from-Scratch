# LOGISTIC REGRESSION FROM SCRATCH
![image](https://user-images.githubusercontent.com/40186859/178145171-645eaa3d-8070-4bec-9d64-136f52ac1cd5.png)

Before begin, let's assume you are already familar with some of the following topics:

* Classification and Regression in Machine Learning
* Binary Classification and Multi-class classification
* Basic Geometry of Line, Plane, and Hyper-Plane in 2D, 3D, and n-D space respectively
* Maxima and Minima
* Loss Function

## 1. WHAT IS LOGISTIC REGRESSION?

In simple Terms, Logistic Regression is a Classification technique, which is best used for Binary Classification Problems. Logistic regression is used to describe data and to explain the relationship between one dependent binary variable and one or more independent variables.

Let’s say, we have a Binary Classification problem, which has only 2 classes true or false. Imagine we want to detect whether a credit card transaction is genuine or fraudulent.

`+1 = Genuine & -1 = Fraudulent`

Firstly, let’s try to plot our data points in a 2D space, considering we can visualize 2D in a better way here.

![image](https://user-images.githubusercontent.com/40186859/178106641-5ad0b866-e8de-47fb-a53b-c8c4121f3af1.png)

### 1.1 HOW DO WE DIFFERENTIATE DATA POINTS ?

As a human being, if we show this image to a little child with no knowledge of Maths or graph and ask them to differentiate between the two points, I am pretty much sure that he will use his common sense and will draw a line in-between the two points which will differentiate them. That line is called a Decision Boundary, as shown below:

![image](https://user-images.githubusercontent.com/40186859/178106757-115ee400-31b1-4ecb-8985-d78dea7034c9.png)

Anything on the left side of the boundary is Fraudulent and anything on the right side of the boundary is Genuine. This is a common-sense approach.

### 1.2. HOW TO BUILD BOUNDARY?

Now, if we look at it from a Geometric perspective, then the decision boundary is nothing but a simple Line in 2D space. And we know that line also has an equation that can be used to represent it. Now a line can be represented using 2 unique variables, `“m”` and `“b”`:

In two dimensions, the equation for non-vertical lines is often given in the slope-intercept form: `y = mx + b` where:

m is the slope or gradient of the line. <br>
b is the y-intercept of the line. <br>
x is the independent variable of the function y = f(x). <br>

If, our line passes through origin then b = 0, then y = mx

In 3D space we have equation of Plane as:

`Ax + By + Cz = D`

If line passes through origin `D=0`, then $ Z = (-)\frac{a}{c}*x + (-)\frac{b}{c} * y$

Now, this is about 2D and 3D space, in n-D space we have Hyper-Plane instead of Line or Plane.

### 1.3. HOW DO WE REPRESENT HYPERPLANE IN EQUATION ?

Before diving into the topic, a question may arise in your mind: What is a hyperplane? In a concise term, a hyperplane is a decision boundary that helps classify the data points.

Let us consider a line with multiple points on it.

For example: If we want to divide a line into two parts such as some points lie on one side and remaining on the other side, we can choose a point as reference.

So for a line a point is hyperplane.

For next example, lets take a wall (2D) . If we want to partition it into two parts a single line (1D) can do that.

Thus if V is an n-dimensional vector space than hyperplane for V is an (n-1)-dimensional subspace.

<i>**Note: Data points falling on either side of the hyperplane can be attributed to different classes.**</i>

Let's deep dive into the topic, imagine we have n-dimension in our Hyper-Plane, then each dimension will have its own slope value, let’s call them 

$w_1, w_2, w_3, ........, w_n$

and let the dimensions we represented as 

$x_1, x_2, x_3, ........, x_n$

and lets have an intercept term as `“b”`. So, the Hyper-Plane equation holds as:

$w_1x_1, w_2x_2, w_3x_3, ........, w_nx_n$

We, can represent it as:

Mathematically, it is represented as $ b + \Sigma_{i=1}^{0} {w_ix_i} = 0$ 

Now, If we consider, $w_i$ as a vector and $x_i$ as another vector, then we can represent ∑$ (w_ix_i)$ as a vector multiplication of 2 different vectors as $w_i^{\ T}x_i$  where $w_i^{\ T}$ is the transposed vector of values represented using all $w_i$ and $x_i$  is the vector represented using all values of $x_i$ . Also, `b` here is a scalar value.

Considering all these, $w_i$ is normal to our Hyper-Plane then only the multiplication would make sense. This means that `w` is Normal to our Hyper-Plane and as we all know, Normal is always perpendicular to our surface.

So now, coming back to our model, after all that maths, we conclude that our model needs to learn the Decision Boundary, using 2 important things, 

For simplicity, let’s consider our plane passes through origin. Then `b=0`.

Now, our model needs to only figure out the values of $w_i$ , which is normal to our Hyper-Plane. The values in our normal is a vector, which means it is a set of values which needs to be found which best suits our data. So the main task in LR boils down to a simple problem of finding a decision boundary, which is a hyperplane, which is represented using ($w_i$ , b) given a Dataset of (+ve, -ve) points that best separate the points.


Let’s imagine that $w_i$ is derived, so even after that how are we going to find out the points whether they are of +1 or -1 class. For that let’s consider the below Diagram:

![image](https://user-images.githubusercontent.com/40186859/178108047-edbd1643-9b1b-41db-905c-aab0d0662a8e.png)

So, If you know the basics of ML problems, it can be explained as given a set of $x_i$ we have to predict $y_i$. So $y_i$ here belongs to the set {+1, -1}. Now, if we want to calculate the value of $d_i$ or $d_j$ we can do it with the below formula:

$d_i = \frac{w^{\ T}x_i}{||w||}$

where w is the normal vector to the out hyperplane, let’s assume that it is Unit Vector for simplicity. Therefore. `||W|| = 1`. Hence, $ \ d_i = w^{\ T}x_i$ and $d_j = w^{\ T}x_j$. Since w and x are on the same side the Hyper-Plane i.e. on the positive side, hence $w^{\ T}x_i$ > 0 and $w^{\ T}x_j$ < 0.

Basically, it means $d_i$ belongs to +1 class and $d_j$ belongs to -1 class. And, this is how we can classify our data points using the Hyper-Plane.

### 1.4. HOW DO WE CALCULATE OUT HYPERPLANE ?

![image](https://user-images.githubusercontent.com/40186859/178108889-5b85097b-9343-4710-a5d3-404818f86839.png)

Well, if you have heard something about optimization problems, our model finds the Hyper-Plane as an optimization problem. Before that, we have to create an optimization equation. Let’s Consider a few cases of $y_i$:

#### Case 01:
If a point is Positive, and we predict as Positive, then

$y_i$ = +1  and  $w^{\ T}x_i$ = +1, then  
$y_i*w^{\ T}x_i$  > 0

#### Case 02:
If a point is Negative, and we predict as Negative, then

$y_i$ = -1  and  $w^{\ T}x_i$ = -1, then  
$y_i*w^{\ T}x_i$  > 0

#### Case 03:
If a point is Positive, and we predict as Negative, then

$y_i$ = +1  and  $w^{\ T}x_i$ = -1, then  
$y_i*w^{\ T}x_i$  < 0

#### Case 04:
If a point is Negative, and we predict as Positive, then

$y_i$ = -1  and  $w^{\ T}x_i$ = +1, then  
$y_i*w^{\ T}x_i$  < 0

Now, if we look closely whenever we made a correct prediction our equation of $y_i*w^{\ T}x_i$ is always positive, irrespective of the cardinality of our data point. Hence our Optimization equation holds, as such

(Max w) $\Sigma_{i=1}^{0}(y_i*w^{\ T}x_i$) > 0

Let’s try to understand what the equation has to offer, the equation says that find me a w (the vector normal to our Hyper-Plane) which has a maximum  of $(y_i*w^{\ T}x_i$) > 0 such that the value of `“i”` ranges from `1 to n`, where `“n”` is the total number of dimensions we have.

It means, for which ever Hyperplane, we have maximum correctly predicted points we will choose that.

### 1.5. HOW DO WE SOLVE THE OPTIMIZATION PROBLEM TO FIND THE OPTIMAL W WHICH HAS THE MAX CORRECTLY CLASSIFIED POINT?

Logistic Regression uses Logistic Function. The logistic function also called the sigmoid function is an S-shaped curve that will take any real-valued number and map it into a worth between 0 and 1, but never exactly at those limits.

![image](https://user-images.githubusercontent.com/40186859/178110203-a43f6684-146e-4c0d-af72-9087e083c9eb.png)


$\sigma(t) = \frac{e^t}{e^t + 1} = \frac{1}{1 + e^{-t}}$

So we use our optimization equation in place of “t”

t = $y_i*w^{\ T}x_i$ s.t. (i = {1,n})

And when we solve this sigmoid function, we get another optimization problem, which is computationally easy to solve. The end Optimization equation becomes as below:

w* = (Argmin w) ∑$log_n(1  +  e^{-t})$ 

So, our equation changes form finding a Max to Min, now we can solve this using optimizer or a Gradient Descent. Now, to solve this equation we use something like Gradient Descent, intuitively it tries to find the minima of a function. In our case, it tries to find the minima of out sigmoid function.

### 1.6 HOW DOES IT MINIMIZE A FUNCTION OR FINDS MINIMA?

Our Optimizer tries to minimize the loss function of our sigmoid, by loss function I mean, it tries to minimize the error made by our model, and eventually finds a Hyper-Plane which has the lowest error. The loss function has the below equation: 

$[y*log(y_p) + (i - y)*log(1 - y_p)]$

where,
y = actual class value of a data point <br>
$y_p$ = predicted class value of data point

And so this is what Logistic Regression is and that is how we get our best Decision Boundary for classification. In broader sense, Logistic Regression tries to find the best decision boundary which best separates the data points of different classes.

### 1.7. CODE FROM SCRATCH

Before that, let’s re-iterate over few key points, so the code could make more sense to us:

X is a set of data points, with m rows and n dimensions. <br>
y is a set of class which define a class for every data point from X as +1 or -1

z = $w^{\ T}X_i$ <br>
w = Set of values for a vector that forms the Normal to our Hyper-Plane <br>
b = Set of scalars of the intercept term, not required if our Hyper-Plane passes through the origin <br>
$y_p$ = predicted value of Xi,  from the sigmoid function <br>
 
Intuitively speaking, our model tries to learn from each iteration using something called a learning rate and gradient value, think this as, once we predict the value using the sigmoid function, we get some values of $y_p$ and then we have y.

We calculate error, and then we try to use the error to predict a new set of `w` values, which we use to repeat the cycle, until we finally find the best value possible.

In today’s code from scratch, I will be working on Iris dataset. So let’s dive into the code

```
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn import linear_model

iris = datasets.load_iris()
X = iris.data[:, :2]     #we use only 2 class
y = (iris.target != 0) * 1
```

Let’s try to plot and see how our data lies. Whether can it be separated using a decision boundary.

```
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', label='0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='1')
plt.legend();
```

![image](https://user-images.githubusercontent.com/40186859/178110611-1cb424a7-b966-4cc7-96f4-faf4ce4265aa.png)

```
class LogisticRegression:
    
    # defining parameters such as learning rate, number ot iterations, whether to include intercept, 
    # and verbose which says whether to print anything or not like, loss etc.
    def __init__(self, learning_rate=0.01, num_iterations=50000, fit_intercept=True, verbose=False):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.fit_intercept = fit_intercept
        self.verbose = verbose
    
    # function to define the Incercept value.
    def __b_intercept(self, X):
        # initially we set it as all 1's
        intercept = np.ones((X.shape[0], 1))
        # then we concatinate them to the value of X, we don't add we just append them at the end.
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid_function(self, z):
        # this is our actual sigmoid function which predicts our yp
        return 1 / (1 + np.exp(-z))
    
    def __loss(self, yp, y):
        # this is the loss function which we use to minimize the error of our model
        return (-y * np.log(yp) - (1 - y) * np.log(1 - yp)).mean()
    
    # this is the function which trains our model.
    def fit(self, X, y):
        
        # as said if we want our intercept term to be added we use fit_intercept=True
        if self.fit_intercept:
            X = self.__b_intercept(X)
        
        # weights initialization of our Normal Vector, initially we set it to 0, then we learn it eventually
        self.W = np.zeros(X.shape[1])
        
        # this for loop runs for the number of iterations provided
        for i in range(self.num_iterations):
            
            # this is our W * Xi
            z = np.dot(X, self.W)
            
            # this is where we predict the values of Y based on W and Xi
            yp = self.__sigmoid_function(z)
            
            # this is where the gradient is calculated form the error generated by our model
            gradient = np.dot(X.T, (yp - y)) / y.size
            
            # this is where we update our values of W, so that we can use the new values for the next iteration
            self.W -= self.learning_rate * gradient
            
            # this is our new W * Xi
            z = np.dot(X, self.W)
            yp = self.__sigmoid_function(z)
            
            # this is where the loss is calculated
            loss = self.__loss(yp, y)
            
            # as mentioned above if we want to print somehting we use verbose, so if verbose=True then our loss get printed
            if(self.verbose ==True and i % 10000 == 0):
                print(f'loss: {loss} \t')
    
    # this is where we predict the probability values based on out generated W values out of all those iterations.
    def predict_prob(self, X):
        # as said if we want our intercept term to be added we use fit_intercept=True
        if self.fit_intercept:
            X = self.__b_intercept(X)
        
        # this is the final prediction that is generated based on the values learned.
        return self.__sigmoid_function(np.dot(X, self.W))
    
    # this is where we predict the actual values 0 or 1 using round. anything less than 0.5 = 0 or more than 0.5 is 1
    def predict(self, X):
        return self.predict_prob(X).round()
```

Let’s train the model by creating a class of it, we will give Learning rate as 0.1 and number of iterations as 300000.

```
model = LogisticRegression(learning_rate=0.1, num_iterations=300000)
model.fit(X, y)
```

Lets us see how well our prediction works:

```
preds = model.predict(X)
(preds == y).mean()

Output: 1.0
```

```
plt.figure(figsize=(10, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', label='0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='1')
plt.legend()
x1_min, x1_max = X[:,0].min(), X[:,0].max(),
x2_min, x2_max = X[:,1].min(), X[:,1].max(),
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
grid = np.c_[xx1.ravel(), xx2.ravel()]
probs = model.predict_prob(grid).reshape(xx1.shape)
plt.contour(xx1, xx2, probs, [0.5], linewidths=1, colors='black');
```

### 1.8. LOGISTIC REGRESSION FROM SCIKIT LEARN

```
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
model = LogisticRegression(solver='liblinear', random_state=0)
model.fit(X, y)
model.predict_proba(X)
```

```
cm = confusion_matrix(y, model.predict(X))

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='white')
plt.show()
```

```
print(classification_report(y, model.predict(X)))
```
