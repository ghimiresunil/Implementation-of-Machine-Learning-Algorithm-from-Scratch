# Support Vector Machine – SVM From Scratch Python

![Support Vector Machine – SVM From Scratch Python](https://user-images.githubusercontent.com/40186859/186307013-a414ceb5-c413-47c8-8979-b5679160cc80.png)

In the 1960s, Support vector Machine (SVM) known as supervised machine learning classification was first developed, and later refined in the 1990s which has become extremely popular nowadays owing to its extremely efficient results. The SVM is a supervised algorithm is capable of performing classification, regression, and outlier detection. But, it is widely used in classification objectives. SVM is known as a fast and dependable classification algorithm that performs well even on less amount of data. Let’s begin today’s tutorial on SVM from scratch python.

![SVM](https://user-images.githubusercontent.com/40186859/186307570-c7c4c74f-c1ee-4b9a-bb27-c8bc84a740a7.png)

## HOW SVM WORKS ?

SVM finds the best N-dimensional hyperplane in space that classifies the data points into distinct classes. Support Vector Machines uses the concept of ‘Support Vectors‘, which are the closest points to the hyperplane. A hyperplane is constructed in such a way that distance to the nearest element(support vectors) is the largest. The better the gap, the better the classifier works.

<img src="https://user-images.githubusercontent.com/40186859/186307734-76e3ac6c-85de-45a3-9a43-be8da8d8cf7f.png" alt = "Selecting Hyperplane with Greater Gap" width="450">

The line (in 2 input feature) or plane (in 3 input feature) is known as a decision boundary. Every new data from test data will be classified according to this decision boundary. The equation of the hyperplane in the ‘M’ dimension:

$y = w_0 + w_1x_1 + w_2x_2 + w_3x_3 + ......= w_0 + \Sigma_{i=1}^mw_ix_i = w_0 + w^{T}X = b + w^{T}X $

where, <br>
$W_i$ = $vectors\(w_0, w_1, w_2, w_3, ...... w_m\)$

![Hyperplane Function h](https://user-images.githubusercontent.com/40186859/187906252-5809efe7-c7c6-4c51-877f-f4e1cb23d545.png)

The point above or on the hyperplane will be classified as class +1, and the point below the hyperplane will be classified as class -1.

![Support Vector](https://user-images.githubusercontent.com/40186859/187906410-defb5694-152e-4094-a25c-a6ad11b02c0e.png)

## SVM IN NON-LINEAR DATA

SVM can also conduct non-linear classification.

![SVM in Linear Data](https://user-images.githubusercontent.com/40186859/187906503-26558f78-7f0c-4837-9053-32fd2adc755c.png)

For the above dataset, it is obvious that it is not possible to draw a linear margin to divide the data sets. In such cases, we use the kernel concept.

SVM works on mapping data to higher dimensional feature space so that data points can be categorized even when the data aren’t otherwise linearly separable. SVM finds mapping function to convert 2D input space into 3D output space. In the above condition, we start by adding Y-axis with an idea of moving dataset into higher dimension.. So, we can draw a graph where the y-axis will be the square of data points of the X-axis.

![Increasing Dimension of Data](https://user-images.githubusercontent.com/40186859/187906570-1ca7370d-3065-4345-a58d-0c65a0c05b15.png)

And now, the data are two dimensional, we can draw a Support Vector Classifier that classifies the dataset into two distinct regions. Now, let’s draw a support vector classifier.

![upport Vector Classifier](https://user-images.githubusercontent.com/40186859/187906664-46375425-a7f2-4a4e-86cb-0a9c6fdba9cf.png)

**_Note_**: This example is taken from [Statquest](https://www.youtube.com/watch?v=efR1C6CvhmE).

## HOW TO TRANSFORM DATA ??

SVM uses a kernel function to draw Support Vector Classifier in a higher dimension. Types of Kernel Functions are :

- Linear
- Polynomial
- Radial Basis Function(rbf)

In the above example, we have used a polynomial kernel function which has a parameter d (degree of polynomial). Kernel systematically increases the degree of the polynomial and the relationship between each pair of observation are used to find Support Vector Classifier. We also use cross-validation to find the good value of d.

### Radial Basis Function Kernel

Widely used kernel in SVM, we will be discussing radial basis Function Kernel in this tutorial for SVM from Scratch Python. Radial kernel finds a Support vector Classifier in infinite dimensions. Radial kernel behaves like the Weighted Nearest Neighbour model that means closest observation will have more influence on classifying new data.

$K(X_1, X_2) = exponent(-\gamma||X_1 - X_2||^2)$

Where, <br>
$||X_1 - X_2||$ = Euclidean distance between $X_1$ & $X_2$

## SOFT MARGIN – SVM
n this method, SVM makes some incorrect classification and tries to balance the tradeoff between finding the line that maximizes the margin and minimizes misclassification. The level of misclassification tolerance is defined as a hyperparameter termed as a penalty- ‘C’.

For large values of C, the optimization will choose a smaller-margin hyperplane if that hyperplane does a better job of getting all the training points classified correctly. Conversely, a very small value of C will cause the optimizer to look for a larger-margin separating hyperplane, even if that hyperplane misclassifies more points. For very tiny values of C, you should get misclassified examples, often even if your training data is linearly separable.

Due to the presence of some outliers, the hyperplane can’t classify the data points region correctly. In this case, we use a soft margin & C hyperparameter.

## SVM IMPLEMENTATION IN PYTHON
In this tutorial, we will be using to implement our SVM algorithm is the Iris dataset. You can download it from this [link](https://www.kaggle.com/code/jchen2186/machine-learning-with-iris-dataset/data). Since the Iris dataset has three classes. Also, there are four features available for us to use. We will be using only two features, i.e Sepal length, and Sepal Width.

![different kernel on Iris Dataset SVM](https://user-images.githubusercontent.com/40186859/187913255-110ac430-d9d6-4534-a014-22f8a5ecfa00.png)

## BONUS – SVM FROM SCRATCH PYTHON!!
Kernel Trick: Earlier, we had studied SVM classifying non-linear datasets by increasing the dimension of data. When we map data to a higher dimension, there are chances that we may overfit the model. Kernel trick actually refers to using efficient and less expensive ways to transform data into higher dimensions.

Kernel function only calculates relationship between every pair of points as if they are in the higher dimensions; they don’t actually do the transformation. This trick , calculating the high dimensional relationships without actually transforming data to the higher dimension, is called the **Kernel Trick**.
