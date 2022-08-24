# Support Vector Machine – SVM From Scratch Python

![Support Vector Machine – SVM From Scratch Python](https://user-images.githubusercontent.com/40186859/186307013-a414ceb5-c413-47c8-8979-b5679160cc80.png)

In the 1960s, Support vector Machine (SVM) known as supervised machine learning classification was first developed, and later refined in the 1990s which has become extremely popular nowadays owing to its extremely efficient results. The SVM is a supervised algorithm is capable of performing classification, regression, and outlier detection. But, it is widely used in classification objectives. SVM is known as a fast and dependable classification algorithm that performs well even on less amount of data. Let’s begin today’s tutorial on SVM from scratch python.

![SVM](https://user-images.githubusercontent.com/40186859/186307570-c7c4c74f-c1ee-4b9a-bb27-c8bc84a740a7.png)

## HOW SVM WORKS ?

SVM finds the best N-dimensional hyperplane in space that classifies the data points into distinct classes. Support Vector Machines uses the concept of ‘Support Vectors‘, which are the closest points to the hyperplane. A hyperplane is constructed in such a way that distance to the nearest element(support vectors) is the largest. The better the gap, the better the classifier works.

<img src="https://user-images.githubusercontent.com/40186859/186307734-76e3ac6c-85de-45a3-9a43-be8da8d8cf7f.png" alt = "Selecting Hyperplane with Greater Gap" width="450">

The line (in 2 input feature) or plane (in 3 input feature) is known as a decision boundary. Every new data from test data will be classified according to this decision boundary. The equation of the hyperplane in the ‘M’ dimension:

$y = w_0 + w_1x_1 + w_2x_2 + w_3x_3 + ......= w_0 + \Sigma_{i=1}^mw_ix_i = w_0 + w^{T}X = b + w^{T}X $

$a$

<!-- $b = biased term (w_0)$ -->
<!-- $X = variables$ -->
